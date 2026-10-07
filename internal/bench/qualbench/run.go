package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Run is one question answered under one condition.
type Run struct {
	Condition  string         `json:"condition"`
	QuestionID string         `json:"question_id"`
	Repeat     int            `json:"repeat"`
	Reply      string         `json:"reply"`
	ToolCalls  map[string]int `json:"tool_calls"`
	// Blocked counts calls to tools the condition withholds, which Claude Code rejected without running them.
	Blocked       map[string]int `json:"blocked,omitempty"`
	NumTurns      int            `json:"num_turns"`
	DurationMS    int64          `json:"duration_ms"`
	CostUSD       float64        `json:"cost_usd"`
	InputTokens   int            `json:"input_tokens"`
	CacheCreation int            `json:"cache_creation_input_tokens"`
	CacheRead     int            `json:"cache_read_input_tokens"`
	OutputTokens  int            `json:"output_tokens"`
	Model         string         `json:"model"`
	Error         string         `json:"error,omitempty"`
	Score         Score          `json:"score"`
}

// TotalInput is every input token the model processed, cached or not.
func (r Run) TotalInput() int { return r.InputTokens + r.CacheCreation + r.CacheRead }

// RunConfig describes the conditions:
//
//   - grep: the built-in Read, Grep and Glob tools only.
//   - chiton: the same built-ins plus chiton's MCP server, in a checkout where `chiton install` wrote its rules
//     and skills. The model chooses which tools to use, as it would in practice.
//   - chiton-only: chiton's MCP server plus Read, with Grep and Glob removed, so the answer has to come from the
//     graph. It measures what the graph itself can answer.
type RunConfig struct {
	Model     string
	GrepDir   string   // checkout used as the working directory without chiton
	ChitonDir string   // checkout used as the working directory with chiton
	GraphDirs []string // CHITON_PROJECT_DIR for each worker; each holds its own chiton-out/
	ChitonBin string
	Timeout   time.Duration
}

const preamble = "You are answering a question about the Go repository in the current working directory. Use the tools available to you to find the answer from the code; do not guess.\n\n"

// builtinTools is the same for grep and chiton, so chiton's MCP tools are the only difference between them.
const builtinTools = "Read,Grep,Glob"

// ask runs claude -p once and parses its stream-json transcript.
func ask(ctx context.Context, cfg RunConfig, condition string, worker int, q Question) Run {
	run := Run{Condition: condition, QuestionID: q.ID, ToolCalls: map[string]int{}}
	ctx, cancel := context.WithTimeout(ctx, cfg.Timeout)
	defer cancel()

	// The prompt goes on stdin: on Windows claude is often an npm .cmd shim, and cmd.exe truncates or
	// mangles an argument holding newlines and quotes.
	tools := builtinTools
	if condition == "chiton-only" {
		tools = "Read"
	}
	args := []string{
		"-p",
		"--output-format", "stream-json", "--verbose",
		"--model", cfg.Model,
		"--setting-sources", "project",
		"--strict-mcp-config",
		"--tools", tools,
		"--no-session-persistence",
	}
	dir := cfg.GrepDir
	allowed := tools
	switch condition {
	case "grep":
	case "chiton", "chiton-only":
		dir = cfg.ChitonDir
		mcpCfg, err := writeMCPConfig(cfg.ChitonBin, cfg.GraphDirs[worker%len(cfg.GraphDirs)], worker)
		if err != nil {
			run.Error = err.Error()
			return run
		}
		defer os.Remove(mcpCfg)
		args = append(args, "--mcp-config", mcpCfg)
		allowed += ",mcp__chiton"
	default:
		run.Error = "unknown condition " + condition
		return run
	}
	args = append(args, "--allowedTools", allowed)

	cmd := exec.CommandContext(ctx, "claude", args...)
	cmd.Dir = dir
	cmd.Stdin = strings.NewReader(preamble + q.Prompt)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	parseStream(out, &run)
	switch {
	case err != nil && run.Reply == "":
		run.Error = strings.TrimSpace(err.Error() + ": " + lastLine(stderr.String()))
	case run.Reply == "" && run.Error == "":
		run.Error = "no result from claude: " + lastLine(stderr.String())
	}
	return run
}

// writeMCPConfig writes a one-server MCP config for chiton serve over the worker's own copy of the graph.
func writeMCPConfig(bin, graphDir string, worker int) (string, error) {
	cfg := map[string]any{"mcpServers": map[string]any{"chiton": map[string]any{
		"command": bin,
		"args":    []string{"serve", "--stdio"},
		"env":     map[string]string{"CHITON_PROJECT_DIR": graphDir, "CHITON_PLAINTEXT_OK": "1"},
	}}}
	b, err := json.Marshal(cfg)
	if err != nil {
		return "", err
	}
	p := filepath.Join(os.TempDir(), fmt.Sprintf("qualbench-mcp-%d-%d.json", os.Getpid(), worker))
	return p, os.WriteFile(p, b, 0o600)
}

// parseStream reads claude's stream-json lines: tool_use blocks in assistant messages, and the final result.
func parseStream(out []byte, run *Run) {
	sc := bufio.NewScanner(bytes.NewReader(out))
	sc.Buffer(make([]byte, 1024*1024), 64*1024*1024)
	for sc.Scan() {
		var ev struct {
			Type    string `json:"type"`
			Message struct {
				Content []struct {
					Type    string          `json:"type"`
					Name    string          `json:"name"`
					IsError bool            `json:"is_error"`
					Content json.RawMessage `json:"content"`
				} `json:"content"`
			} `json:"message"`
			Subtype    string  `json:"subtype"`
			IsError    bool    `json:"is_error"`
			Result     string  `json:"result"`
			NumTurns   int     `json:"num_turns"`
			DurationMS int64   `json:"duration_ms"`
			CostUSD    float64 `json:"total_cost_usd"`
			Usage      struct {
				Input         int `json:"input_tokens"`
				CacheCreation int `json:"cache_creation_input_tokens"`
				CacheRead     int `json:"cache_read_input_tokens"`
				Output        int `json:"output_tokens"`
			} `json:"usage"`
			ModelUsage map[string]json.RawMessage `json:"modelUsage"`
		}
		if json.Unmarshal(sc.Bytes(), &ev) != nil {
			continue
		}
		switch ev.Type {
		case "assistant":
			for _, c := range ev.Message.Content {
				if c.Type == "tool_use" {
					run.ToolCalls[c.Name]++
				}
			}
		case "user":
			// A withheld tool comes back as an error result naming it; move it from ToolCalls to Blocked.
			for _, c := range ev.Message.Content {
				if c.Type != "tool_result" || !c.IsError {
					continue
				}
				_, rest, ok := strings.Cut(string(c.Content), "No such tool available: ")
				if !ok {
					continue
				}
				name, _, _ := strings.Cut(rest, ".")
				if run.ToolCalls[name] > 0 {
					run.ToolCalls[name]--
					if run.ToolCalls[name] == 0 {
						delete(run.ToolCalls, name)
					}
				}
				if run.Blocked == nil {
					run.Blocked = map[string]int{}
				}
				run.Blocked[name]++
			}
		case "result":
			run.Reply = ev.Result
			run.NumTurns = ev.NumTurns
			run.DurationMS = ev.DurationMS
			run.CostUSD = ev.CostUSD
			run.InputTokens = ev.Usage.Input
			run.CacheCreation = ev.Usage.CacheCreation
			run.CacheRead = ev.Usage.CacheRead
			run.OutputTokens = ev.Usage.Output
			for m := range ev.ModelUsage {
				if run.Model == "" || len(m) > len(run.Model) {
					run.Model = m
				}
			}
			if ev.IsError {
				run.Error = "claude reported an error: " + ev.Subtype
			}
		}
	}
}

func lastLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.LastIndex(s, "\n"); i >= 0 {
		return s[i+1:]
	}
	return s
}

// RunAll answers every question under every condition, repeat times, on parallel workers, and grades each run.
// progress is called after each run finishes.
func RunAll(ctx context.Context, cfg RunConfig, set QuestionSet, conditions []string, repeat, parallel int, progress func(Run, int, int)) []Run {
	type job struct {
		q         Question
		condition string
		repeat    int
	}
	var jobs []job
	for r := 1; r <= repeat; r++ {
		for _, q := range set.Questions {
			for _, c := range conditions {
				jobs = append(jobs, job{q, c, r})
			}
		}
	}
	ch := make(chan job)
	var mu sync.Mutex
	var runs []Run
	var wg sync.WaitGroup
	for w := range parallel {
		wg.Go(func() {
			for j := range ch {
				run := ask(ctx, cfg, j.condition, w, j.q)
				run.Repeat = j.repeat
				if run.Error == "" {
					run.Score = Grade(run.Reply, j.q.Expected, set.Module)
				}
				mu.Lock()
				runs = append(runs, run)
				n := len(runs)
				mu.Unlock()
				if progress != nil {
					progress(run, n, len(jobs))
				}
			}
		})
	}
	for _, j := range jobs {
		ch <- j
	}
	close(ch)
	wg.Wait()
	return runs
}
