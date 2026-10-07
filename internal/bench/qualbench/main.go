// Command qualbench is chiton's answer-quality benchmark. It asks Claude Code structural questions about a Go
// repository twice, once with only the built-in Read, Grep and Glob tools and once with chiton's MCP server
// added, and grades both answers against ground truth computed from the source with go/ast.
//
//	go run ./internal/bench/qualbench gen    -repo <checkout> -out questions.json
//	go run ./internal/bench/qualbench run    -questions questions.json -grep-dir <checkout> \
//	    -chiton-dir <checkout with chiton install> -chiton-bin <chiton> -graph-dirs <dir>[,<dir>...] -out results.json
//	go run ./internal/bench/qualbench report -questions questions.json -results results.json
//
// It needs the claude CLI on PATH, signed in. See docs/BENCHMARK.md for the full procedure.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: qualbench gen|run|report [flags]")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "gen":
		err = cmdGen(os.Args[2:])
	case "run":
		err = cmdRun(os.Args[2:])
	case "report":
		err = cmdReport(os.Args[2:])
	default:
		err = fmt.Errorf("unknown command %q", os.Args[1])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "qualbench:", err)
		os.Exit(1)
	}
}

func cmdGen(args []string) error {
	fs := flag.NewFlagSet("gen", flag.ExitOnError)
	repo := fs.String("repo", ".", "checkout to generate questions for (the directory holding go.mod)")
	out := fs.String("out", "questions.json", "where to write the question set")
	_ = fs.Parse(args)

	facts, err := Analyze(*repo)
	if err != nil {
		return err
	}
	set := QuestionSet{Module: facts.Module, Commit: gitHead(*repo), Questions: Generate(facts)}
	if err := writeJSON(*out, set); err != nil {
		return err
	}
	fmt.Printf("wrote %d questions for %s@%s to %s\n", len(set.Questions), set.Module, set.Commit, *out)
	return nil
}

func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	questions := fs.String("questions", "questions.json", "question set from qualbench gen")
	grepDir := fs.String("grep-dir", "", "checkout for the condition without chiton")
	chitonDir := fs.String("chiton-dir", "", "checkout for the condition with chiton (after chiton build and chiton install)")
	chitonBin := fs.String("chiton-bin", "chiton", "chiton binary the MCP server runs")
	graphDirs := fs.String("graph-dirs", "", "comma-separated CHITON_PROJECT_DIRs, one per parallel worker (default: -chiton-dir)")
	model := fs.String("model", "sonnet", "model passed to claude --model")
	conditions := fs.String("conditions", "grep,chiton", "conditions to run")
	repeat := fs.Int("repeat", 1, "times to ask each question under each condition")
	parallel := fs.Int("parallel", 1, "concurrent claude runs; give each its own -graph-dirs entry")
	timeout := fs.Duration("timeout", 10*time.Minute, "limit per run")
	out := fs.String("out", "results.json", "where to write the runs")
	_ = fs.Parse(args)

	var set QuestionSet
	if err := readJSON(*questions, &set); err != nil {
		return err
	}
	if *grepDir == "" || *chitonDir == "" {
		return fmt.Errorf("-grep-dir and -chiton-dir are required")
	}
	gd := []string{*chitonDir}
	if *graphDirs != "" {
		gd = strings.Split(*graphDirs, ",")
	}
	cfg := RunConfig{Model: *model, GrepDir: *grepDir, ChitonDir: *chitonDir, GraphDirs: gd, ChitonBin: *chitonBin, Timeout: *timeout}
	runs := RunAll(context.Background(), cfg, set, strings.Split(*conditions, ","), *repeat, *parallel, func(r Run, n, total int) {
		status := fmt.Sprintf("F1 %.2f", r.Score.F1)
		if r.Error != "" {
			status = "error: " + r.Error
		}
		fmt.Printf("[%d/%d] %-6s %-12s r%d  %s  %d turns  %ds\n", n, total, r.Condition, r.QuestionID, r.Repeat, status, r.NumTurns, r.DurationMS/1000)
	})
	return writeJSON(*out, runs)
}

func cmdReport(args []string) error {
	fs := flag.NewFlagSet("report", flag.ExitOnError)
	questions := fs.String("questions", "questions.json", "question set from qualbench gen")
	results := fs.String("results", "results.json", "runs from qualbench run")
	_ = fs.Parse(args)

	var set QuestionSet
	if err := readJSON(*questions, &set); err != nil {
		return err
	}
	var runs []Run
	if err := readJSON(*results, &runs); err != nil {
		return err
	}
	WriteReport(os.Stdout, set, runs)
	return nil
}

func gitHead(dir string) string {
	out, err := exec.Command("git", "-C", dir, "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	// Windows PowerShell writes a UTF-8 BOM, which encoding/json rejects.
	return json.Unmarshal(bytes.TrimPrefix(b, []byte("\xef\xbb\xbf")), v)
}
