package hooks

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// RunOptions controls one hook-stage invocation.
type RunOptions struct {
	Stage  Stage
	Config StageConfig
	Root   string // path to analyze (typically the repo root)
}

// Result is what Run returns to the caller.
type Result struct {
	Stage           Stage             `json:"stage"`
	FilesScanned    int               `json:"files_scanned"`
	RulesRun        int               `json:"rules_run"`
	TotalFindings   int               `json:"total_findings"`
	BlockingMin     analyze.Severity  `json:"blocking_min,omitempty"`
	BlockingCount   int               `json:"blocking_count"`
	BlockingDetails []analyze.Finding `json:"blocking_details,omitempty"`
	Duration        time.Duration     `json:"duration"`
}

// Blocks reports whether this Result should fail the git hook.
func (r Result) Blocks() bool { return r.BlockingCount > 0 }

// Run executes the analyzer at root, filters findings by the stage's
// MinBlockingSeverity, and returns a Result. The caller decides what to print
// and how to translate Blocks() into the process exit code.
//
// The analyzer's timeout is enforced via the parent context — pass one with
// a deadline derived from cfg.TimeoutSeconds if you want timeouts honored.
func Run(ctx context.Context, gs store.GraphStore, opts RunOptions) (*Result, error) {
	az := analyze.New(gs, nil)
	start := time.Now()
	res, err := az.Analyze(ctx, opts.Root)
	if err != nil {
		return nil, fmt.Errorf("analyze: %w", err)
	}

	min := opts.Config.MinBlockingSeverity()
	out := &Result{
		Stage:         opts.Stage,
		FilesScanned: res.FilesScanned,
		RulesRun:      res.RulesRun,
		TotalFindings: len(res.Findings),
		BlockingMin:   min,
		Duration:      time.Since(start),
	}
	if min == "" {
		return out, nil
	}
	threshold := analyze.SeverityRank(min)
	for _, f := range res.Findings {
		if analyze.SeverityRank(f.Severity) >= threshold {
			out.BlockingCount++
			out.BlockingDetails = append(out.BlockingDetails, f)
		}
	}
	return out, nil
}

// FormatResult writes a human-readable summary of r to w.
func FormatResult(w io.Writer, r *Result) {
	fmt.Fprintf(w, "chiton %s: %d files scanned, %d findings (%d blocking) [%s]\n",
		r.Stage, r.FilesScanned, r.TotalFindings, r.BlockingCount, r.Duration.Round(time.Millisecond))
	if r.BlockingCount == 0 {
		return
	}
	fmt.Fprintf(w, "blocked by findings at or above severity %q:\n", r.BlockingMin)
	for _, f := range r.BlockingDetails {
		loc := f.TargetURI
		if f.Span != nil {
			loc = fmt.Sprintf("%s:%d", loc, f.Span.StartLine)
		}
		fmt.Fprintf(w, "  [%s] %-30s %s\n    %s\n", f.Severity, f.RuleID, loc, f.Message)
	}
}
