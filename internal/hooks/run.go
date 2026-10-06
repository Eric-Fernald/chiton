package hooks

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"time"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	gitpkg "github.com/Eric-Fernald/chiton/internal/ingest/git"
	"github.com/Eric-Fernald/chiton/internal/walk"
)

// RunOptions controls one hook-stage invocation.
type RunOptions struct {
	Stage  Stage
	Config StageConfig
	Root   string // repository root
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
	Skipped         []string          `json:"skipped,omitempty"`
	Duration        time.Duration     `json:"duration"`
	// Scan is the analysis behind the verdict, which the caller may Record into a store.
	Scan *analyze.Result `json:"-"`
}

// Blocks reports whether this Result should fail the git hook.
func (r Result) Blocks() bool { return r.BlockingCount > 0 }

// Run analyzes in memory without a store (pre-commit checks the staged blobs, pre-push walks Root) and
// collects findings at or above the stage's MinBlockingSeverity. Bound it with Config.TimeoutSeconds via ctx.
func Run(ctx context.Context, opts RunOptions) (*Result, error) {
	az := analyze.New(nil, nil)
	start := time.Now()
	var res *analyze.Result
	var err error
	if opts.Stage == StagePreCommit {
		res, err = scanStaged(ctx, az, opts.Root)
	} else {
		res, err = az.Scan(ctx, opts.Root)
	}
	if err != nil {
		return nil, fmt.Errorf("analyze: %w", err)
	}

	minSev := opts.Config.MinBlockingSeverity()
	out := &Result{
		Stage:         opts.Stage,
		FilesScanned:  res.FilesScanned,
		RulesRun:      res.RulesRun,
		TotalFindings: len(res.Findings),
		BlockingMin:   minSev,
		Skipped:       res.Skipped,
		Duration:      time.Since(start),
		Scan:          res,
	}
	if minSev == "" {
		return out, nil
	}
	threshold := analyze.SeverityRank(minSev)
	for _, f := range res.Findings {
		if analyze.SeverityRank(f.Severity) >= threshold {
			out.BlockingCount++
			out.BlockingDetails = append(out.BlockingDetails, f)
		}
	}
	return out, nil
}

// scanStaged runs the file rules over the blobs staged in root's index, so the gate sees what the commit holds.
// It skips the paths the tree walk ignores (vendor/, .chitonignore entries, ...), as pre-push does.
func scanStaged(ctx context.Context, az *analyze.Analyzer, root string) (*analyze.Result, error) {
	absRoot, err := filepath.Abs(root) // StagedFiles yields absolute paths, which Skip relates to the root
	if err != nil {
		return nil, err
	}
	policy, err := walk.LoadPolicy(absRoot)
	if err != nil {
		return nil, err
	}
	staged, err := gitpkg.StagedFiles(ctx, root, func(p string) bool {
		return extast.ForFile(p) != nil && !policy.Skip(p, false)
	})
	if err != nil {
		return nil, err
	}
	files := make([]analyze.SourceFile, len(staged))
	for i, s := range staged {
		files[i] = analyze.SourceFile{Path: s.Path, Content: s.Content}
	}
	return az.ScanSources(ctx, files)
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
