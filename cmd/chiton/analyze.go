package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/config"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

const chitonDir = ".chiton"

// analyzeCmd runs deterministic rules itself (chiton analyze <path>). Its pending and submit subcommands
// run the LLM-rule subagent cycle, mirroring `chiton extract pending|submit`.
func analyzeCmd() *cobra.Command {
	var category, minSeverity, since string
	var diff, noLLM bool

	cmd := &cobra.Command{
		Use:   "analyze [path]",
		Short: "Run deterministic rules over a directory and persist findings",
		Long: `Walk the given path, run every registered file-scope rule on each supported
file, then run graph-scope rules over the full extracted set. Findings are
upserted as graph nodes (VIOLATES -> rule, AFFECTS -> file/symbol) so they can
be queried by 'chiton findings list' or the MCP tools.

With --diff, the live finding set is compared against the most recent saved
baseline (under .chiton/baselines/) and the new/resolved/changed breakdown is
printed; the current findings are then saved as the baseline for HEAD. Use
--since <ref> to diff against the baseline recorded for a specific commit.

LLM rules are dispatched separately via 'chiton analyze pending' / 'submit';
this command only reports how many LLM-rule tasks are pending (unless --no-llm).`,
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) == 0 {
				return fmt.Errorf("analyze requires a <path> (or use a subcommand: pending, submit)")
			}
			path := args[0]
			cfg, gs, err := loadStore(false)
			if err != nil {
				return err
			}
			defer gs.Close()

			out := cmd.OutOrStdout()
			ctx := cmd.Context()
			fmt.Fprintf(out, "analyze: scanning %s\n", path)

			az := analyze.New(gs, nil)
			res, err := az.Analyze(ctx, path)
			if err != nil {
				return err
			}

			minRank := -1
			if minSeverity != "" {
				minRank = analyze.SeverityRank(analyze.Severity(minSeverity))
			}
			filtered := res.Findings[:0:0]
			for _, f := range res.Findings {
				if category != "" && !strings.EqualFold(string(f.Category), category) {
					continue
				}
				if minRank >= 0 && analyze.SeverityRank(f.Severity) < minRank {
					continue
				}
				filtered = append(filtered, f)
			}

			fmt.Fprintf(out, "analyze: %d files scanned, %d rules run, %d findings [%s]\n",
				res.FilesScanned, res.RulesRun, len(filtered), res.Duration.Round(time.Millisecond))
			for _, f := range filtered {
				loc := f.TargetURI
				if f.Span != nil {
					loc = fmt.Sprintf("%s:%d", loc, f.Span.StartLine)
				}
				fmt.Fprintf(out, "  [%s] %-30s %s\n  %s\n", f.Severity, f.RuleID, loc, f.Message)
			}

			if diff || since != "" {
				if err := runDiff(ctx, out, gs, diff, since); err != nil {
					return err
				}
			}

			if !noLLM {
				reportPendingLLM(ctx, out, gs, cfg, path)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&category, "category", "", "Filter output to one category")
	cmd.Flags().StringVar(&minSeverity, "min-severity", "", "Filter output to findings at or above this severity (info|low|medium|high|critical)")
	cmd.Flags().BoolVar(&diff, "diff", false, "Compare against the most recent baseline and record a new one for HEAD")
	cmd.Flags().StringVar(&since, "since", "", "Diff against the baseline recorded for this git ref (does not record a new baseline)")
	cmd.Flags().BoolVar(&noLLM, "no-llm", false, "Do not report pending LLM-rule tasks")

	cmd.AddCommand(analyzePendingCmd(), analyzeSubmitCmd())
	return cmd
}

// runDiff loads a baseline, diffs it against the live findings, prints the
// breakdown, and (for --diff, not --since) records a fresh baseline for HEAD.
func runDiff(ctx context.Context, out io.Writer, gs store.GraphStore, diff bool, since string) error {
	baselineDir := analyze.BaselineDir(chitonDir)
	current, err := analyze.CurrentFindings(ctx, gs)
	if err != nil {
		return fmt.Errorf("load current findings: %w", err)
	}

	var base *analyze.Baseline
	if since != "" {
		sha := analyze.GitCommit(".", since)
		if sha == "" {
			return fmt.Errorf("cannot resolve git ref %q", since)
		}
		base, err = analyze.LoadBaseline(baselineDir, sha)
		if err != nil {
			return fmt.Errorf("load baseline: %w", err)
		}
		if base == nil {
			fmt.Fprintf(out, "diff: no baseline recorded for %s (%s) — run 'chiton analyze --diff' at that commit first\n", since, short(sha))
			return nil
		}
	} else {
		base, err = analyze.LatestBaseline(baselineDir)
		if err != nil {
			return fmt.Errorf("load baseline: %w", err)
		}
	}

	var baseFindings []analyze.FindingView
	baseLabel := "(none)"
	if base != nil {
		baseFindings = base.Findings
		baseLabel = short(base.Commit)
	}
	d := analyze.Diff(baseFindings, current)
	printDiff(out, d, baseLabel)

	// --since is read-only; --diff records the current state as the baseline.
	if since == "" {
		commit := analyze.GitCommit(".", "HEAD")
		label := commit
		if label == "" {
			label = "working"
		}
		// Record each commit's baseline once. Overwriting it from a dirty tree would make a later diff
		// compare the change against itself and hide the findings it introduced.
		if existing, _ := analyze.LoadBaseline(baselineDir, label); existing != nil {
			fmt.Fprintf(out, "diff: baseline for %s already recorded; not overwriting\n", short(label))
			return nil
		}
		path, err := analyze.SaveBaseline(baselineDir, commit, current, time.Now())
		if err != nil {
			return fmt.Errorf("save baseline: %w", err)
		}
		fmt.Fprintf(out, "diff: baseline for %s saved to %s\n", short(label), path)
	}
	return nil
}

func printDiff(out io.Writer, d analyze.DiffResult, baseLabel string) {
	fmt.Fprintf(out, "\ndiff vs baseline %s: %d new, %d resolved, %d changed\n",
		baseLabel, len(d.New), len(d.Resolved), len(d.Changed))
	for _, f := range d.New {
		fmt.Fprintf(out, "  + [%s] %-30s %s\n      %s\n", f.Severity, f.RuleID, loc(f), f.Message)
	}
	for _, f := range d.Resolved {
		fmt.Fprintf(out, "  - [%s] %-30s %s\n", f.Severity, f.RuleID, loc(f))
	}
	for _, c := range d.Changed {
		fmt.Fprintf(out, "  ~ [%s] %-30s %s\n      %s -> %s\n",
			c.After.Severity, c.After.RuleID, loc(c.After), c.Before.Message, c.After.Message)
	}
}

func loc(f analyze.FindingView) string {
	if f.Span != nil {
		return fmt.Sprintf("%s:%d", f.TargetURI, f.Span.StartLine)
	}
	return f.TargetURI
}

func short(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

// reportPendingLLM prints how many LLM-rule tasks are awaiting dispatch. It is
// best-effort: a walk failure is reported, not fatal.
func reportPendingLLM(ctx context.Context, out io.Writer, gs store.GraphStore, cfg *config.Config, path string) {
	svc := analyze.NewLLMService(gs, nil, cfg.Semantic.ConfidenceThreshold, chitonDir)
	specs, err := svc.Pending(ctx, path, "", 0)
	if err != nil {
		fmt.Fprintf(out, "analyze: note: could not compute pending LLM tasks: %v\n", err)
		return
	}
	if len(specs) == 0 {
		return
	}
	fmt.Fprintf(out, "analyze: %d LLM-rule task(s) pending — run 'chiton analyze pending --out tasks.json' to dispatch (or --no-llm to skip)\n", len(specs))
}

func analyzePendingCmd() *cobra.Command {
	var rule, outFile, path string
	var limit int

	cmd := &cobra.Command{
		Use:   "pending [path]",
		Short: "Emit LLM-rule task specs for files not yet analysed",
		Long: `Walk the path and print one task spec per (LLM rule, file) pair that has not
been analysed yet (tracked in .chiton/llm-analyzed.json). Run each task — inspect
the code in "input" for the issue the rule describes — then feed the structured
results back with 'chiton analyze submit'. chiton calls no LLM itself.`,
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) == 1 {
				path = args[0]
			}
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			// Pending does not touch the store; nil keeps it usable before a build.
			svc := analyze.NewLLMService(nil, nil, cfg.Semantic.ConfidenceThreshold, chitonDir)
			specs, err := svc.Pending(cmd.Context(), path, rule, limit)
			if err != nil {
				return err
			}
			return emitSpecs(cmd.OutOrStdout(), specs, outFile,
				fmt.Sprintf("analyze: %d pending LLM task(s)", len(specs)))
		},
	}
	cmd.Flags().StringVar(&rule, "rule", "", "Limit to one LLM rule (exact id or substring, e.g. 'llm/injection-risk')")
	cmd.Flags().IntVar(&limit, "limit", 0, "Max task specs (0 = all)")
	cmd.Flags().StringVar(&outFile, "out", "", "Write specs to this file instead of stdout")
	cmd.Flags().StringVar(&path, "path", ".", "Directory to analyse")
	return cmd
}

func analyzeSubmitCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "submit <results.json>",
		Short: "Ingest LLM-rule findings (an LLMResult object or array)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			data, err := os.ReadFile(args[0])
			if err != nil {
				return fmt.Errorf("read %s: %w", args[0], err)
			}
			results, err := parseOneOrMany[analyze.LLMResult](data, "an LLMResult")
			if err != nil {
				return err
			}
			gs, err := openGraphStore(cfg.GraphPath, false)
			if err != nil {
				return err
			}
			defer gs.Close()

			svc := analyze.NewLLMService(gs, nil, cfg.Semantic.ConfidenceThreshold, chitonDir)
			out := cmd.OutOrStdout()
			var findings, ambiguous int
			for _, r := range results {
				rep, err := svc.Submit(cmd.Context(), r)
				if err != nil {
					return err
				}
				findings += rep.FindingsAdded
				ambiguous += rep.Ambiguous
				fmt.Fprintf(out, "submit: %s (%s) — %d finding(s), %d ambiguous\n",
					rep.TaskID, r.RuleID, rep.FindingsAdded, rep.Ambiguous)
			}
			fmt.Fprintf(out, "submit: done — %d task(s), %d finding(s), %d ambiguous\n",
				len(results), findings, ambiguous)
			return nil
		},
	}
	return cmd
}
