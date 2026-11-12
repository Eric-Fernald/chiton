package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/config"
	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/extract/subagent"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/verify"
)

// verifyCmd is the parent for checks of INFERRED nodes against the sources they came from.
func verifyCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "verify",
		Short: "Check inferred concepts and decisions against their sources",
	}
	cmd.Long = `verify keeps the INFERRED layer honest in two ways.

  chiton verify sweep                        # deterministic cleanup, no LLM
  chiton verify pending --out tasks.json     # claims whose evidence changed
  # ... judge each task's claim against its evidence, producing results.json ...
  chiton verify submit results.json          # apply the verdicts

pending and submit follow the other subagent flows: chiton emits task specs and
ingests structured results, and never calls an LLM itself.`
	cmd.AddCommand(verifySweepCmd(), verifyPendingCmd(), verifySubmitCmd())
	return cmd
}

func verifyPendingCmd() *cobra.Command {
	var kind, outFile string
	var limit int

	cmd := &cobra.Command{
		Use:   "pending",
		Short: "Print verification task specs as JSON",
		Long: `Print one task per inferred claim that has never been checked, or whose claim
or evidence changed since its last verdict. A concept is checked against the
text of the sections linked to it. A decision is checked against the current
source of each symbol it governs, read from files under the project root.
Symbols whose file changed since it was last indexed are skipped until the next
build or sync.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, gs, err := loadStore(true)
			if err != nil {
				return err
			}
			defer gs.Close()
			specs, err := verify.New(gs, config.ProjectDir(), cfg.Semantic.ConfidenceThreshold).Pending(cmd.Context(), kind, limit)
			if err != nil {
				return err
			}
			return emitSpecs(cmd.OutOrStdout(), specs, outFile, fmt.Sprintf("verify: %d pending task(s)", len(specs)))
		},
	}
	cmd.Flags().StringVar(&kind, "kind", "", "Filter to one kind: concept_support | decision_drift")
	cmd.Flags().IntVar(&limit, "limit", 0, "Max specs (0 = all)")
	cmd.Flags().StringVar(&outFile, "out", "", "Write specs to this file instead of stdout")
	return cmd
}

func verifySubmitCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "submit <results.json>",
		Short: "Apply verification verdicts (a Result object or array)",
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
			results, err := parseOneOrMany[verify.Result](data, "a verification Result")
			if err != nil {
				return err
			}
			gs, err := openGraphStore(cfg.GraphPath, false)
			if err != nil {
				return err
			}
			defer gs.Close()
			svc := verify.New(gs, config.ProjectDir(), cfg.Semantic.ConfidenceThreshold)
			// A refused result is not a usage mistake, so keep the flag help out of the output.
			cmd.SilenceUsage = true
			return runVerifySubmit(cmd.Context(), svc, results, cmd.OutOrStdout())
		},
	}
}

// runVerifySubmit applies each result and prints what happened. One refused result (outdated evidence, a quote
// that is not in it) does not stop the rest. It returns an error when any were refused.
func runVerifySubmit(ctx context.Context, svc *verify.Service, results []verify.Result, out io.Writer) error {
	var applied, recorded, refused int
	for _, r := range results {
		rep, err := svc.Submit(ctx, r)
		switch {
		case err != nil:
			refused++
			fmt.Fprintf(out, "submit: %s: refused — %v\n", r.TaskID, err)
		case rep.Applied:
			applied++
			fmt.Fprintf(out, "submit: %s: %s — %s\n", rep.TaskID, rep.Verdict, rep.Effect)
		default:
			recorded++
			fmt.Fprintf(out, "submit: %s: %s — %s\n", rep.TaskID, rep.Verdict, rep.Effect)
		}
	}
	fmt.Fprintf(out, "submit: done — %d task(s), %d applied, %d recorded only, %d refused\n", len(results), applied, recorded, refused)
	if refused > 0 {
		return fmt.Errorf("%d of %d verification result(s) were refused", refused, len(results))
	}
	return nil
}

func verifySweepCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "sweep",
		Short: "Demote concepts no source links to and re-link decisions to moved symbols",
		Long: `sweep makes two deterministic passes over the graph. It calls no LLM.

Concepts that no doc section, image or email links to any more become AMBIGUOUS,
which the read tools hide by default. The next extraction that names such a
concept takes it over and makes it visible again.

Decisions whose governed symbol was deleted are looked up again by exact name,
which restores the link when the symbol only moved to another file. Names that
match nothing are listed.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, gs, err := loadStore(false)
			if err != nil {
				return err
			}
			defer gs.Close()
			return runVerifySweep(cmd.Context(), gs, cfg.Semantic.ConfidenceThreshold, cmd.OutOrStdout())
		},
	}
}

func runVerifySweep(ctx context.Context, gs store.GraphStore, threshold float32, out io.Writer) error {
	concepts, err := subagent.New(gs, nil, threshold).SweepOrphans(ctx)
	if err != nil {
		return err
	}
	links, err := meeting.NewLinkService(gs, threshold).Relink(ctx)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "verify sweep: %d concept(s) demoted to AMBIGUOUS (no source links to them)\n", concepts.Demoted)
	fmt.Fprintf(out, "verify sweep: %d decision link(s) restored\n", links.Relinked)
	if len(links.Unresolved) > 0 {
		fmt.Fprintf(out, "verify sweep: %d decision reference(s) match no node: %s\n",
			len(links.Unresolved), strings.Join(links.Unresolved, ", "))
	}
	return nil
}
