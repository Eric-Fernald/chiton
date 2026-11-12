package main

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/extract/subagent"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// verifyCmd is the parent for checks of INFERRED nodes against the sources they came from.
func verifyCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "verify",
		Short: "Check inferred concepts and decisions against their sources",
	}
	cmd.AddCommand(verifySweepCmd())
	return cmd
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
