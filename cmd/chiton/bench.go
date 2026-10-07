package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/bench"
)

// benchCmd compares the token cost of answering dependency questions naively, with grep and read, and with
// chiton. It builds the graph in memory with no store or server, so it runs the same anywhere.
func benchCmd() *cobra.Command {
	var name string
	var queries, synth int
	var asJSON bool

	cmd := &cobra.Command{
		Use:   "bench [path]",
		Short: "Benchmark token cost: chiton neighborhoods vs naive vs grep+read",
		Long: `Build an in-memory graph over a corpus and measure how many tokens it costs
to answer dependency questions three ways: reading every file (naive), reading
every file that mentions the symbol (grep+read), and chiton's lean neighborhood
payload. Token cost uses the same chars÷4 estimate for every strategy.

Pass a path (default "."), or --synth N to generate and benchmark a reproducible
synthetic corpus of N Go files instead. Output is a Markdown table by default.`,
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			path := "."
			if len(args) == 1 {
				path = args[0]
			}

			if synth > 0 {
				dir, err := os.MkdirTemp("", "chiton-bench-synth-*")
				if err != nil {
					return fmt.Errorf("create temp corpus: %w", err)
				}
				defer os.RemoveAll(dir)
				if err := bench.GenerateSynthetic(dir, synth); err != nil {
					return err
				}
				path = dir
				if name == "" {
					name = fmt.Sprintf("synthetic (%d files)", synth)
				}
			}
			if name == "" {
				abs, _ := filepath.Abs(path)
				name = filepath.Base(abs)
			}

			rep, err := bench.Run(cmd.Context(), name, path, queries)
			if err != nil {
				return err
			}

			out := cmd.OutOrStdout()
			if asJSON {
				return printJSON(out, rep)
			}
			bench.FormatMarkdown(out, []*bench.Report{rep})
			return nil
		},
	}

	cmd.Flags().StringVar(&name, "name", "", "Corpus label for the report (default: directory name)")
	cmd.Flags().IntVar(&queries, "queries", 5, "Number of highest-degree symbols to benchmark")
	cmd.Flags().IntVar(&synth, "synth", 0, "Generate and benchmark a synthetic corpus of N Go files instead of a path")
	cmd.Flags().BoolVar(&asJSON, "json", false, "Output JSON instead of a Markdown table")
	return cmd
}
