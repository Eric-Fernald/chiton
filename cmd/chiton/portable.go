package main

import (
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/config"
	"github.com/Eric-Fernald/chiton/internal/graph/portable"
)

// exportCmd writes the whole graph to a portable, self-describing JSON
// snapshot (nodes with inlined embeddings + edges). This is the supported way
// to back up, share, or migrate a graph between machines/versions, and the
// counterpart to `chiton merge`.
func exportCmd() *cobra.Command {
	var outFile string

	cmd := &cobra.Command{
		Use:   "export",
		Short: "Export the graph to a portable JSON snapshot",
		Long: `Write every node (with embeddings inlined) and edge to a single JSON file.
The snapshot is portable across machines and chiton versions — unlike the raw
BadgerDB store directory, which is only safe to copy between identical versions.

Pair with 'chiton merge' to import or combine graphs across repositories.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			// Read-only so export can run while 'chiton serve' holds the store.
			gs, err := openGraphStore(cfg.GraphPath, true)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			var w io.Writer
			if outFile == "" {
				w = cmd.OutOrStdout()
			} else {
				f, err := os.Create(outFile)
				if err != nil {
					return fmt.Errorf("create %s: %w", outFile, err)
				}
				defer f.Close()
				w = f
			}

			if err := gs.Snapshot(cmd.Context(), w); err != nil {
				return fmt.Errorf("snapshot: %w", err)
			}
			if outFile != "" {
				fmt.Fprintf(cmd.OutOrStdout(), "export: wrote %s\n", outFile)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&outFile, "out", "", "Output file (default: stdout)")
	return cmd
}

// mergeCmd loads one or more exported snapshots into the current graph store.
// Because node IDs are stable hashes of (source_uri, label), this both imports
// a single graph and merges several repos' graphs into one — nodes they share
// (external packages, shared concepts) collapse to one, and re-running is
// idempotent.
func mergeCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "merge <snapshot.json>...",
		Aliases: []string{"import"},
		Short:   "Merge exported graph snapshot(s) into the current graph",
		Long: `Read one or more JSON snapshots (from 'chiton export') and upsert their
nodes and edges into the configured graph store.

Node IDs are stable content hashes, so merging is idempotent and dedups nodes
shared across graphs — the basis for cross-repo knowledge graphs. Point
graph_path at a fresh store to import, or at an existing one to combine.`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			gs, err := openGraphStore(cfg.GraphPath, false)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			out := cmd.OutOrStdout()
			var total portable.MergeStats
			for _, path := range args {
				f, err := os.Open(path)
				if err != nil {
					return fmt.Errorf("open %s: %w", path, err)
				}
				snap, err := portable.Read(f)
				f.Close()
				if err != nil {
					return fmt.Errorf("%s: %w", path, err)
				}
				stats, err := portable.Merge(cmd.Context(), gs, snap, portable.DefaultBatch)
				if err != nil {
					return fmt.Errorf("%s: %w", path, err)
				}
				total.Nodes += stats.Nodes
				total.Edges += stats.Edges
				total.Embeddings += stats.Embeddings
				fmt.Fprintf(out, "merge: %s — %d nodes, %d edges, %d embeddings\n",
					path, stats.Nodes, stats.Edges, stats.Embeddings)
			}
			fmt.Fprintf(out, "merge: done — %d file(s), %d nodes, %d edges, %d embeddings\n",
				len(args), total.Nodes, total.Edges, total.Embeddings)
			fmt.Fprintln(out, "merge: run 'chiton cluster' to refresh communities, 'chiton serve' to pick up changes")
			return nil
		},
	}
	return cmd
}
