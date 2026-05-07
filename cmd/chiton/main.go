package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	goast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/config"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

func main() {
	if err := rootCmd().Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func rootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "chiton",
		Short: "On-device knowledge graph engine for code, docs, and meetings",
		Long: `Chiton ingests code, docs, and meeting transcripts into a typed graph,
clusters them with Leiden, and serves the result to AI assistants over MCP
so they spend fewer tokens navigating architecture, docs, and prior tool calls.`,
	}

	root.AddCommand(buildCmd(), queryCmd(), serveCmd())
	return root
}

func buildCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "build <path>",
		Short: "Walk a directory and extract its graph",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}

			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "build: scanning %s\n", args[0])
			fmt.Fprintf(out, "build: graph → %s\n", cfg.GraphPath)

			gs, err := badgerstore.Open(cfg.GraphPath)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			ex := &goast.GoExtractor{}
			nodes, edges, err := ex.ExtractDir(cmd.Context(), args[0])
			if err != nil {
				return fmt.Errorf("extract: %w", err)
			}

			if err := gs.UpsertNodes(cmd.Context(), nodes); err != nil {
				return fmt.Errorf("upsert nodes: %w", err)
			}
			if err := gs.UpsertEdges(cmd.Context(), edges); err != nil {
				return fmt.Errorf("upsert edges: %w", err)
			}

			fmt.Fprintf(out, "build: done — %d nodes, %d edges\n", len(nodes), len(edges))
			return nil
		},
	}
}

func queryCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "query <label>",
		Short: "Look up a node and its neighborhood",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			_ = cfg
			fmt.Fprintf(cmd.OutOrStdout(), "query: searching for %q\n", args[0])
			fmt.Fprintln(cmd.OutOrStdout(), "query: not yet implemented — come back after Step 7 (MCP server)")
			return nil
		},
	}
}

func serveCmd() *cobra.Command {
	var stdio bool
	var addr string

	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Start the MCP server",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			_ = cfg
			if stdio {
				fmt.Fprintln(cmd.OutOrStdout(), "serve: stdio mode — not yet implemented")
			} else {
				fmt.Fprintf(cmd.OutOrStdout(), "serve: HTTP mode on %s — not yet implemented\n", addr)
			}
			return nil
		},
	}

	cmd.Flags().BoolVar(&stdio, "stdio", false, "serve over stdio (for MCP assistants)")
	cmd.Flags().StringVar(&addr, "http", ":7474", "HTTP listen address (for debug)")
	return cmd
}
