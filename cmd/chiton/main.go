package main

import (
	"fmt"
	"os"

	"github.com/Eric-Fernald/chiton/internal/config"
	"github.com/spf13/cobra"
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
			_ = cfg
			fmt.Fprintf(cmd.OutOrStdout(), "build: scanning %s (graph → %s)\n", args[0], cfg.GraphPath)
			fmt.Fprintln(cmd.OutOrStdout(), "build: not yet implemented — come back after Step 4 (graph store)")
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
