package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/cache"
	"github.com/Eric-Fernald/chiton/internal/config"
	goast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
	mcpsrv "github.com/Eric-Fernald/chiton/internal/mcp"
	mcpsdk "github.com/mark3labs/mcp-go/server"
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

	root.AddCommand(buildCmd(), queryCmd(), serveCmd(), installCmd())
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

			cachePath := filepath.Join(filepath.Dir(cfg.GraphPath), "cache.json")
			ch, err := cache.Load(cachePath)
			if err != nil {
				return fmt.Errorf("open cache: %w", err)
			}

			ex := &goast.GoExtractor{}
			start := time.Now()
			var scanned, skipped, totalNodes, totalEdges int

			walkErr := filepath.WalkDir(args[0], func(path string, d fs.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if d.IsDir() || !strings.HasSuffix(path, ".go") {
					return nil
				}
				scanned++

				content, err := os.ReadFile(path)
				if err != nil {
					fmt.Fprintf(os.Stderr, "chiton: skip %s: %v\n", path, err)
					return nil
				}
				sum := sha256.Sum256(content)
				hexSum := fmt.Sprintf("%x", sum[:])

				absPath, _ := filepath.Abs(path)
				uri := "file://" + filepath.ToSlash(absPath)

				if !ch.Changed(uri, hexSum) {
					skipped++
					return nil
				}

				nodes, edges, err := ex.ExtractFile(cmd.Context(), path)
				if err != nil {
					fmt.Fprintf(os.Stderr, "chiton: skip %s: %v\n", path, err)
					return nil
				}

				if err := gs.UpsertNodes(cmd.Context(), nodes); err != nil {
					return fmt.Errorf("upsert nodes for %s: %w", path, err)
				}
				if err := gs.UpsertEdges(cmd.Context(), edges); err != nil {
					return fmt.Errorf("upsert edges for %s: %w", path, err)
				}

				ch.Mark(uri, hexSum)
				totalNodes += len(nodes)
				totalEdges += len(edges)
				return nil
			})
			if walkErr != nil {
				return walkErr
			}

			if err := ch.Save(); err != nil {
				return fmt.Errorf("save cache: %w", err)
			}

			fmt.Fprintf(out, "build: done — %d files scanned, %d skipped (unchanged), %d nodes, %d edges [%s]\n",
				scanned, skipped, totalNodes, totalEdges, time.Since(start).Round(time.Millisecond))
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

func installCmd() *cobra.Command {
	var assistant string
	var dir string

	cmd := &cobra.Command{
		Use:   "install",
		Short: "Register chiton with an AI assistant (writes .mcp.json and rules file)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			exe, err := os.Executable()
			if err != nil {
				return fmt.Errorf("resolve executable: %w", err)
			}
			exe, err = filepath.EvalSymlinks(exe)
			if err != nil {
				return fmt.Errorf("resolve symlinks: %w", err)
			}

			out := cmd.OutOrStdout()

			mcpPath := filepath.Join(dir, ".mcp.json")
			if err := writeMCPJSON(mcpPath, exe); err != nil {
				return err
			}
			fmt.Fprintf(out, "install: wrote %s\n", mcpPath)

			switch assistant {
			case "claude-code":
				rulesPath := filepath.Join(dir, "CLAUDE.md")
				if err := writeClaudeRules(rulesPath); err != nil {
					return err
				}
				fmt.Fprintf(out, "install: wrote %s\n", rulesPath)
			default:
				return fmt.Errorf("unknown assistant %q — supported: claude-code", assistant)
			}

			fmt.Fprintln(out, "install: done — restart Claude Code to pick up the new MCP server.")
			return nil
		},
	}

	cmd.Flags().StringVar(&assistant, "assistant", "claude-code", "AI assistant to configure (claude-code)")
	cmd.Flags().StringVar(&dir, "dir", ".", "Directory to write config files into")
	return cmd
}

func writeMCPJSON(path, exe string) error {
	type serverEntry struct {
		Command string   `json:"command"`
		Args    []string `json:"args"`
	}
	type mcpConfig struct {
		MCPServers map[string]serverEntry `json:"mcpServers"`
	}
	cfg := mcpConfig{
		MCPServers: map[string]serverEntry{
			"chiton": {Command: exe, Args: []string{"serve", "--stdio"}},
		},
	}
	b, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

const claudeRules = `# Chiton — knowledge graph context

This project runs **chiton**, an on-device knowledge graph engine.
The graph is built from the codebase's AST and stored locally.

## When to use chiton tools

Before using grep or file-read to answer architecture or navigation questions,
call the chiton MCP tools instead:

| Tool | Use when |
|---|---|
| ` + "`get_node_neighborhood`" + ` | "What does X depend on?", "What calls Y?", "Show me X and its neighbors" |
| ` + "`query_graph`" + ` | Discovering what symbols or files exist matching a name |
| ` + "`shortest_path`" + ` | "How are A and B connected?" |

Chiton returns typed graph relationships (CALLS, IMPORTS, CONTAINS, etc.)
that answer structural questions with far fewer tokens than reading source files.

**Still use grep/read** when you need actual source code content to understand
an implementation, locate a bug, or make an edit — not just to answer
"what exists" or "how things are connected."

## Building the graph

` + "```" + `sh
chiton build .          # index this workspace
chiton serve --stdio    # already running as MCP server
` + "```" + `
`

func writeClaudeRules(path string) error {
	existing, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	// Skip write if the rules are already present.
	if len(existing) > 0 {
		return os.WriteFile(path, append(existing, []byte("\n"+claudeRules)...), 0o644)
	}
	return os.WriteFile(path, []byte(claudeRules), 0o644)
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

			gs, err := badgerstore.Open(cfg.GraphPath)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			s := mcpsrv.New(gs)

			if stdio {
				return mcpsdk.ServeStdio(s)
			}
			fmt.Fprintf(cmd.OutOrStdout(), "serve: listening on %s\n", addr)
			return mcpsdk.NewSSEServer(s).Start(addr)
		},
	}

	cmd.Flags().BoolVar(&stdio, "stdio", false, "serve over stdio (for MCP assistants)")
	cmd.Flags().StringVar(&addr, "http", ":7474", "HTTP listen address (for debug)")
	return cmd
}
