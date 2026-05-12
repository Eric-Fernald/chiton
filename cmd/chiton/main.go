package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/cache"
	"github.com/Eric-Fernald/chiton/internal/config"
	goast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/metrics"
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

	root.AddCommand(buildCmd(), queryCmd(), serveCmd(), installCmd(), clusterCmd())
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

func clusterCmd() *cobra.Command {
	var resolution float64
	var seed int64
	var topN int

	cmd := &cobra.Command{
		Use:   "cluster",
		Short: "Run Leiden community detection and betweenness metrics on the graph",
		Long: `Run Leiden community detection and betweenness metrics on the graph.

Results are written to a JSON sidecar file (chiton-out/cluster.json) so this
command can run while chiton serve is active — no write lock on the store is
needed. Restart chiton serve after clustering to load the new results.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "cluster: reading graph at %s\n", cfg.GraphPath)

			// Open read-only so this command can run while chiton serve is active.
			gs, err := badgerstore.OpenReadOnly(cfg.GraphPath)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			ctx := cmd.Context()
			nodes, err := gs.AllNodes(ctx)
			if err != nil {
				return fmt.Errorf("load nodes: %w", err)
			}
			edges, err := gs.AllEdges(ctx)
			if err != nil {
				return fmt.Errorf("load edges: %w", err)
			}
			fmt.Fprintf(out, "cluster: loaded %d nodes, %d edges\n", len(nodes), len(edges))

			if len(nodes) == 0 {
				fmt.Fprintln(out, "cluster: graph is empty, nothing to do")
				return nil
			}

			start := time.Now()
			rng := rand.New(rand.NewSource(seed))

			commMap := cluster.Run(nodes, edges, resolution, rng)
			metricsMap := metrics.Compute(nodes, edges)

			// Build sidecar results.
			commSet := map[int]struct{}{}
			nodeResults := make(map[string]cluster.NodeResult, len(nodes))
			for _, n := range nodes {
				c := commMap[n.ID]
				commSet[c] = struct{}{}
				m := metricsMap[n.ID]
				nodeResults[n.ID] = cluster.NodeResult{
					Label:       n.Label,
					SourceURI:   n.SourceURI,
					CommunityID: c,
					Betweenness: m.Betweenness,
					Degree:      m.Degree,
				}
			}

			results := &cluster.Results{
				GeneratedAt:    time.Now(),
				NodeCount:      len(nodes),
				CommunityCount: len(commSet),
				Nodes:          nodeResults,
			}

			sidecarPath := cluster.SidecarPath(cfg.GraphPath)
			if err := cluster.WriteSidecar(sidecarPath, results); err != nil {
				return err
			}

			elapsed := time.Since(start).Round(time.Millisecond)
			fmt.Fprintf(out, "cluster: done in %s — %d nodes in %d communities\n",
				elapsed, len(nodes), len(commSet))
			fmt.Fprintf(out, "cluster: results written to %s\n", sidecarPath)
			fmt.Fprintln(out, "cluster: restart chiton serve to load the new community data")

			// Print top-N god nodes by betweenness.
			type godNode struct {
				label       string
				betweenness float64
				degree      int
				community   int
			}
			god := make([]godNode, 0, len(nodes))
			for _, n := range nodes {
				r := nodeResults[n.ID]
				god = append(god, godNode{n.Label, r.Betweenness, r.Degree, r.CommunityID})
			}
			sort.Slice(god, func(i, j int) bool {
				return god[i].betweenness > god[j].betweenness
			})
			if topN > len(god) {
				topN = len(god)
			}
			fmt.Fprintf(out, "\nTop %d god nodes (highest betweenness):\n", topN)
			for i := 0; i < topN; i++ {
				g := god[i]
				fmt.Fprintf(out, "  %-40s  betweenness=%.4f  degree=%d  community=%d\n",
					g.label, g.betweenness, g.degree, g.community)
			}
			return nil
		},
	}

	cmd.Flags().Float64Var(&resolution, "resolution", 1.0, "Leiden resolution parameter γ (higher = smaller communities)")
	cmd.Flags().Int64Var(&seed, "seed", 42, "Random seed for reproducible clustering")
	cmd.Flags().IntVar(&topN, "top", 10, "Number of god nodes to print")
	return cmd
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

			gs, err := badgerstore.OpenReadOnly(cfg.GraphPath)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			clusterResults, err := cluster.LoadSidecar(cluster.SidecarPath(cfg.GraphPath))
			if err != nil {
				fmt.Fprintf(cmd.ErrOrStderr(), "serve: warning: could not load cluster sidecar: %v\n", err)
			}

			s := mcpsrv.New(gs, clusterResults)

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
