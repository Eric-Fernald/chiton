package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules"
	"github.com/Eric-Fernald/chiton/internal/cache"
	"github.com/Eric-Fernald/chiton/internal/config"
	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/metrics"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
	mcpsrv "github.com/Eric-Fernald/chiton/internal/mcp"
	"github.com/Eric-Fernald/chiton/internal/report"
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

	root.AddCommand(initCmd(), buildCmd(), queryCmd(), serveCmd(), installCmd(), clusterCmd(), reportCmd(), analyzeCmd(), findingsCmd())
	return root
}

func analyzeCmd() *cobra.Command {
	var category string
	var minSeverity string

	cmd := &cobra.Command{
		Use:   "analyze <path>",
		Short: "Run deterministic rules over a directory and persist findings",
		Long: `Walk the given path, run every registered file-scope rule on each supported
file, then run graph-scope rules over the full extracted set. Findings are
upserted as graph nodes (VIOLATES → rule, AFFECTS → file/symbol) so they can
be queried by 'chiton findings list' or the MCP tools.`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			gs, err := badgerstore.Open(cfg.GraphPath)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "analyze: scanning %s\n", args[0])

			az := analyze.New(gs, nil)
			res, err := az.Analyze(cmd.Context(), args[0])
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
			return nil
		},
	}

	cmd.Flags().StringVar(&category, "category", "", "Filter output to one category")
	cmd.Flags().StringVar(&minSeverity, "min-severity", "", "Filter output to findings at or above this severity (info|low|medium|high|critical)")
	return cmd
}

func findingsCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "findings",
		Short: "Inspect persisted analyze findings",
	}
	cmd.AddCommand(findingsListCmd(), findingsExplainCmd())
	return cmd
}

func findingsListCmd() *cobra.Command {
	var category, minSeverity, scope string
	var limit int
	var asJSON bool

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List findings from the most recent analyze run",
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

			views, err := analyze.List(cmd.Context(), gs, analyze.ListOptions{
				Category:    category,
				MinSeverity: analyze.Severity(minSeverity),
				Scope:       scope,
				Limit:       limit,
			})
			if err != nil {
				return err
			}

			out := cmd.OutOrStdout()
			if asJSON {
				b, err := json.MarshalIndent(views, "", "  ")
				if err != nil {
					return err
				}
				fmt.Fprintln(out, string(b))
				return nil
			}
			if len(views) == 0 {
				fmt.Fprintln(out, "no findings match")
				return nil
			}
			for _, v := range views {
				loc := v.TargetURI
				if v.Span != nil {
					loc = fmt.Sprintf("%s:%d", loc, v.Span.StartLine)
				}
				fmt.Fprintf(out, "  %s  [%s] %-30s %s\n    %s\n    %s\n",
					v.ID[:8], v.Severity, v.RuleID, loc, v.Message, "")
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&category, "category", "", "Filter by category")
	cmd.Flags().StringVar(&minSeverity, "min-severity", "", "Filter to findings at or above this severity")
	cmd.Flags().StringVar(&scope, "scope", "", "Substring match against target_uri (e.g. 'internal/auth')")
	cmd.Flags().IntVar(&limit, "limit", 0, "Max findings to return (0 = unlimited)")
	cmd.Flags().BoolVar(&asJSON, "json", false, "Output JSON")
	return cmd
}

func findingsExplainCmd() *cobra.Command {
	var asJSON bool
	cmd := &cobra.Command{
		Use:   "explain <id-or-label>",
		Short: "Show a finding and its rule + related findings",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			gs, err := badgerstore.OpenReadOnly(cfg.GraphPath)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			exp, err := analyze.Explain(cmd.Context(), gs, args[0])
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			if asJSON {
				b, err := json.MarshalIndent(exp, "", "  ")
				if err != nil {
					return err
				}
				fmt.Fprintln(out, string(b))
				return nil
			}
			f := exp.Finding
			fmt.Fprintf(out, "Finding %s\n", f.ID)
			fmt.Fprintf(out, "  rule:     %s\n", f.RuleID)
			fmt.Fprintf(out, "  severity: %s\n", f.Severity)
			fmt.Fprintf(out, "  category: %s\n", f.Category)
			fmt.Fprintf(out, "  target:   %s\n", f.TargetURI)
			if f.Span != nil {
				fmt.Fprintf(out, "  line:     %d\n", f.Span.StartLine)
			}
			fmt.Fprintf(out, "  message:  %s\n", f.Message)
			if f.Fix != "" {
				fmt.Fprintf(out, "  fix:      %s\n", f.Fix)
			}
			if exp.Rule != nil {
				fmt.Fprintf(out, "\nRule %s\n", exp.Rule.ID)
				fmt.Fprintf(out, "  summary: %s\n", exp.Rule.Summary)
				if exp.Rule.Fix != "" {
					fmt.Fprintf(out, "  fix:     %s\n", exp.Rule.Fix)
				}
			}
			if len(exp.Related) > 0 {
				fmt.Fprintf(out, "\n%d related finding(s) on the same target:\n", len(exp.Related))
				for _, r := range exp.Related {
					fmt.Fprintf(out, "  [%s] %s — %s\n", r.Severity, r.RuleID, r.Message)
				}
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&asJSON, "json", false, "Output JSON")
	return cmd
}

const chitonGitignore = `# Auto-generated by chiton init. Do not edit manually.
# Track only the repo-level policy files; ignore runtime artifacts.
*
!.gitignore
!hooks.yaml
!sources.yaml
`

const chitonHooksYAML = `pre-commit:
  block-on: [critical, high]
  llm: false
  diff-baseline: HEAD~1
  timeout-seconds: 30
pre-push:
  block-on: [critical]
  llm: true
`

const chitonSourcesYAML = `# Source allowlist for remote providers.
# Set allow_remote_llm or allow_remote_embed to true only for sources
# you explicitly trust to leave the device.
sources: []
`

const schemaVersion = "1\n"

func initCmd() *cobra.Command {
	var dir string
	var encrypt bool

	cmd := &cobra.Command{
		Use:   "init",
		Short: "Initialize .chiton/ directory structure in a repository",
		Long: `Create the .chiton/ directory with default policy files and an auto-generated
.gitignore that tracks hooks.yaml and sources.yaml but ignores runtime artifacts
(graph store, cache, vector index, audit log).

Run this once at the root of each repository you want to index with chiton.
After init, run "chiton build ." to populate the graph.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			chitonDir := filepath.Join(dir, ".chiton")
			out := cmd.OutOrStdout()

			created := 0
			skipped := 0

			writeIfAbsent := func(path, content string) error {
				if _, err := os.Stat(path); err == nil {
					fmt.Fprintf(out, "init: skip    %s (already exists)\n", path)
					skipped++
					return nil
				}
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					return fmt.Errorf("mkdir %s: %w", filepath.Dir(path), err)
				}
				if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
					return fmt.Errorf("write %s: %w", path, err)
				}
				fmt.Fprintf(out, "init: created %s\n", path)
				created++
				return nil
			}

			files := []struct{ path, content string }{
				{filepath.Join(chitonDir, ".gitignore"), chitonGitignore},
				{filepath.Join(chitonDir, "hooks.yaml"), chitonHooksYAML},
				{filepath.Join(chitonDir, "sources.yaml"), chitonSourcesYAML},
				{filepath.Join(chitonDir, "schema_version"), schemaVersion},
			}
			for _, f := range files {
				if err := writeIfAbsent(f.path, f.content); err != nil {
					return err
				}
			}

			if !encrypt {
				fmt.Fprintln(out, "init: note: encryption disabled (--encrypt=false); graph will be stored in plaintext")
			}

			fmt.Fprintf(out, "init: done — %d file(s) created, %d skipped\n", created, skipped)
			fmt.Fprintf(out, "init: next step: chiton build .\n")
			return nil
		},
	}

	cmd.Flags().StringVar(&dir, "dir", ".", "Directory to initialise (default: current directory)")
	cmd.Flags().BoolVar(&encrypt, "encrypt", true, "Enable encryption at rest (default: true; use --encrypt=false for CI/test environments)")
	return cmd
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

			start := time.Now()
			var scanned, skipped, totalNodes, totalEdges int

			walkErr := filepath.WalkDir(args[0], func(path string, d fs.DirEntry, err error) error {
				if err != nil {
					return err
				}
				ex := extast.ForFile(path)
				if d.IsDir() || ex == nil {
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
	var depth int
	var edgeKinds []string
	var file string
	var sourcePrefix string
	var asJSON bool

	cmd := &cobra.Command{
		Use:   "query <label>",
		Short: "Look up a node and its neighborhood",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}

			gs, err := badgerstore.OpenReadOnly(cfg.GraphPath)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			ctx := cmd.Context()
			label := args[0]

			limit := 1
			if file != "" {
				limit = 20
			}
			nodes, err := gs.SearchNodes(ctx, label, limit)
			if err != nil {
				return fmt.Errorf("search: %w", err)
			}
			if len(nodes) == 0 {
				fmt.Fprintf(cmd.OutOrStdout(), "no node found matching %q\n", label)
				return nil
			}

			seed := pickNodeByFile(nodes, file)

			var kinds []model.EdgeKind
			for _, k := range edgeKinds {
				kinds = append(kinds, model.EdgeKind(strings.ToUpper(k)))
			}

			if depth < 1 {
				depth = 1
			} else if depth > 5 {
				depth = 5
			}
			sub, err := gs.Neighbors(ctx, seed.ID, depth, kinds)
			if err != nil {
				return fmt.Errorf("neighbors: %w", err)
			}

			// Remove the seed from the neighbor list (BFS may return it).
			noSelf := sub.Nodes[:0:0]
			for _, n := range sub.Nodes {
				if n.ID != seed.ID {
					noSelf = append(noSelf, n)
				}
			}
			filteredNodes, filteredEdges := filterNodesByPrefix(seed, noSelf, sub.Edges, sourcePrefix)

			out := cmd.OutOrStdout()

			if asJSON {
				type result struct {
					Seed  model.Node   `json:"seed"`
					Nodes []model.Node `json:"nodes"`
					Edges []model.Edge `json:"edges"`
				}
				b, err := json.MarshalIndent(result{Seed: seed, Nodes: filteredNodes, Edges: filteredEdges}, "", "  ")
				if err != nil {
					return err
				}
				fmt.Fprintln(out, string(b))
				return nil
			}

			printNodeLine := func(n model.Node, indent string) {
				span := ""
				if n.Span != nil {
					span = fmt.Sprintf(" [%d:%d]", n.Span.StartLine, n.Span.EndLine)
				}
				fmt.Fprintf(out, "%s%-10s  %-36s  %s%s\n", indent, n.Kind, n.Label, n.SourceURI, span)
			}

			fmt.Fprintln(out, "seed:")
			printNodeLine(seed, "  ")

			fmt.Fprintf(out, "\n%d neighbor(s):\n", len(filteredNodes))
			for _, n := range filteredNodes {
				printNodeLine(n, "  ")
			}

			labels := map[string]string{seed.ID: seed.Label}
			for _, n := range filteredNodes {
				labels[n.ID] = n.Label
			}
			fmt.Fprintf(out, "\n%d edge(s):\n", len(filteredEdges))
			for _, e := range filteredEdges {
				from := labels[e.Src]
				if from == "" {
					from = e.Src[:8]
				}
				to := labels[e.Dst]
				if to == "" {
					to = e.Dst[:8]
				}
				fmt.Fprintf(out, "  %-15s  %s → %s\n", e.Kind, from, to)
			}

			return nil
		},
	}

	cmd.Flags().IntVar(&depth, "depth", 2, "Hop depth (1–5)")
	cmd.Flags().StringSliceVar(&edgeKinds, "edge-kinds", nil, "Filter to these edge types, e.g. CALLS,IMPORTS")
	cmd.Flags().StringVar(&file, "file", "", "Source file hint to disambiguate label matches (substring of source_uri)")
	cmd.Flags().StringVar(&sourcePrefix, "source-prefix", "", "Only return neighbors whose source_uri contains this substring")
	cmd.Flags().BoolVar(&asJSON, "json", false, "Output raw JSON")
	return cmd
}

// pickNodeByFile mirrors the MCP server's pickByFile logic: among nodes
// matching the file hint, symbol nodes are preferred over file nodes.
func pickNodeByFile(nodes []model.Node, file string) model.Node {
	if file == "" {
		return nodes[0]
	}
	lower := strings.ToLower(file)
	var firstMatch *model.Node
	for i := range nodes {
		n := &nodes[i]
		if strings.Contains(strings.ToLower(n.SourceURI), lower) {
			if firstMatch == nil {
				firstMatch = n
			}
			if n.Kind == model.KindSymbol {
				return *n
			}
		}
	}
	if firstMatch != nil {
		return *firstMatch
	}
	return nodes[0]
}

// filterNodesByPrefix removes neighbor nodes whose source_uri does not contain
// prefix, and removes edges where either endpoint is absent after filtering.
// The seed is always retained so its edges survive.
func filterNodesByPrefix(seed model.Node, nodes []model.Node, edges []model.Edge, prefix string) ([]model.Node, []model.Edge) {
	if prefix == "" {
		return nodes, edges
	}
	lower := strings.ToLower(prefix)
	kept := map[string]bool{seed.ID: true}
	var filteredNodes []model.Node
	for _, n := range nodes {
		if strings.Contains(strings.ToLower(n.SourceURI), lower) {
			filteredNodes = append(filteredNodes, n)
			kept[n.ID] = true
		}
	}
	var filteredEdges []model.Edge
	for _, e := range edges {
		if kept[e.Src] && kept[e.Dst] {
			filteredEdges = append(filteredEdges, e)
		}
	}
	return filteredNodes, filteredEdges
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

func reportCmd() *cobra.Command {
	var outFile string
	var toStdout bool

	cmd := &cobra.Command{
		Use:   "report",
		Short: "Generate GRAPH_REPORT.md (god nodes + community summaries)",
		Long: `Generate a Markdown report of the graph: god nodes ranked by betweenness
centrality, per-community node lists, and surprising cross-community edges.

Requires a prior 'chiton cluster' run.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}

			clusterResults, err := cluster.LoadSidecar(cluster.SidecarPath(cfg.GraphPath))
			if err != nil {
				return fmt.Errorf("load cluster sidecar: %w", err)
			}
			if clusterResults == nil {
				return fmt.Errorf("no cluster data found — run 'chiton cluster' first")
			}

			gs, err := badgerstore.OpenReadOnly(cfg.GraphPath)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			edges, err := gs.AllEdges(cmd.Context())
			if err != nil {
				return fmt.Errorf("load edges: %w", err)
			}

			var w io.Writer
			if toStdout {
				w = cmd.OutOrStdout()
			} else {
				f, err := os.Create(outFile)
				if err != nil {
					return fmt.Errorf("create %s: %w", outFile, err)
				}
				defer f.Close()
				w = f
				fmt.Fprintf(cmd.OutOrStdout(), "report: writing to %s\n", outFile)
			}

			if err := report.Generate(w, clusterResults, edges, report.Default); err != nil {
				return fmt.Errorf("generate report: %w", err)
			}

			if !toStdout {
				fmt.Fprintf(cmd.OutOrStdout(), "report: done — %d nodes, %d communities\n",
					clusterResults.NodeCount, clusterResults.CommunityCount)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&outFile, "out", "GRAPH_REPORT.md", "Output file path")
	cmd.Flags().BoolVar(&toStdout, "stdout", false, "Write to stdout instead of a file")
	return cmd
}
