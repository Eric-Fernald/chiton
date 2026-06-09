package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules"
	"github.com/Eric-Fernald/chiton/internal/cache"
	"github.com/Eric-Fernald/chiton/internal/config"
	chitoncrypto "github.com/Eric-Fernald/chiton/internal/crypto"
	"github.com/Eric-Fernald/chiton/internal/embed"
	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/extract/subagent"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/metrics"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/retrieve"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
	"github.com/Eric-Fernald/chiton/internal/hooks"
	urlingest "github.com/Eric-Fernald/chiton/internal/ingest/url"
	"github.com/Eric-Fernald/chiton/internal/install"
	mcpsrv "github.com/Eric-Fernald/chiton/internal/mcp"
	"github.com/Eric-Fernald/chiton/internal/report"
	"github.com/Eric-Fernald/chiton/internal/vector"
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
		Version: resolveVersion(),
	}

	root.AddCommand(initCmd(), buildCmd(), queryCmd(), serveCmd(), installCmd(), uninstallCmd(), doctorCmd(), clusterCmd(), reportCmd(), visualizeCmd(), dashboardCmd(), analyzeCmd(), findingsCmd(), hooksCmd(), embedCmd(), searchCmd(), extractCmd(), syncCmd(), meetingCmd(), addURLCmd(), teamsCmd(), browserCmd(), emailCmd(), exportCmd(), mergeCmd(), benchCmd(), versionCmd())
	return root
}

// newEmbedder constructs the configured embedding provider, with optional
// provider/model overrides from CLI flags (empty = use config).
func newEmbedder(cfg *config.Config, providerOverride, modelOverride string) (embed.Embedder, error) {
	provider := cfg.Embed.Provider
	if providerOverride != "" {
		provider = providerOverride
	}
	model := cfg.Embed.Model
	if modelOverride != "" {
		model = modelOverride
	}
	return embed.New(embed.Config{
		Provider: provider,
		Model:    model,
		Endpoint: cfg.Embed.Endpoint,
		Dim:      cfg.Embed.Dim,
	})
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
	var asJSON, includeAmbiguous bool

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List findings from the most recent analyze run",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			gs, err := openGraphStore(cfg.GraphPath, true)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			views, err := analyze.List(cmd.Context(), gs, analyze.ListOptions{
				Category:         category,
				MinSeverity:      analyze.Severity(minSeverity),
				Scope:            scope,
				Limit:            limit,
				IncludeAmbiguous: includeAmbiguous,
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
	cmd.Flags().BoolVar(&includeAmbiguous, "include-ambiguous", false, "Include low-confidence AMBIGUOUS LLM findings")
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
			gs, err := openGraphStore(cfg.GraphPath, true)
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
					fmt.Fprintf(out, "  [%s] %s â€” %s\n", r.Severity, r.RuleID, r.Message)
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
				fmt.Fprintf(out, "init: done â€” %d file(s) created, %d skipped\n", created, skipped)
				fmt.Fprintf(out, "init: next step: chiton build .\n")
				return nil
			}

			keyfilePath := chitoncrypto.ResolveKeyfilePath(chitonDir)
			if _, err := os.Stat(keyfilePath); err == nil {
				fmt.Fprintf(out, "init: skip    %s (keyfile already present â€” reuse passphrase to open)\n", keyfilePath)
			} else {
				passphrase, err := chitoncrypto.GetPassphrase("Choose a chiton passphrase (used to encrypt the graph):", true)
				if err != nil {
					return fmt.Errorf("passphrase: %w", err)
				}
				kf, err := chitoncrypto.DefaultKeyFile()
				if err != nil {
					return err
				}
				if err := kf.Save(keyfilePath, false); err != nil {
					return err
				}
				// Derive once to surface any KDF issues immediately and warm
				// the password-derivation path (Argon2id at 64 MiB takes ~1s).
				if _, err := kf.DeriveKey(passphrase); err != nil {
					return fmt.Errorf("derive key: %w", err)
				}
				fmt.Fprintf(out, "init: created %s\n", keyfilePath)
				fmt.Fprintln(out, "init: passphrase set. Subsequent commands read CHITON_PASSPHRASE from the environment or prompt interactively.")
				created++
			}

			fmt.Fprintf(out, "init: done â€” %d file(s) created, %d skipped\n", created, skipped)
			fmt.Fprintf(out, "init: next step: chiton build .\n")
			return nil
		},
	}

	cmd.Flags().StringVar(&dir, "dir", ".", "Directory to initialise (default: current directory)")
	cmd.Flags().BoolVar(&encrypt, "encrypt", true, "Enable encryption at rest (default: true; use --encrypt=false for CI/test environments)")
	return cmd
}

// openGraphStore opens the badger store with encryption if a keyfile is
// detected next to it. readOnly chooses between Open and OpenReadOnly.
func openGraphStore(graphPath string, readOnly bool) (*badgerstore.Store, error) {
	key, _, err := chitoncrypto.ResolveKeyForGraph(graphPath)
	if err != nil {
		return nil, fmt.Errorf("resolve encryption key: %w", err)
	}
	opts := badgerstore.Options{EncryptionKey: key}
	if readOnly {
		return badgerstore.OpenReadOnlyWith(graphPath, opts)
	}
	return badgerstore.OpenWith(graphPath, opts)
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
			fmt.Fprintf(out, "build: graph â†’ %s\n", cfg.GraphPath)

			gs, err := openGraphStore(cfg.GraphPath, false)
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
					// A traversal error (e.g. a permission-denied subdirectory)
					// shouldn't abort the whole build and discard cache progress;
					// log it and keep indexing the rest of the tree.
					fmt.Fprintf(os.Stderr, "chiton: skip %s: %v\n", path, err)
					return nil
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
			// Persist cache progress even if the walk aborted (e.g. an upsert
			// error), so a partial build still skips files already indexed.
			saveErr := ch.Save()
			if walkErr != nil {
				return walkErr
			}
			if saveErr != nil {
				return fmt.Errorf("save cache: %w", saveErr)
			}

			fmt.Fprintf(out, "build: done â€” %d files scanned, %d skipped (unchanged), %d nodes, %d edges [%s]\n",
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

			gs, err := openGraphStore(cfg.GraphPath, true)
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
				fmt.Fprintf(out, "  %-15s  %s â†’ %s\n", e.Kind, from, to)
			}

			return nil
		},
	}

	cmd.Flags().IntVar(&depth, "depth", 2, "Hop depth (1â€“5)")
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
	var assistant, dir, graphPath string
	var userScope, withHooks bool

	supported := strings.Join(installerNames(), ", ")

	cmd := &cobra.Command{
		Use:   "install",
		Short: "Register chiton with one or more AI assistants",
		Long: `Write the MCP server config, skill files, and always-on rules block for
each supported assistant. Re-running install is idempotent: the chiton block
inside rules files is replaced in place rather than appended.

Use --assistant <name> for a specific assistant, --assistant all for every
supported assistant, or --assistant auto (default) to install only for
assistants whose footprint is detected in the workspace.

Supported assistants: ` + supported + `.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			installers, err := install.Resolve(assistant, dir)
			if err != nil {
				return err
			}
			opts := install.Options{
				Dir:       dir,
				UserScope: userScope,
				WithHooks: withHooks,
				GraphPath: graphPath,
			}
			out := cmd.OutOrStdout()
			for _, in := range installers {
				rep, err := in.Install(opts)
				if err != nil {
					return fmt.Errorf("%s: %w", in.Name(), err)
				}
				printInstallReport(out, rep, "install")
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&assistant, "assistant", "auto", "Assistant to install for: auto, all, a name (e.g. claude-code), or a comma-separated list")
	cmd.Flags().StringVar(&dir, "dir", ".", "Workspace directory to install into")
	cmd.Flags().BoolVar(&userScope, "user-scope", false, "Install into the user-global config location (where supported)")
	cmd.Flags().BoolVar(&withHooks, "with-hooks", false, "Install opt-in PreToolUse hooks (assistant-specific)")
	cmd.Flags().StringVar(&graphPath, "graph", "", "Graph path to record in the MCP config (default: chiton's built-in default)")
	return cmd
}

func uninstallCmd() *cobra.Command {
	var assistant, dir string

	supported := strings.Join(installerNames(), ", ")

	cmd := &cobra.Command{
		Use:   "uninstall",
		Short: "Remove chiton's entries from one or more AI assistants",
		Long: `Remove the chiton MCP entry and rules block from each selected assistant's
config files. Only chiton-owned content is touched; the user's other tool
registrations and notes are preserved.

Supported assistants: ` + supported + `.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			installers, err := install.Resolve(assistant, dir)
			if err != nil {
				return err
			}
			opts := install.Options{Dir: dir}
			out := cmd.OutOrStdout()
			for _, in := range installers {
				rep, err := in.Uninstall(opts)
				if err != nil {
					return fmt.Errorf("%s: %w", in.Name(), err)
				}
				printInstallReport(out, rep, "uninstall")
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&assistant, "assistant", "auto", "Assistant to uninstall from: auto, all, a name, or a comma-separated list")
	cmd.Flags().StringVar(&dir, "dir", ".", "Workspace directory to uninstall from")
	return cmd
}

func doctorCmd() *cobra.Command {
	var dir string
	var asJSON bool

	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Report the install state across all supported AI assistants",
		Long: `For each supported assistant, report whether chiton's MCP entry and rules
block are present in this workspace. Writes nothing.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			out := cmd.OutOrStdout()
			opts := install.Options{Dir: dir}
			reports := make([]install.DoctorReport, 0, len(install.Available()))
			for _, in := range install.Available() {
				rep, err := in.Doctor(opts)
				if err != nil {
					return fmt.Errorf("%s: %w", in.Name(), err)
				}
				reports = append(reports, rep)
			}
			if asJSON {
				b, err := json.MarshalIndent(reports, "", "  ")
				if err != nil {
					return err
				}
				fmt.Fprintln(out, string(b))
				return nil
			}
			for _, rep := range reports {
				status := "not detected"
				if rep.Registered {
					status = "registered"
				} else if rep.Detected {
					status = "detected (not registered)"
				}
				fmt.Fprintf(out, "%s â€” %s\n", rep.Installer, status)
				for _, ok := range rep.OK {
					fmt.Fprintf(out, "  ok:    %s\n", ok)
				}
				for _, issue := range rep.Issues {
					fmt.Fprintf(out, "  issue: %s\n", issue)
				}
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&dir, "dir", ".", "Workspace directory to inspect")
	cmd.Flags().BoolVar(&asJSON, "json", false, "Output JSON")
	return cmd
}

func installerNames() []string {
	avail := install.Available()
	names := make([]string, 0, len(avail))
	for _, i := range avail {
		names = append(names, i.Name())
	}
	return names
}

func printInstallReport(out io.Writer, rep install.Report, verb string) {
	fmt.Fprintf(out, "%s: %s\n", verb, rep.Installer)
	for _, p := range rep.Wrote {
		fmt.Fprintf(out, "  wrote   %s\n", p)
	}
	for _, p := range rep.Skipped {
		fmt.Fprintf(out, "  skip    %s\n", p)
	}
	for _, n := range rep.Notes {
		fmt.Fprintf(out, "  note    %s\n", n)
	}
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
command can run while chiton serve is active â€” no write lock on the store is
needed. Restart chiton serve after clustering to load the new results.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "cluster: reading graph at %s\n", cfg.GraphPath)

			// Open read-only so this command can run while chiton serve is active.
			gs, err := openGraphStore(cfg.GraphPath, true)
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
			fmt.Fprintf(out, "cluster: done in %s â€” %d nodes in %d communities\n",
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

	cmd.Flags().Float64Var(&resolution, "resolution", 1.0, "Leiden resolution parameter Î³ (higher = smaller communities)")
	cmd.Flags().Int64Var(&seed, "seed", 42, "Random seed for reproducible clustering")
	cmd.Flags().IntVar(&topN, "top", 10, "Number of god nodes to print")
	return cmd
}

func serveCmd() *cobra.Command {
	var stdio bool
	var addr string
	var unixPath string
	var writable bool
	var graphPath string

	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Start the MCP server",
		Long: `Start the MCP server over one of three transports (identical tool surface):
  --stdio        assistant-launched (the assistant starts/stops the process)
  --http :7474   HTTP + SSE — a shared daemon across sessions, or for debugging
  --unix <path>  a Unix domain socket — low-latency shared daemon (Linux/macOS)

By default the store opens read-only so 'chiton build', 'embed', and 'cluster'
can run concurrently; the read tools (semantic_search, pending_extractions)
work in this mode. Pass --writable to allow the submit_extraction write tool
over MCP (this takes the store's write lock, so concurrent build/embed cannot
run). The 'chiton extract submit' CLI is the alternative write path.

Pass --graph to point at a specific store; this is what 'chiton install'
records in the MCP config so the server uses the same store as 'chiton build'.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			// --graph overrides the configured/default store path so the MCP
			// config written by `chiton install --graph <path>` resolves to the
			// intended store regardless of the server's working directory.
			if graphPath != "" {
				cfg.GraphPath = graphPath
			}

			gs, err := openGraphStore(cfg.GraphPath, !writable)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			clusterResults, err := cluster.LoadSidecar(cluster.SidecarPath(cfg.GraphPath))
			if err != nil {
				fmt.Fprintf(cmd.ErrOrStderr(), "serve: warning: could not load cluster sidecar: %v\n", err)
			}

			deps := buildSemanticDeps(cmd, cfg, gs, clusterResults)
			s := mcpsrv.New(gs, clusterResults, deps)

			// Transport selection: stdio (assistant-launched), Unix socket
			// (low-latency shared daemon), or HTTP/SSE (default; shared daemon
			// across sessions / debugging). All expose the identical tool surface.
			switch {
			case stdio:
				return mcpsdk.ServeStdio(s)
			case unixPath != "":
				return serveUnix(cmd, mcpsdk.NewSSEServer(s), unixPath)
			default:
				fmt.Fprintf(cmd.OutOrStdout(), "serve: listening on %s (SSE)\n", addr)
				return mcpsdk.NewSSEServer(s).Start(addr)
			}
		},
	}

	cmd.Flags().BoolVar(&stdio, "stdio", false, "serve over stdio (for MCP assistants)")
	cmd.Flags().StringVar(&addr, "http", ":7474", "HTTP/SSE listen address (shared daemon / debug)")
	cmd.Flags().StringVar(&unixPath, "unix", "", "serve over a Unix domain socket at this path (Linux/macOS shared daemon)")
	cmd.Flags().BoolVar(&writable, "writable", false, "open the store read-write so submit_extraction works over MCP (disables concurrent build/embed)")
	cmd.Flags().StringVar(&graphPath, "graph", "", "Graph store path to serve (default: configured/built-in graph_path)")
	return cmd
}

// buildSemanticDeps assembles the Phase 2 MCP services. The retriever is wired
// only when embeddings exist (semantic_search degrades gracefully otherwise);
// the extractor is always available. Embedder construction failures are logged,
// not fatal — the structural tools must still serve.
func buildSemanticDeps(cmd *cobra.Command, cfg *config.Config, gs store.GraphStore, clusterResults *cluster.Results) mcpsrv.Deps {
	errOut := cmd.ErrOrStderr()

	// The LLM-rule and diff tools depend only on the graph store, never on the
	// embedder, so wire them first — a misconfigured embedding provider must not
	// take the Phase 2.5 analysis tools offline.
	var deps mcpsrv.Deps
	deps.LLM = analyze.NewLLMService(gs, nil, cfg.Semantic.ConfidenceThreshold, ".chiton")
	deps.BaselineDir = analyze.BaselineDir(".chiton")
	// Phase 4: meeting→code linkage (read pending always; submit guarded by the
	// store's read-only flag) and URL ingest (add_url; also read-only-guarded).
	deps.MeetingLinker = meeting.NewLinkService(gs, cfg.Semantic.ConfidenceThreshold)
	deps.URLIngester = urlingest.New(gs)

	emb, err := newEmbedder(cfg, "", "")
	if err != nil {
		fmt.Fprintf(errOut, "serve: warning: embeddings disabled: %v\n", err)
		return deps
	}
	deps.Extractor = subagent.New(gs, emb, cfg.Semantic.ConfidenceThreshold)

	es, ok := gs.(store.EmbeddingStore)
	if !ok {
		return deps
	}
	embs, err := es.AllEmbeddings(cmd.Context())
	if err != nil {
		fmt.Fprintf(errOut, "serve: warning: could not load embeddings: %v\n", err)
		return deps
	}
	if len(embs) == 0 {
		fmt.Fprintln(errOut, "serve: note: no embeddings indexed — semantic_search disabled (run 'chiton embed')")
		return deps
	}
	idx := vector.NewFlat()
	for _, e := range embs {
		idx.Add([]vector.Item{{ID: e.NodeID, Vec: e.Vec}})
	}
	deps.Retriever = retrieve.New(gs, idx, emb, clusterResults)
	fmt.Fprintf(errOut, "serve: semantic_search enabled (%d embeddings)\n", len(embs))
	return deps
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
				return fmt.Errorf("no cluster data found â€” run 'chiton cluster' first")
			}

			gs, err := openGraphStore(cfg.GraphPath, true)
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
				fmt.Fprintf(cmd.OutOrStdout(), "report: done â€” %d nodes, %d communities\n",
					clusterResults.NodeCount, clusterResults.CommunityCount)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&outFile, "out", "GRAPH_REPORT.md", "Output file path")
	cmd.Flags().BoolVar(&toStdout, "stdout", false, "Write to stdout instead of a file")
	return cmd
}

func hooksCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "hooks",
		Short: "Install and run chiton's git pre-commit / pre-push hooks",
		Long: `Manage chiton-owned git hooks in this repository.

The committed-to-repo policy lives in .chiton/hooks.yaml. The hook scripts
themselves live under .git/hooks/ and invoke 'chiton hooks run --stage <stage>'.

Subcommands:
  install      write .git/hooks/{pre-commit,pre-push}
  uninstall    remove chiton-owned hook scripts
  status       report which hooks are installed
  run          execute one stage (used by the hook scripts themselves)`,
	}
	cmd.AddCommand(hooksInstallCmd(), hooksUninstallCmd(), hooksStatusCmd(), hooksRunCmd())
	return cmd
}

func hooksInstallCmd() *cobra.Command {
	var dir string
	var stagesFlag []string

	cmd := &cobra.Command{
		Use:   "install",
		Short: "Write chiton-owned git hooks into .git/hooks/",
		RunE: func(cmd *cobra.Command, _ []string) error {
			exe, err := install.ResolveExe("")
			if err != nil {
				return err
			}
			hooksDir, err := resolveGitHooksDir(dir)
			if err != nil {
				return err
			}
			stages, err := parseStages(stagesFlag)
			if err != nil {
				return err
			}
			rep, err := hooks.Install(hooksDir, exe, stages)
			if err != nil {
				return err
			}
			printHooksReport(cmd.OutOrStdout(), rep, "install")
			return nil
		},
	}
	cmd.Flags().StringVar(&dir, "dir", ".", "Repository root containing .git/")
	cmd.Flags().StringSliceVar(&stagesFlag, "stage", []string{"pre-commit", "pre-push"}, "Hook stages to install")
	return cmd
}

func hooksUninstallCmd() *cobra.Command {
	var dir string
	var stagesFlag []string

	cmd := &cobra.Command{
		Use:   "uninstall",
		Short: "Remove chiton-owned hook scripts (foreign hooks are left intact)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			hooksDir, err := resolveGitHooksDir(dir)
			if err != nil {
				return err
			}
			stages, err := parseStages(stagesFlag)
			if err != nil {
				return err
			}
			rep, err := hooks.Uninstall(hooksDir, stages)
			if err != nil {
				return err
			}
			printHooksReport(cmd.OutOrStdout(), rep, "uninstall")
			return nil
		},
	}
	cmd.Flags().StringVar(&dir, "dir", ".", "Repository root containing .git/")
	cmd.Flags().StringSliceVar(&stagesFlag, "stage", []string{"pre-commit", "pre-push"}, "Hook stages to remove")
	return cmd
}

func hooksStatusCmd() *cobra.Command {
	var dir string
	var asJSON bool

	cmd := &cobra.Command{
		Use:   "status",
		Short: "Report which chiton hooks are installed",
		RunE: func(cmd *cobra.Command, _ []string) error {
			hooksDir, err := resolveGitHooksDir(dir)
			if err != nil {
				return err
			}
			entries, err := hooks.Status(hooksDir, hooks.AllStages())
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			if asJSON {
				b, err := json.MarshalIndent(entries, "", "  ")
				if err != nil {
					return err
				}
				fmt.Fprintln(out, string(b))
				return nil
			}
			for _, e := range entries {
				state := "absent"
				switch {
				case e.Exists && e.ChitonOwned:
					state = "chiton-owned"
				case e.Exists:
					state = "present (not chiton-owned)"
				}
				fmt.Fprintf(out, "%-12s %s  %s\n", string(e.Stage), state, e.Path)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&dir, "dir", ".", "Repository root")
	cmd.Flags().BoolVar(&asJSON, "json", false, "Output JSON")
	return cmd
}

func hooksRunCmd() *cobra.Command {
	var dir string
	var stageRaw string

	cmd := &cobra.Command{
		Use:   "run",
		Short: "Run one hook stage (invoked by the installed git hook script)",
		Long: `Read .chiton/hooks.yaml, run the analyzer over the workspace, and exit
non-zero if any finding crosses the stage's block-on threshold.

This is what the installed .git/hooks/pre-commit and pre-push scripts call.
You usually don't invoke it directly.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			stage := hooks.Stage(stageRaw)
			if !slices.Contains(hooks.Stages(), stage) {
				return fmt.Errorf("unknown stage %q â€” supported: pre-commit, pre-push", stageRaw)
			}

			hooksYaml := filepath.Join(dir, ".chiton", "hooks.yaml")
			cfg, err := hooks.Load(hooksYaml)
			if err != nil {
				return err
			}
			stageCfg := cfg.For(stage)

			runCfg, err := config.Load()
			if err != nil {
				return err
			}
			gs, err := openGraphStore(runCfg.GraphPath, false)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			ctx := cmd.Context()
			if stageCfg.TimeoutSeconds > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, time.Duration(stageCfg.TimeoutSeconds)*time.Second)
				defer cancel()
			}

			res, err := hooks.Run(ctx, gs, hooks.RunOptions{
				Stage:  stage,
				Config: stageCfg,
				Root:   dir,
			})
			if err != nil {
				return err
			}
			hooks.FormatResult(cmd.OutOrStdout(), res)
			if res.Blocks() {
				return fmt.Errorf("chiton %s: blocked by %d finding(s) at or above %q",
					res.Stage, res.BlockingCount, res.BlockingMin)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&dir, "dir", ".", "Repository root")
	cmd.Flags().StringVar(&stageRaw, "stage", "pre-commit", "Hook stage to execute")
	return cmd
}

func parseStages(raw []string) ([]hooks.Stage, error) {
	if len(raw) == 0 {
		return hooks.Stages(), nil
	}
	out := make([]hooks.Stage, 0, len(raw))
	for _, s := range raw {
		stage := hooks.Stage(strings.TrimSpace(s))
		if !slices.Contains(hooks.AllStages(), stage) {
			return nil, fmt.Errorf("unknown stage %q (supported: pre-commit, pre-push, post-commit, post-checkout)", s)
		}
		out = append(out, stage)
	}
	return out, nil
}

func resolveGitHooksDir(repoRoot string) (string, error) {
	hooksDir := filepath.Join(repoRoot, ".git", "hooks")
	gitDir := filepath.Join(repoRoot, ".git")
	info, err := os.Stat(gitDir)
	if err != nil {
		return "", fmt.Errorf("no .git in %s â€” not a git repository?", repoRoot)
	}
	if !info.IsDir() {
		// .git can be a file in worktree layouts: "gitdir: <path>"
		b, err := os.ReadFile(gitDir)
		if err != nil {
			return "", fmt.Errorf("read .git file: %w", err)
		}
		const prefix = "gitdir:"
		line := strings.TrimSpace(string(b))
		if !strings.HasPrefix(line, prefix) {
			return "", fmt.Errorf("unrecognised .git file format")
		}
		actual := strings.TrimSpace(strings.TrimPrefix(line, prefix))
		if !filepath.IsAbs(actual) {
			actual = filepath.Join(repoRoot, actual)
		}
		hooksDir = filepath.Join(actual, "hooks")
	}
	return hooksDir, nil
}

func printHooksReport(w io.Writer, rep hooks.InstallReport, verb string) {
	for _, p := range rep.Wrote {
		fmt.Fprintf(w, "%s: wrote %s\n", verb, p)
	}
	for _, p := range rep.Skipped {
		fmt.Fprintf(w, "%s: skip  %s\n", verb, p)
	}
	for _, n := range rep.Notes {
		fmt.Fprintf(w, "%s: note  %s\n", verb, n)
	}
}
