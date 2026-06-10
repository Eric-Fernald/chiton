package analyze

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// Analyzer runs registered rules against a directory tree and persists the
// resulting findings (and rule metadata) as graph nodes/edges.
type Analyzer struct {
	store    store.GraphStore
	registry *Registry
	now      func() time.Time
}

// New returns an Analyzer wired to the given store and rule registry.
// Pass nil for registry to use the package-default registry.
func New(s store.GraphStore, reg *Registry) *Analyzer {
	if reg == nil {
		reg = DefaultRegistry()
	}
	return &Analyzer{
		store:    s,
		registry: reg,
		now:      time.Now,
	}
}

// Result summarises one analyze run.
type Result struct {
	FilesScanned int
	RulesRun     int
	Findings     []Finding
	Duration     time.Duration
}

// Analyze walks root, extracts each supported source file, runs FileScope
// rules over each file, then runs GraphScope rules over the full extracted
// set. All findings (and the rules they violated) are upserted into the store.
func (a *Analyzer) Analyze(ctx context.Context, root string) (*Result, error) {
	start := a.now()
	rules := a.registry.All()

	var fileRules []FileRule
	var graphRules []GraphRule
	for _, r := range rules {
		switch v := r.(type) {
		case FileRule:
			fileRules = append(fileRules, v)
		case GraphRule:
			graphRules = append(graphRules, v)
		}
	}

	w, err := a.walkAndMatch(ctx, root, fileRules)
	if err != nil {
		return nil, err
	}
	findings := append(w.findings, a.matchGraphRules(ctx, graphRules, w.nodes, w.edges)...)

	if err := a.persist(ctx, rules, findings); err != nil {
		return nil, err
	}
	// A full Analyze runs every rule, so all deterministic findings for the
	// scanned files are in scope to prune (ruleScope nil = no restriction).
	if err := a.pruneResolved(ctx, w.scannedURIs, findings, nil); err != nil {
		return nil, err
	}

	return &Result{
		FilesScanned: w.filesScanned,
		// Count only rules actually executed here; LLM rules are dispatched
		// separately via LLMService and are neither FileRule nor GraphRule.
		RulesRun: len(fileRules) + len(graphRules),
		Findings: findings,
		Duration: a.now().Sub(start),
	}, nil
}

// walkResult accumulates the per-file outputs of walkAndMatch.
type walkResult struct {
	nodes    []model.Node
	edges    []model.Edge
	findings []Finding
	// scannedURIs records every file extracted this run, so pruneResolved
	// only removes stale findings for files actually re-analysed (a partial
	// `analyze <subdir>` never touches findings outside its scope).
	scannedURIs  map[string]bool
	filesScanned int
}

// walkAndMatch walks root, extracts each supported source file, and runs the
// applicable FileScope rules over it.
func (a *Analyzer) walkAndMatch(ctx context.Context, root string, fileRules []FileRule) (*walkResult, error) {
	w := &walkResult{scannedURIs: map[string]bool{}}
	walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// Skip common noise directories
			if skipDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		ex := extast.ForFile(path)
		if ex == nil {
			return nil
		}
		content, err := os.ReadFile(path)
		if err != nil {
			return nil // skip unreadable
		}
		nodes, edges, err := ex.ExtractFile(ctx, path)
		if err != nil {
			return nil
		}
		w.filesScanned++

		fileNode := findFileNode(nodes)
		if fileNode.SourceURI != "" {
			w.scannedURIs[fileNode.SourceURI] = true
		}
		ft := FileTarget{
			Path:     path,
			URI:      fileNode.SourceURI,
			Content:  content,
			FileNode: fileNode,
			Nodes:    nonFileNodes(nodes),
			Edges:    edges,
		}

		ext := fileExt(path)
		for _, fr := range fileRules {
			if !ruleApplies(fr.Meta().Languages, ext) {
				continue
			}
			w.findings = append(w.findings, fr.MatchFile(ctx, ft)...)
		}

		w.nodes = append(w.nodes, nodes...)
		w.edges = append(w.edges, edges...)
		return nil
	})
	if walkErr != nil {
		return nil, fmt.Errorf("walk: %w", walkErr)
	}
	return w, nil
}

// matchGraphRules runs the GraphScope rules over the full extracted set.
func (a *Analyzer) matchGraphRules(ctx context.Context, graphRules []GraphRule, nodes []model.Node, edges []model.Edge) []Finding {
	gt := GraphTarget{Nodes: nodes, Edges: edges}
	var findings []Finding
	for _, gr := range graphRules {
		findings = append(findings, gr.MatchGraph(ctx, gt)...)
	}
	return findings
}

// persist upserts rule nodes (so VIOLATES edges resolve), finding nodes,
// and VIOLATES/AFFECTS edges.
func (a *Analyzer) persist(ctx context.Context, rules []Rule, findings []Finding) error {
	now := a.now()

	nodes := make([]model.Node, 0, len(rules)+len(findings))
	for _, r := range rules {
		nodes = append(nodes, RuleAsNode(r.Meta(), now))
	}
	for _, f := range findings {
		nodes = append(nodes, f.AsNode(now))
	}
	if err := a.store.UpsertNodes(ctx, nodes); err != nil {
		return fmt.Errorf("upsert nodes: %w", err)
	}

	edges := make([]model.Edge, 0, len(findings)*2)
	for _, f := range findings {
		edges = append(edges, f.ViolatesEdge(now))
		if ae := f.AffectsEdge(now); ae.ID != "" {
			edges = append(edges, ae)
		}
	}
	if err := a.store.UpsertEdges(ctx, edges); err != nil {
		return fmt.Errorf("upsert edges: %w", err)
	}
	return nil
}

// pruneResolved deletes deterministic findings that are no longer produced for
// a re-scanned file, so the store reflects live truth (and `--diff` can detect
// resolved findings). LLM findings are left alone — they have their own
// lifecycle via LLMService — as are findings for files outside this scan.
//
// ruleScope, when non-nil, restricts pruning to findings whose rule_id is in the
// set. The incremental AnalyzeFile runs only FileRules, so it passes the IDs of
// the rules it ran: without this, GraphScope findings (architecture/dead-symbol)
// — which target the same file URI and are also DETERMINISTIC — would be wiped on
// every single-file edit and never regenerate until a full `chiton analyze`. A
// nil scope (full Analyze, which runs every rule) prunes all deterministic kinds.
func (a *Analyzer) pruneResolved(ctx context.Context, scanned map[string]bool, current []Finding, ruleScope map[string]bool) error {
	if len(scanned) == 0 {
		return nil
	}
	newIDs := make(map[string]bool, len(current))
	for _, f := range current {
		newIDs[f.FindingID()] = true
	}
	all, err := a.store.AllNodes(ctx)
	if err != nil {
		return fmt.Errorf("load nodes for prune: %w", err)
	}
	var stale []model.Node
	for _, n := range all {
		if n.Kind != model.KindFinding {
			continue
		}
		if rp, _ := n.Attrs["rule_provenance"].(string); rp != string(ProvenanceDeterministic) {
			continue
		}
		if ruleScope != nil {
			if rid, _ := n.Attrs["rule_id"].(string); !ruleScope[rid] {
				continue // a rule not run in this pass — leave its findings intact
			}
		}
		turi, _ := n.Attrs["target_uri"].(string)
		if !scanned[turi] || newIDs[n.ID] {
			continue
		}
		stale = append(stale, n)
	}
	return deleteFindingNodes(ctx, a.store, stale)
}

// deleteFindingNodes removes the given finding nodes (and their edges) using the
// store's batch NodeDeleter capability when available, falling back to per-node
// DeleteBySourceURI otherwise. Finding source URIs are unique, so both paths
// delete exactly the intended nodes.
func deleteFindingNodes(ctx context.Context, gs store.GraphStore, nodes []model.Node) error {
	if len(nodes) == 0 {
		return nil
	}
	if d, ok := gs.(store.NodeDeleter); ok {
		ids := make([]string, len(nodes))
		for i, n := range nodes {
			ids[i] = n.ID
		}
		if _, err := d.DeleteByIDs(ctx, ids); err != nil {
			return fmt.Errorf("prune findings: %w", err)
		}
		return nil
	}
	for _, n := range nodes {
		if _, err := gs.DeleteBySourceURI(ctx, n.SourceURI); err != nil {
			return fmt.Errorf("prune finding %s: %w", n.ID, err)
		}
	}
	return nil
}

func findFileNode(nodes []model.Node) model.Node {
	for _, n := range nodes {
		if n.Kind == model.KindFile {
			return n
		}
	}
	return model.Node{}
}

func nonFileNodes(nodes []model.Node) []model.Node {
	out := make([]model.Node, 0, len(nodes))
	for _, n := range nodes {
		if n.Kind != model.KindFile {
			out = append(out, n)
		}
	}
	return out
}

// fileExt returns the lowercase extension without the leading dot.
func fileExt(path string) string {
	e := strings.ToLower(filepath.Ext(path))
	if strings.HasPrefix(e, ".") {
		return e[1:]
	}
	return e
}

// ruleApplies returns true if the rule's language filter accepts the file.
// nil/empty filter accepts everything.
func ruleApplies(langs []string, ext string) bool {
	if len(langs) == 0 {
		return true
	}
	for _, l := range langs {
		if strings.EqualFold(l, ext) {
			return true
		}
	}
	return false
}
