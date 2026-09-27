package analyze

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
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

	fileRulesOnce sync.Once
	fileRules     fileRuleSet // AnalyzeFile's rule list, built lazily by fileRuleSet
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

// Analyze runs FileScope rules over each supported file under root, then GraphScope rules over
// the whole extracted set, and upserts every finding plus the rules they violated.
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

	ids, err := a.persist(ctx, rules, findings)
	if err != nil {
		return nil, err
	}
	// A full Analyze runs every rule, so all deterministic findings for the
	// scanned files are in scope to prune (ruleScope nil = no restriction).
	if err := a.pruneResolved(ctx, w.scannedURIs, ids, nil); err != nil {
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
	// scannedURIs scopes pruneResolved to files re-analysed now, so `analyze <subdir>` spares the rest.
	scannedURIs  map[string]bool
	filesScanned int
}

// walkAndMatch walks root, extracts each supported source file, and runs the
// applicable FileScope rules over it.
func (a *Analyzer) walkAndMatch(ctx context.Context, root string, fileRules []FileRule) (*walkResult, error) {
	w := &walkResult{scannedURIs: map[string]bool{}}
	langs := make([][]string, len(fileRules))
	for i, fr := range fileRules {
		langs[i] = fr.Meta().Languages
	}
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
		ft, nodes, ok := loadFileTarget(ctx, path)
		if !ok {
			return nil
		}
		w.filesScanned++
		if ft.URI != "" {
			w.scannedURIs[ft.URI] = true
		}

		ext := FileExt(path)
		for i, fr := range fileRules {
			if ruleApplies(langs[i], ext) {
				w.findings = append(w.findings, fr.MatchFile(ctx, ft)...)
			}
		}

		w.nodes = append(w.nodes, nodes...)
		w.edges = append(w.edges, ft.Edges...)
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

// persist upserts rule nodes (so VIOLATES edges resolve), finding nodes, and VIOLATES/AFFECTS edges,
// returning the set of finding IDs so each is hashed once.
func (a *Analyzer) persist(ctx context.Context, rules []Rule, findings []Finding) (map[string]bool, error) {
	now := a.now()

	nodes := make([]model.Node, 0, len(rules)+len(findings))
	ruleNodes := make(map[string]string, len(rules)) // rule ID -> RuleNodeID
	for _, r := range rules {
		n := RuleAsNode(r.Meta(), now)
		ruleNodes[n.Label] = n.ID
		nodes = append(nodes, n)
	}
	edges := make([]model.Edge, 0, len(findings)*2)
	ids := make(map[string]bool, len(findings))
	for _, f := range findings {
		id := f.FindingID()
		ids[id] = true
		rule, ok := ruleNodes[f.RuleID]
		if !ok {
			rule = RuleNodeID(f.RuleID)
			ruleNodes[f.RuleID] = rule
		}
		nodes, edges = f.appendGraph(nodes, edges, id, rule, now)
	}
	if err := a.store.UpsertNodes(ctx, nodes); err != nil {
		return nil, fmt.Errorf("upsert nodes: %w", err)
	}
	if err := a.store.UpsertEdges(ctx, edges); err != nil {
		return nil, fmt.Errorf("upsert edges: %w", err)
	}
	return ids, nil
}

// pruneResolved deletes deterministic findings not in newIDs for a scanned file (LLM findings keep
// their own lifecycle); a non-nil ruleScope limits it to the rule IDs that actually ran.
func (a *Analyzer) pruneResolved(ctx context.Context, scanned map[string]bool, newIDs map[string]bool, ruleScope map[string]bool) error {
	if len(scanned) == 0 {
		return nil
	}
	findings, err := store.NodesMatching(ctx, a.store, store.NodeFilter{
		Kinds: []model.NodeKind{model.KindFinding},
		Attrs: map[string]string{"rule_provenance": string(ProvenanceDeterministic)},
	})
	if err != nil {
		return fmt.Errorf("load nodes for prune: %w", err)
	}
	var stale []model.Node
	for _, n := range findings {
		if ruleScope != nil && !ruleScope[model.AttrString(n.Attrs, "rule_id")] {
			continue // a rule not run in this pass — leave its findings intact
		}
		if !scanned[model.AttrString(n.Attrs, "target_uri")] || newIDs[n.ID] {
			continue
		}
		stale = append(stale, n)
	}
	return deleteFindingNodes(ctx, a.store, stale)
}

// deleteFindingNodes removes the nodes (and edges) via NodeDeleter when available, else per-node
// DeleteBySourceURI; finding source URIs are unique, so both delete exactly these nodes.
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

// loadFileTarget reads and extracts path; ok is false for unsupported, unreadable, or unparsable
// files. nodes is the full extraction, including the file node that ft.Nodes omits.
func loadFileTarget(ctx context.Context, path string) (ft FileTarget, nodes []model.Node, ok bool) {
	ex := extast.ForFile(path)
	if ex == nil {
		return ft, nil, false
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return ft, nil, false
	}
	nodes, edges, err := ex.ExtractSource(ctx, path, content)
	if err != nil {
		return ft, nil, false
	}
	return newFileTarget(path, content, nodes, edges), nodes, true
}

// newFileTarget builds path's FileTarget from its content and full extraction (file node included).
func newFileTarget(path string, content []byte, nodes []model.Node, edges []model.Edge) FileTarget {
	fileNode := findFileNode(nodes)
	return FileTarget{
		Path:     path,
		URI:      fileNode.SourceURI,
		Content:  content,
		FileNode: fileNode,
		Nodes:    nonFileNodes(nodes),
		Edges:    edges,
		memo:     &fileMemo{m: map[string]any{}},
	}
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
