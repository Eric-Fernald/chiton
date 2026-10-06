package analyze

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"os"
	"slices"
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
	fileRules     fileRuleSet // AnalyzeFile's rule list, which the fileRuleSet method builds lazily

	rulesMu   sync.Mutex
	rulesDone map[string]bool // rule node IDs this Analyzer has upserted; rule metadata never changes
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
	Skipped      []string // paths the walk or extractor could not read; the scan went on without them

	rules     []Rule          // rule nodes Record upserts
	scanned   map[string]bool // file URIs whose resolved findings Record prunes
	ruleScope map[string]bool // rule IDs Record may prune; nil means every rule
	prior     *PriorFindings  // prune candidates read before a re-sync; nil queries the store
}

// Analyze runs FileScope rules over each supported file under root, then GraphScope rules over
// the whole extracted set, and upserts every finding plus the rules they violated.
func (a *Analyzer) Analyze(ctx context.Context, root string) (*Result, error) {
	start := a.now()
	res, err := a.Scan(ctx, root)
	if err != nil {
		return nil, err
	}
	if err := a.Record(ctx, res); err != nil {
		return nil, err
	}
	res.Duration = a.now().Sub(start)
	return res, nil
}

// Scan is Analyze without the store: it runs every FileScope and GraphScope rule over the supported files
// under root, stopping once ctx is done, and returns findings that Record can persist.
func (a *Analyzer) Scan(ctx context.Context, root string) (*Result, error) {
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
	graphFindings, err := a.matchGraphRules(ctx, graphRules, w.nodes, w.edges)
	if err != nil {
		return nil, err
	}
	return &Result{
		FilesScanned: w.filesScanned,
		// Count only rules actually executed here; LLM rules are dispatched
		// separately via LLMService and are neither FileRule nor GraphRule.
		RulesRun: len(fileRules) + len(graphRules),
		Findings: append(w.findings, graphFindings...),
		Duration: a.now().Sub(start),
		Skipped:  w.skipped,
		rules:    rules,
		scanned:  w.scannedURIs, // every rule ran, so ruleScope stays nil
	}, nil
}

// SourceFile is one file's path and content for ScanSources, such as a blob staged in the git index.
type SourceFile struct {
	Path    string
	Content []byte
}

// ScanSources runs the FileScope rules over in-memory files without touching the store. GraphScope rules
// need the whole corpus (a partial one reports live symbols as dead), so they are skipped.
func (a *Analyzer) ScanSources(ctx context.Context, files []SourceFile) (*Result, error) {
	start := a.now()
	res := newFileResult()
	for _, sf := range files {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		ex := extast.ForFile(sf.Path)
		if ex == nil {
			continue
		}
		nodes, edges, err := ex.ExtractSource(ctx, sf.Path, sf.Content)
		if err != nil {
			res.Skipped = append(res.Skipped, sf.Path)
			continue
		}
		a.matchFile(ctx, newFileTarget(sf.Path, sf.Content, nodes, edges), res)
	}
	res.Duration = a.now().Sub(start)
	return res, nil
}

// Record upserts res's rule nodes and findings, then deletes the findings of res's files, from the rules
// res ran, that res no longer holds. The Analyzer needs a store.
func (a *Analyzer) Record(ctx context.Context, res *Result) error {
	if a.store == nil {
		return errors.New("analyze: Record needs an Analyzer with a store")
	}
	ids, err := a.persist(ctx, res.rules, res.Findings)
	if err != nil {
		return err
	}
	return a.pruneResolved(ctx, res.scanned, ids, res.ruleScope, res.prior)
}

// walkResult accumulates the per-file outputs of walkAndMatch.
type walkResult struct {
	nodes    []model.Node
	edges    []model.Edge
	findings []Finding
	// scannedURIs limits pruneResolved to the files analysed in this pass, so `analyze <subdir>` leaves
	// findings for other files alone.
	scannedURIs  map[string]bool
	filesScanned int
	skipped      []string
}

// walkAndMatch walks root, extracts each supported source file, and runs the
// applicable FileScope rules over it.
func (a *Analyzer) walkAndMatch(ctx context.Context, root string, fileRules []FileRule) (*walkResult, error) {
	w := &walkResult{scannedURIs: map[string]bool{}}
	langs := make([][]string, len(fileRules))
	for i, fr := range fileRules {
		langs[i] = fr.Meta().Languages
	}
	skipped, walkErr := walkSources(ctx, root, func(path string) error {
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
	w.skipped = skipped
	return w, nil
}

// matchGraphRules runs the GraphScope rules over the full extracted set, stopping once ctx is done.
func (a *Analyzer) matchGraphRules(ctx context.Context, graphRules []GraphRule, nodes []model.Node, edges []model.Edge) ([]Finding, error) {
	gt := GraphTarget{Nodes: nodes, Edges: edges}
	var findings []Finding
	for _, gr := range graphRules {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		findings = append(findings, gr.MatchGraph(ctx, gt)...)
	}
	return findings, nil
}

// persist upserts rule nodes (so VIOLATES edges resolve), finding nodes, and VIOLATES and AFFECTS edges.
// It returns the set of finding IDs so callers need not hash each finding again.
func (a *Analyzer) persist(ctx context.Context, rules []Rule, findings []Finding) (map[string]bool, error) {
	now := a.now()

	nodes := make([]model.Node, 0, len(rules)+len(findings))
	var fresh []string // rule nodes this call upserts for the first time
	for _, r := range rules {
		n := RuleAsNode(r.Meta(), now)
		if a.rulePersisted(n.ID) {
			continue
		}
		fresh = append(fresh, n.ID)
		nodes = append(nodes, n)
	}
	edges := make([]model.Edge, 0, len(findings)*2)
	ids := make(map[string]bool, len(findings))
	for _, f := range findings {
		id := f.FindingID()
		ids[id] = true
		nodes, edges = f.appendGraph(nodes, edges, id, RuleNodeID(f.RuleID), now)
	}
	if err := a.store.UpsertNodes(ctx, nodes); err != nil {
		return nil, fmt.Errorf("upsert nodes: %w", err)
	}
	if err := a.store.UpsertEdges(ctx, edges); err != nil {
		return nil, fmt.Errorf("upsert edges: %w", err)
	}
	a.markRulesPersisted(fresh)
	return ids, nil
}

// rulePersisted reports whether this Analyzer already upserted rule node id.
func (a *Analyzer) rulePersisted(id string) bool {
	a.rulesMu.Lock()
	defer a.rulesMu.Unlock()
	return a.rulesDone[id]
}

func (a *Analyzer) markRulesPersisted(ids []string) {
	a.rulesMu.Lock()
	defer a.rulesMu.Unlock()
	if a.rulesDone == nil {
		a.rulesDone = make(map[string]bool, len(ids))
	}
	for _, id := range ids {
		a.rulesDone[id] = true
	}
}

// pruneByURIMax is the scanned-file count up to which pruneResolved queries findings per target_uri, so the
// store's byte probe skips other files' records undecoded; larger passes load every finding once.
const pruneByURIMax = 8

// pruneResolved deletes a scanned file's deterministic findings that are missing from newIDs. LLM findings
// are pruned separately. A non-nil ruleScope limits the delete to the rule IDs that ran, and a non-nil prior
// supplies the candidates instead of a store query.
func (a *Analyzer) pruneResolved(ctx context.Context, scanned, newIDs, ruleScope map[string]bool, prior *PriorFindings) error {
	if len(scanned) == 0 {
		return nil
	}
	var findings []model.Node
	if prior != nil {
		findings = prior.nodes
	} else {
		var err error
		if findings, err = a.deterministicFindings(ctx, scanned); err != nil {
			return fmt.Errorf("load nodes for prune: %w", err)
		}
	}
	var stale []model.Node
	for _, n := range findings {
		if ruleScope != nil && !ruleScope[model.AttrString(n.Attrs, "rule_id")] {
			continue // this pass did not run the rule, so leave its findings intact
		}
		if !scanned[model.AttrString(n.Attrs, "target_uri")] || newIDs[n.ID] {
			continue
		}
		stale = append(stale, n)
	}
	return deleteFindingNodes(ctx, a.store, stale)
}

// deterministicFindings loads the deterministic findings that may target a scanned file.
func (a *Analyzer) deterministicFindings(ctx context.Context, scanned map[string]bool) ([]model.Node, error) {
	filter := func(uri string) store.NodeFilter {
		attrs := map[string]string{"rule_provenance": string(ProvenanceDeterministic)}
		if uri != "" {
			attrs["target_uri"] = uri
		}
		return store.NodeFilter{Kinds: []model.NodeKind{model.KindFinding}, Attrs: attrs}
	}
	if len(scanned) > pruneByURIMax {
		return store.NodesMatching(ctx, a.store, filter(""))
	}
	var out []model.Node
	for _, uri := range slices.Sorted(maps.Keys(scanned)) {
		ns, err := store.NodesMatching(ctx, a.store, filter(uri))
		if err != nil {
			return nil, err
		}
		out = append(out, ns...)
	}
	return out, nil
}

// deleteFindingNodes removes the nodes and their edges via store.DeleteNodes. Finding source URIs are unique, so
// both of its paths delete exactly these nodes.
func deleteFindingNodes(ctx context.Context, gs store.GraphStore, nodes []model.Node) error {
	if _, err := store.DeleteNodes(ctx, gs, nodes); err != nil {
		return fmt.Errorf("prune findings: %w", err)
	}
	return nil
}

// loadFileTarget reads and extracts path. ok is false for an unsupported, unreadable, or unparsable
// file. nodes is the full extraction, including the file node that ft.Nodes omits.
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
