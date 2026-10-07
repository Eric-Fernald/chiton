// Package bench implements chiton's token-cost benchmark (4 chars per token). It costs each dependency
// question as naive (every file), grep+read (files naming the symbol), chiton's lean MCP neighborhood at
// depth 2 and depth 1, and the query_graph lookup that finds the symbol.
package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/walk"
)

// queryGraphLimit is query_graph's default result limit, so the lookup column costs what the tool returns.
const queryGraphLimit = 10

// estimateTokens approximates token count from character count (chars ÷ 4), the
// standard rough estimate; identical across strategies so ratios are fair.
func estimateTokens(chars int) int { return chars / 4 }

// QueryResult is the token cost of one question under each strategy. Chiton is the depth-2 neighborhood,
// ChitonDepth1 the depth-1 one, and QueryGraph the query_graph lookup for the symbol's name, which finds
// the symbol but returns no edges.
type QueryResult struct {
	Question     string `json:"question"`
	Naive        int    `json:"naive"`
	GrepRead     int    `json:"grep_read"`
	Chiton       int    `json:"chiton"`
	ChitonDepth1 int    `json:"chiton_depth1"`
	QueryGraph   int    `json:"query_graph"`
}

// Options controls how Run picks its questions.
type Options struct {
	// MaxQueries is how many of the highest-degree symbols to benchmark (default 5).
	MaxQueries int
	// IncludeTests lets symbols defined in test files be query targets. Test files are always part of
	// the corpus, so they still count toward naive and grep+read; this only changes which symbols get asked about.
	IncludeTests bool
}

// Report is the full benchmark output for one corpus.
type Report struct {
	Name         string        `json:"name"`
	Path         string        `json:"path"`
	Files        int           `json:"files"`
	Nodes        int           `json:"nodes"`
	Edges        int           `json:"edges"`
	CorpusTokens int           `json:"corpus_tokens"`
	Depth        int           `json:"depth"`
	IncludeTests bool          `json:"include_tests"`
	Queries      []QueryResult `json:"queries"`
}

// leanNode and leanEdge mirror internal/mcp's lean shapes field-for-field, never omitting Provenance or
// Confidence, so a smaller payload can't understate the chiton column.
type leanNode struct {
	Kind       model.NodeKind    `json:"kind"`
	Label      string            `json:"label"`
	SourceURI  string            `json:"source_uri"`
	Span       *model.SourceSpan `json:"span,omitempty"`
	Attrs      map[string]any    `json:"attrs,omitempty"`
	Provenance model.Provenance  `json:"provenance"`
	Confidence float32           `json:"confidence"`
}

type leanEdge struct {
	Kind  model.EdgeKind `json:"kind"`
	From  string         `json:"from"`
	To    string         `json:"to"`
	Attrs map[string]any `json:"attrs,omitempty"`
}

// toLeanNode projects a model.Node to the MCP server's lean node shape.
func toLeanNode(n model.Node) leanNode {
	return leanNode{
		Kind:       n.Kind,
		Label:      n.Label,
		SourceURI:  n.SourceURI,
		Span:       n.Span,
		Attrs:      n.Attrs,
		Provenance: n.Provenance,
		Confidence: n.Confidence,
	}
}

// Run builds an in-memory graph over the supported files under path and costs the neighborhoods of its
// highest-degree symbols. name labels the report.
func Run(ctx context.Context, name, path string, opts Options) (*Report, error) {
	maxQueries := opts.MaxQueries
	if maxQueries <= 0 {
		maxQueries = 5
	}
	const depth = 2

	var (
		nodes      []model.Node
		edges      []model.Edge
		fileTokens = map[string]int{} // abs path → token estimate of its content
		fileText   = map[string]string{}
		files      int
	)

	policy, err := walk.LoadPolicy(path)
	if err != nil {
		return nil, err
	}
	// Unreadable entries are skipped so the benchmark keeps going over the rest.
	_, walkErr := walk.Files(ctx, path, policy, func(p string) error {
		ex := extast.ForFile(p)
		if ex == nil {
			return nil
		}
		content, err := os.ReadFile(p)
		if err != nil {
			return nil
		}
		files++
		fileText[p] = string(content)
		fileTokens[p] = estimateTokens(len(content))

		ns, es, err := ex.ExtractFile(ctx, p)
		if err != nil {
			return nil
		}
		nodes = append(nodes, ns...)
		edges = append(edges, es...)
		return nil
	}, nil)
	if walkErr != nil {
		return nil, walkErr
	}
	if files == 0 {
		return nil, fmt.Errorf("no supported files under %s", path)
	}

	corpusTokens := 0
	for _, t := range fileTokens {
		corpusTokens += t
	}

	nodes, edges = resolveFuncRefs(nodes, edges)
	g := newGraph(nodes, edges)
	targets := g.topSymbols(maxQueries, opts.IncludeTests)

	rep := &Report{
		Name:         name,
		Path:         path,
		Files:        files,
		Nodes:        len(nodes),
		Edges:        len(edges),
		CorpusTokens: corpusTokens,
		Depth:        depth,
		IncludeTests: opts.IncludeTests,
	}

	for _, t := range targets {
		d2, err := g.neighborhoodTokens(t, depth)
		if err != nil {
			return nil, err
		}
		d1, err := g.neighborhoodTokens(t, 1)
		if err != nil {
			return nil, err
		}
		qg, err := g.queryGraphTokens(baseIdent(g.nodes[t].Label))
		if err != nil {
			return nil, err
		}
		rep.Queries = append(rep.Queries, QueryResult{
			Question:     fmt.Sprintf("What does %s depend on / connect to?", g.nodes[t].Label),
			Naive:        corpusTokens,
			GrepRead:     grepReadTokens(g.nodes[t].Label, fileText, fileTokens),
			Chiton:       d2,
			ChitonDepth1: d1,
			QueryGraph:   qg,
		})
	}
	return rep, nil
}

// resolveFuncRefs merges each pkg:// function reference into the definition it names, as the MCP server does
// when it walks the graph, so a neighborhood here holds the same nodes the server would return: the callers in
// other files and packages, attached to the definition. A reference naming no definition, or more than one
// (build-tag variants), stays as it is.
func resolveFuncRefs(nodes []model.Node, edges []model.Edge) ([]model.Node, []model.Edge) {
	importPath := map[string]string{} // file URI → import path
	for _, n := range nodes {
		if n.Kind == model.KindFile {
			if ip := model.AttrString(n.Attrs, model.AttrImportPath); ip != "" {
				importPath[n.SourceURI] = ip
			}
		}
	}
	defs := map[string][]string{} // reference ID → definition IDs
	for _, n := range nodes {
		ip := importPath[n.SourceURI]
		if ip == "" || n.Kind != model.KindSymbol || model.IsExternal(n) || strings.Contains(n.Label, ".") ||
			model.AttrString(n.Attrs, "lang_kind") != "func" {
			continue
		}
		ref := model.PkgFuncRefID(ip, n.Label)
		defs[ref] = append(defs[ref], n.ID)
	}
	canon := func(id string) string {
		if ds := defs[id]; len(ds) == 1 {
			return ds[0]
		}
		return id
	}
	keptNodes := nodes[:0:0]
	for _, n := range nodes {
		if canon(n.ID) == n.ID {
			keptNodes = append(keptNodes, n)
		}
	}
	seen := map[string]bool{}
	keptEdges := edges[:0:0]
	for _, e := range edges {
		e.Src, e.Dst = canon(e.Src), canon(e.Dst)
		key := e.Src + "\x00" + e.Dst + "\x00" + string(e.Kind)
		if e.Src == e.Dst || seen[key] {
			continue
		}
		seen[key] = true
		keptEdges = append(keptEdges, e)
	}
	return keptNodes, keptEdges
}

// grepReadTokens models "grep -lw <symbol> | xargs cat" on the base identifier (after the last '.'),
// since qualified labels like "LinkService.Submit" never appear verbatim in source.
func grepReadTokens(label string, fileText map[string]string, fileTokens map[string]int) int {
	needle := baseIdent(label)
	total := 0
	for path, text := range fileText {
		if containsWord(text, needle) {
			total += fileTokens[path]
		}
	}
	return total
}

// containsWord reports whether word occurs in text as a whole identifier, the way `grep -w` or `rg -w` would
// match it, so "Open" finds Open( but not OpenReadOnly.
func containsWord(text, word string) bool {
	if word == "" {
		return false
	}
	for i := 0; ; {
		j := strings.Index(text[i:], word)
		if j < 0 {
			return false
		}
		start, end := i+j, i+j+len(word)
		if (start == 0 || !isIdentByte(text[start-1])) && (end == len(text) || !isIdentByte(text[end])) {
			return true
		}
		i = start + 1
	}
}

func isIdentByte(c byte) bool {
	return c == '_' || c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= 0x80
}

// baseIdent returns the identifier after the last '.', e.g. "LinkService.Submit"
// → "Submit", "http.Get" → "Get", "node" → "node".
func baseIdent(label string) string {
	if i := strings.LastIndex(label, "."); i >= 0 && i+1 < len(label) {
		return label[i+1:]
	}
	return label
}

// graph is a minimal in-memory adjacency over the extracted nodes/edges, enough
// to rank by degree and walk depth-bounded neighborhoods.
type graph struct {
	nodes     map[string]model.Node
	adj       map[string][]int // node ID → indices into edges (both directions)
	edges     []model.Edge
	degree    map[string]int
	sortedIDs []string // node IDs in store key order, built on first queryGraphTokens call
}

func newGraph(nodes []model.Node, edges []model.Edge) *graph {
	g := &graph{
		nodes:  make(map[string]model.Node, len(nodes)),
		adj:    make(map[string][]int),
		edges:  edges,
		degree: make(map[string]int),
	}
	for _, n := range nodes {
		g.nodes[n.ID] = n
	}
	for i, e := range edges {
		g.adj[e.Src] = append(g.adj[e.Src], i)
		g.adj[e.Dst] = append(g.adj[e.Dst], i)
		g.degree[e.Src]++
		g.degree[e.Dst]++
	}
	return g
}

// topSymbols returns the IDs of the n highest-degree symbol nodes, ties broken by ID for reproducibility.
// Unless includeTests is set, symbols defined in test files are skipped: a busy test helper is rarely what
// a developer asks about, and its wide neighborhood would dominate the sample.
func (g *graph) topSymbols(n int, includeTests bool) []string {
	type ranked struct {
		id  string
		deg int
	}
	var rs []ranked
	for id, node := range g.nodes {
		if node.Kind != model.KindSymbol {
			continue
		}
		if g.degree[id] == 0 {
			continue
		}
		// Skip external import targets (fmt, http.Get). Their huge degree doesn't match any real question
		// and would inflate every column.
		if model.IsExternal(node) {
			continue
		}
		if !includeTests && analyze.IsTestSource(node.SourceURI) {
			continue
		}
		rs = append(rs, ranked{id, g.degree[id]})
	}
	sort.Slice(rs, func(i, j int) bool {
		if rs[i].deg != rs[j].deg {
			return rs[i].deg > rs[j].deg
		}
		return rs[i].id < rs[j].id
	})
	if len(rs) > n {
		rs = rs[:n]
	}
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = r.id
	}
	return out
}

// queryGraphTokens costs the query_graph payload for query: the first queryGraphLimit nodes, in ID order as
// the store returns them, whose label contains query case-insensitively, as a JSON array of lean nodes.
func (g *graph) queryGraphTokens(query string) (int, error) {
	if g.sortedIDs == nil {
		g.sortedIDs = make([]string, 0, len(g.nodes))
		for id := range g.nodes {
			g.sortedIDs = append(g.sortedIDs, id)
		}
		sort.Strings(g.sortedIDs)
	}
	q := strings.ToLower(query)
	lean := []leanNode{}
	for _, id := range g.sortedIDs {
		n := g.nodes[id]
		if !strings.Contains(strings.ToLower(n.Label), q) {
			continue
		}
		lean = append(lean, toLeanNode(n))
		if len(lean) == queryGraphLimit {
			break
		}
	}
	b, err := json.Marshal(lean)
	if err != nil {
		return 0, err
	}
	return estimateTokens(len(b)), nil
}

// neighborhoodTokens computes the token cost of the lean MCP neighborhood
// payload for seed within depth hops (undirected BFS, deduplicated).
func (g *graph) neighborhoodTokens(seed string, depth int) (int, error) {
	visited := map[string]bool{seed: true}
	frontier := []string{seed}
	seenEdge := map[int]bool{}
	var neighborNodes []leanNode
	var leanEdges []leanEdge

	for d := 0; d < depth && len(frontier) > 0; d++ {
		var next []string
		for _, id := range frontier {
			for _, ei := range g.adj[id] {
				if seenEdge[ei] {
					continue
				}
				seenEdge[ei] = true
				e := g.edges[ei]
				leanEdges = append(leanEdges, leanEdge{
					Kind:  e.Kind,
					From:  g.nodes[e.Src].Label,
					To:    g.nodes[e.Dst].Label,
					Attrs: e.Attrs,
				})
				for _, other := range []string{e.Src, e.Dst} {
					if !visited[other] {
						visited[other] = true
						next = append(next, other)
						neighborNodes = append(neighborNodes, toLeanNode(g.nodes[other]))
					}
				}
			}
		}
		frontier = next
	}

	sn := g.nodes[seed]
	payload := struct {
		Seed    leanNode   `json:"seed"`
		Nodes   []leanNode `json:"nodes"`
		Edges   []leanEdge `json:"edges"`
		Summary string     `json:"summary"`
	}{
		Seed:  toLeanNode(sn),
		Nodes: neighborNodes,
		Edges: leanEdges,
		// This matches the MCP neighborhood handler's summary; both marshal compact JSON.
		Summary: fmt.Sprintf("%d nodes, %d edges within depth %d of %q", len(neighborNodes), len(leanEdges), depth, sn.Label),
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return 0, err
	}
	return estimateTokens(len(b)), nil
}
