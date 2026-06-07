// Package bench implements chiton's honest token-cost benchmark: for a corpus,
// it answers a deterministic set of dependency questions three ways and reports
// the token cost of each, so the README/docs can publish a table rather than a
// single hero number.
//
// The three strategies model how an assistant actually gets context:
//
//   - naive     — read every file in scope (what a model does with no tooling).
//   - grep+read — read the full contents of files that mention the symbol
//     (the grep_search + read_file default of Copilot / Claude Code).
//   - chiton    — the lean neighborhood payload chiton's MCP tools return.
//
// Token cost uses the same "chars ÷ 4" estimate as the Phase 0 benchmark
// (PHASE0_BENCHMARK.md) so every strategy is measured identically. The graph is
// built in memory from the AST extractors — no store, no MCP server — and the
// chiton payload mirrors the lean node/edge shape the MCP server emits.
package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// estimateTokens approximates token count from character count (chars ÷ 4), the
// standard rough estimate; identical across strategies so ratios are fair.
func estimateTokens(chars int) int { return chars / 4 }

// QueryResult is the token cost of one question under each strategy.
type QueryResult struct {
	Question string `json:"question"`
	Naive    int    `json:"naive"`
	GrepRead int    `json:"grep_read"`
	Chiton   int    `json:"chiton"`
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
	Queries      []QueryResult `json:"queries"`
}

// leanNode/leanEdge mirror the EXACT shape the MCP server returns (see
// internal/mcp/server.go leanNode/leanEdge and toLeanNode/toLeanEdge) so the
// chiton column measures what the model actually receives — fields and all.
// Attrs/Provenance/Confidence are part of the real payload (Provenance and
// Confidence are not omitempty and are always set), so omitting them would
// understate chiton's token cost and flatter the benchmark.
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

// Run builds an in-memory graph over the supported files under path and
// measures the token cost of the maxQueries highest-degree symbols' dependency
// neighborhoods (depth 2). name labels the corpus in the report.
func Run(ctx context.Context, name, path string, maxQueries int) (*Report, error) {
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

	walkErr := filepath.WalkDir(path, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable entries, keep benchmarking the rest
		}
		ex := extast.ForFile(p)
		if d.IsDir() || ex == nil {
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
	})
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

	g := newGraph(nodes, edges)
	targets := g.topSymbols(maxQueries)

	rep := &Report{
		Name:         name,
		Path:         path,
		Files:        files,
		Nodes:        len(nodes),
		Edges:        len(edges),
		CorpusTokens: corpusTokens,
		Depth:        depth,
	}

	for _, t := range targets {
		chitonTokens, err := g.neighborhoodTokens(t, depth)
		if err != nil {
			return nil, err
		}
		rep.Queries = append(rep.Queries, QueryResult{
			Question: fmt.Sprintf("What does %s depend on / connect to?", g.nodes[t].Label),
			Naive:    corpusTokens,
			GrepRead: grepReadTokens(g.nodes[t].Label, fileText, fileTokens),
			Chiton:   chitonTokens,
		})
	}
	return rep, nil
}

// grepReadTokens models "grep -l <symbol> | xargs cat": the token cost of
// reading every file that mentions the symbol. It greps the *base* identifier a
// developer would actually type — the part after the last '.' — because chiton
// stores qualified method labels (e.g. "LinkService.Submit") that never appear
// verbatim in source. Grepping a base name like "Submit" matches more broadly,
// which faithfully captures grep's imprecision rather than hiding it.
func grepReadTokens(label string, fileText map[string]string, fileTokens map[string]int) int {
	needle := baseIdent(label)
	total := 0
	for path, text := range fileText {
		if strings.Contains(text, needle) {
			total += fileTokens[path]
		}
	}
	return total
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
	nodes  map[string]model.Node
	adj    map[string][]int // node ID → indices into edges (both directions)
	edges  []model.Edge
	degree map[string]int
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

// topSymbols returns the IDs of the n highest-degree symbol nodes, ties broken
// by ID for reproducibility.
func (g *graph) topSymbols(n int) []string {
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
		// Skip external package/symbol nodes (e.g. fmt, http.Get): they are
		// import targets with sky-high degree, not things a user asks "what
		// does X depend on" about. Benchmarking them models no real query and
		// inflates every column. Real questions target corpus-internal symbols.
		if ext, _ := node.Attrs["external"].(bool); ext {
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
		// Match the MCP neighborhood handler's summary line so the payload is
		// byte-faithful to what the model actually receives.
		Summary: fmt.Sprintf("%d nodes, %d edges within depth %d of %q", len(neighborNodes), len(leanEdges), depth, sn.Label),
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return 0, err
	}
	return estimateTokens(len(b)), nil
}
