// Package bench implements chiton's token-cost benchmark: each dependency question is costed (chars ÷ 4)
// as naive (read every file), grep+read (files naming the symbol), and chiton's lean MCP neighborhood.
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

// leanNode/leanEdge mirror internal/mcp's lean shapes field-for-field (Provenance and Confidence are
// never omitted) so the chiton column isn't flattered by a smaller payload.
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

// Run builds an in-memory graph over the supported files under path and costs the depth-2
// neighborhoods of its maxQueries highest-degree symbols; name labels the report.
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

// grepReadTokens models "grep -l <symbol> | xargs cat" on the base identifier (after the last '.'),
// since qualified labels like "LinkService.Submit" never appear verbatim in source.
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
		// Skip external import targets (fmt, http.Get): their huge degree models no real question
		// and would inflate every column.
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
		// Same summary as the MCP neighborhood handler. MCP indents its JSON but this compact
		// marshal doesn't, so the chiton column slightly understates the real payload.
		Summary: fmt.Sprintf("%d nodes, %d edges within depth %d of %q", len(neighborNodes), len(leanEdges), depth, sn.Label),
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return 0, err
	}
	return estimateTokens(len(b)), nil
}
