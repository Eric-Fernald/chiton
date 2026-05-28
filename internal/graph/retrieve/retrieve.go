// Package retrieve implements chiton's hybrid retrieval policy:
//
//	embed query → vector-seed top-K → graph-expand H hops → rerank → token budget
//
// Vector search supplies semantic recall (the gap a pure graph leaves on cold,
// natural-language queries); graph expansion restores the typed-relationship
// precision that is chiton's reason to exist. The reranker blends the two
// signals plus community membership and provenance into one ordering.
package retrieve

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/embed"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/vector"
)

// Rerank weights. score = wVector·vScore + wProx·proximity + wEdge·edgeConf,
// plus a community bonus, all scaled by a provenance penalty. Tuned so that an
// exact vector hit (vScore≈1) outranks a 1-hop neighbour (proximity 0.5) that
// has no vector score of its own, while still letting strongly-connected
// neighbours surface above weak vector matches.
const (
	wVector        = 0.6
	wProximity     = 0.3
	wEdgeConf      = 0.1
	communityBonus = 0.1
	ambiguousScale = 0.85 // multiplicative penalty for AMBIGUOUS provenance
)

// Engine runs hybrid retrieval against a store + vector index + embedder.
// clusters is optional (nil when no cluster pass has run); it only feeds the
// community bonus.
type Engine struct {
	store    store.GraphStore
	index    vector.Index
	embedder embed.Embedder
	clusters *cluster.Results
}

// New builds an Engine. store, index, and embedder are required.
func New(s store.GraphStore, idx vector.Index, e embed.Embedder, clusters *cluster.Results) *Engine {
	return &Engine{store: s, index: idx, embedder: e, clusters: clusters}
}

// Options tune a single query. Zero values fall back to sensible defaults
// (SeedK 8, Hops 1, MaxNodes 30, MaxTokens 4000).
type Options struct {
	SeedK            int
	Hops             int
	MaxNodes         int
	MaxTokens        int
	EdgeKinds        []model.EdgeKind
	SourcePrefix     string
	IncludeAmbiguous bool
}

func (o Options) withDefaults() Options {
	if o.SeedK <= 0 {
		o.SeedK = 8
	}
	if o.Hops <= 0 {
		o.Hops = 1
	}
	if o.MaxNodes <= 0 {
		o.MaxNodes = 30
	}
	if o.MaxTokens <= 0 {
		o.MaxTokens = 4000
	}
	return o
}

// Scored is a node with its rerank score and graph distance from the nearest
// vector seed (0 == was itself a seed).
type Scored struct {
	Node  model.Node `json:"node"`
	Score float32    `json:"score"`
	Hop   int        `json:"hop"`
}

// Result is the ranked, token-budgeted output of a query.
type Result struct {
	Nodes   []Scored     `json:"nodes"`
	Edges   []model.Edge `json:"edges"`
	Summary string       `json:"summary"`
}

// Query runs the hybrid retrieval pipeline for a natural-language query.
func (e *Engine) Query(ctx context.Context, query string, opts Options) (Result, error) {
	opts = opts.withDefaults()

	if e.index.Len() == 0 {
		return Result{Summary: "no embeddings indexed — run `chiton embed` first"}, nil
	}

	// 1. Embed the query.
	vecs, err := e.embedder.Embed(ctx, []string{query})
	if err != nil {
		return Result{}, fmt.Errorf("embed query: %w", err)
	}
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return Result{Summary: "query produced an empty embedding"}, nil
	}

	// 2. Vector seed.
	seeds := e.index.Search(vecs[0], opts.SeedK)
	if len(seeds) == 0 {
		return Result{Summary: "no semantic matches"}, nil
	}
	vScore := make(map[string]float32, len(seeds))
	for _, s := range seeds {
		vScore[s.ID] = s.Score
	}

	// 3. Graph-expand from each seed, tracking the minimum hop distance and
	//    accumulating the connecting edges.
	hop := map[string]int{}
	nodeByID := map[string]model.Node{}
	edgeConf := map[string]float32{} // best connecting-edge confidence per node
	var edges []model.Edge
	seenEdge := map[string]bool{}

	for _, s := range seeds {
		seedNode, err := e.store.GetNode(ctx, s.ID)
		if err != nil {
			return Result{}, err
		}
		if seedNode == nil {
			continue // embedding outlived its node; skip
		}
		nodeByID[s.ID] = *seedNode
		if _, ok := hop[s.ID]; !ok {
			hop[s.ID] = 0
		}

		sub, err := e.store.Neighbors(ctx, s.ID, opts.Hops, opts.EdgeKinds)
		if err != nil {
			return Result{}, err
		}
		for _, n := range sub.Nodes {
			nodeByID[n.ID] = n
		}
		for _, ed := range sub.Edges {
			if !seenEdge[ed.ID] {
				seenEdge[ed.ID] = true
				edges = append(edges, ed)
			}
			if ed.Confidence > edgeConf[ed.Dst] {
				edgeConf[ed.Dst] = ed.Confidence
			}
			if ed.Confidence > edgeConf[ed.Src] {
				edgeConf[ed.Src] = ed.Confidence
			}
		}
		// BFS hop distances: the store returns the subgraph but not per-node
		// depth, so approximate via edge walking from the seed.
		assignHops(s.ID, sub, hop)
	}

	// 4. Rerank.
	topComm := e.community(seeds[0].ID)
	prefix := strings.ToLower(opts.SourcePrefix)

	scored := make([]Scored, 0, len(nodeByID))
	for id, n := range nodeByID {
		if prefix != "" && !strings.Contains(strings.ToLower(n.SourceURI), prefix) {
			continue
		}
		if n.Provenance == model.ProvAmbiguous && !opts.IncludeAmbiguous {
			continue
		}
		h := hop[id]
		sc := wVector*vScore[id] +
			wProximity*float32(1.0/float64(1+h)) +
			wEdgeConf*edgeConf[id]
		if topComm != nil && n.CommunityID != nil && *n.CommunityID == *topComm {
			sc += communityBonus
		}
		if n.Provenance == model.ProvAmbiguous {
			sc *= ambiguousScale
		}
		scored = append(scored, Scored{Node: n, Score: sc, Hop: h})
	}

	sort.Slice(scored, func(i, j int) bool {
		if scored[i].Score != scored[j].Score {
			return scored[i].Score > scored[j].Score
		}
		return scored[i].Node.ID < scored[j].Node.ID
	})
	if len(scored) > opts.MaxNodes {
		scored = scored[:opts.MaxNodes]
	}

	// 5. Token-budget trim (~4 chars/token, matching graphSummaryHandler).
	scored = budgetTrim(scored, opts.MaxTokens)
	edges = pruneEdges(edges, scored)

	return Result{
		Nodes: scored,
		Edges: edges,
		Summary: fmt.Sprintf("%d nodes, %d edges for %q (seeded by %d vector matches)",
			len(scored), len(edges), query, len(seeds)),
	}, nil
}

// assignHops walks sub's edges outward from seedID, recording the minimum hop
// distance to each reachable node (capped by the subgraph the store returned).
func assignHops(seedID string, sub store.Subgraph, hop map[string]int) {
	adj := map[string][]string{}
	for _, e := range sub.Edges {
		adj[e.Src] = append(adj[e.Src], e.Dst)
		adj[e.Dst] = append(adj[e.Dst], e.Src)
	}
	dist := map[string]int{seedID: 0}
	queue := []string{seedID}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, nb := range adj[cur] {
			if _, ok := dist[nb]; ok {
				continue
			}
			dist[nb] = dist[cur] + 1
			queue = append(queue, nb)
		}
	}
	for id, d := range dist {
		if cur, ok := hop[id]; !ok || d < cur {
			hop[id] = d
		}
	}
}

// community returns the community ID of node id, or nil if unknown.
func (e *Engine) community(id string) *int {
	if e.clusters == nil {
		return nil
	}
	if nr, ok := e.clusters.Nodes[id]; ok {
		c := nr.CommunityID
		return &c
	}
	return nil
}

// budgetTrim keeps the highest-ranked nodes whose cumulative serialized size
// fits within maxTokens (~4 chars per token). Always keeps at least one node.
func budgetTrim(scored []Scored, maxTokens int) []Scored {
	budget := maxTokens * 4
	used := 0
	kept := scored[:0]
	for i, s := range scored {
		b, _ := json.Marshal(s.Node)
		cost := len(b)
		if i > 0 && used+cost > budget {
			break
		}
		used += cost
		kept = append(kept, s)
	}
	return kept
}

// pruneEdges drops edges whose endpoints are not both in the kept node set.
func pruneEdges(edges []model.Edge, scored []Scored) []model.Edge {
	keep := make(map[string]bool, len(scored))
	for _, s := range scored {
		keep[s.Node.ID] = true
	}
	out := edges[:0]
	for _, e := range edges {
		if keep[e.Src] && keep[e.Dst] {
			out = append(out, e)
		}
	}
	return out
}
