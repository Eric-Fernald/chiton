// Package metrics computes undirected degree and [0,1]-normalized Brandes betweenness for chiton nodes.
// Above samplingThreshold nodes, it estimates betweenness from a seeded sample of pivot sources.
package metrics

import (
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

const (
	samplingThreshold = 5_000
	samplePivots      = 500
)

// NodeMetrics holds computed centrality metrics for a single node.
type NodeMetrics struct {
	Degree      int
	Betweenness float64 // normalized to [0, 1]
}

// Compute returns metrics for every node in the given slice. Unknown node IDs
// in the edge list are silently ignored.
func Compute(nodes []model.Node, edges []model.Edge) map[string]NodeMetrics {
	if len(nodes) == 0 {
		return nil
	}

	idToIdx := make(map[string]int, len(nodes))
	idxToID := make([]string, len(nodes))
	for i, n := range nodes {
		idToIdx[n.ID] = i
		idxToID[i] = n.ID
	}

	g := buildIntGraph(len(nodes), edges, idToIdx)

	// Exact mode makes every node a source. Sampling uses k seeded pivots, scaling each contribution by n/k.
	var raw []float64
	if g.n <= samplingThreshold {
		sources := make([]int, g.n)
		for i := range sources {
			sources[i] = i
		}
		raw = betweenness(g, sources, 1.0)
	} else {
		k := min(samplePivots, g.n)
		sources := rand.New(rand.NewSource(42)).Perm(g.n)[:k]
		raw = betweenness(g, sources, float64(g.n)/float64(k))
	}

	result := make(map[string]NodeMetrics, len(nodes))
	for i, id := range idxToID {
		result[id] = NodeMetrics{
			Degree:      len(g.adj[i]),
			Betweenness: raw[i],
		}
	}
	return result
}

// ── internal graph ────────────────────────────────────────────────────────────

type intGraph struct {
	n   int
	adj [][]int // undirected, deduplicated
}

func buildIntGraph(n int, edges []model.Edge, idToIdx map[string]int) *intGraph {
	adj := make([][]int, n)
	seen := make(map[[2]int]bool)
	for _, e := range edges {
		u, ok1 := idToIdx[e.Src]
		v, ok2 := idToIdx[e.Dst]
		if !ok1 || !ok2 || u == v {
			continue
		}
		key := [2]int{u, v}
		if u > v {
			key = [2]int{v, u}
		}
		if seen[key] {
			continue
		}
		seen[key] = true
		adj[u] = append(adj[u], v)
		adj[v] = append(adj[v], u)
	}
	return &intGraph{n: n, adj: adj}
}

// ── Brandes algorithm ─────────────────────────────────────────────────────────

const (
	// maxWorkers caps betweenness's goroutines, since each holds five n-length buffers.
	maxWorkers = 8
	// chunkSources is the fixed number of sources summed into one partial vector. Because the grouping never
	// depends on the worker count, scores are bit-identical on every machine.
	chunkSources = 32
)

// betweenness runs Brandes from each source on up to maxWorkers goroutines, scaling every contribution by
// scale, and normalizes the sums. It runs in O(len(sources)·m) total work.
func betweenness(g *intGraph, sources []int, scale float64) []float64 {
	return betweennessWorkers(g, sources, scale, min(runtime.GOMAXPROCS(0), maxWorkers))
}

// betweennessWorkers splits sources into fixed chunks of chunkSources, has workers claim chunks and sum each
// into its own partial vector, then adds the partials in chunk order after the join. The float summation
// order depends only on the sources, never on workers or scheduling. Partials number at most
// ceil(len(sources)/chunkSources): 16 for the 500 sampled pivots, and at most 157 for exact mode (n <= 5000).
func betweennessWorkers(g *intGraph, sources []int, scale float64, workers int) []float64 {
	nChunks := (len(sources) + chunkSources - 1) / chunkSources
	workers = max(1, min(workers, nChunks))
	partial := make([][]float64, nChunks)
	var next atomic.Int64
	var wg sync.WaitGroup
	for range workers {
		wg.Go(func() {
			b := newBrandes(g)
			for {
				c := int(next.Add(1)) - 1
				if c >= nChunks {
					return
				}
				p := make([]float64, g.n)
				for _, s := range sources[c*chunkSources : min((c+1)*chunkSources, len(sources))] {
					b.accumulate(s, p, scale)
				}
				partial[c] = p
			}
		})
	}
	wg.Wait()
	bet := make([]float64, g.n)
	for _, p := range partial {
		for i, v := range p {
			bet[i] += v
		}
	}
	normalize(bet, g.n)
	return bet
}

// brandes holds scratch buffers reused across sources. Only entries a BFS reached are reset afterwards,
// so each source costs O(reached nodes + edges) rather than O(n).
type brandes struct {
	g     *intGraph
	sigma []float64
	dist  []int
	delta []float64
	order []int // BFS visit order, used as the queue and, reversed, as the stack
}

func newBrandes(g *intGraph) *brandes {
	dist := make([]int, g.n)
	for i := range dist {
		dist[i] = -1
	}
	return &brandes{g: g, sigma: make([]float64, g.n), dist: dist, delta: make([]float64, g.n), order: make([]int, 0, g.n)}
}

// accumulate runs BFS from source s and back-propagates dependency scores
// into bet, multiplied by scale. This is the inner loop of Brandes' algorithm.
func (b *brandes) accumulate(s int, bet []float64, scale float64) {
	adj, sigma, dist, delta := b.g.adj, b.sigma, b.dist, b.delta
	sigma[s] = 1.0
	dist[s] = 0

	order := append(b.order[:0], s)
	for head := 0; head < len(order); head++ {
		v := order[head]
		for _, w := range adj[v] {
			if dist[w] < 0 {
				dist[w] = dist[v] + 1
				order = append(order, w)
			}
			if dist[w] == dist[v]+1 {
				sigma[w] += sigma[v]
			}
		}
	}

	// Predecessors of w are its neighbours one level closer to s (adjacency is symmetric and deduplicated).
	for i := len(order) - 1; i >= 0; i-- {
		w := order[i]
		for _, v := range adj[w] {
			if dist[v] == dist[w]-1 {
				delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
			}
		}
		if w != s {
			bet[w] += delta[w] * scale
		}
	}

	for _, v := range order {
		sigma[v], dist[v], delta[v] = 0, -1, 0
	}
	b.order = order
}

// normalize divides raw sums by (n-1)(n-2), twice the undirected pair count, because Brandes
// accumulates each shortest path once from each endpoint.
func normalize(bet []float64, n int) {
	if n <= 2 {
		return
	}
	norm := float64((n - 1) * (n - 2))
	for i := range bet {
		bet[i] /= norm
		// Sampling extrapolates by n/k, which can push a node past 1, so clamp to the documented [0,1] range.
		if bet[i] > 1 {
			bet[i] = 1
		}
	}
}
