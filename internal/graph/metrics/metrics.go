// Package metrics computes graph centrality metrics for chiton nodes.
//
// Two metrics are computed:
//   - Degree: number of incident edges (undirected).
//   - Betweenness: fraction of shortest paths that pass through the node,
//     normalized to [0, 1]. Uses the Brandes (2001) algorithm. For graphs
//     larger than samplingThreshold nodes, betweenness is approximated by
//     running BFS from a random sample of pivot nodes and scaling the result.
//
// Reference: Brandes, U. (2001). A faster algorithm for betweenness centrality.
// Journal of Mathematical Sociology 25(2), 163–177.
package metrics

import (
	"math/rand"

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

	var raw []float64
	if g.n <= samplingThreshold {
		raw = brandesExact(g)
	} else {
		rng := rand.New(rand.NewSource(42))
		raw = brandesSampled(g, samplePivots, rng)
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

// brandesExact computes exact normalized betweenness for all nodes via the
// Brandes algorithm. Complexity O(n·m).
func brandesExact(g *intGraph) []float64 {
	bet := make([]float64, g.n)
	for s := 0; s < g.n; s++ {
		accumulate(g, s, bet, 1.0)
	}
	normalize(bet, g.n)
	return bet
}

// brandesSampled approximates betweenness by running Brandes from k random
// pivot sources and scaling by n/k.
func brandesSampled(g *intGraph, k int, rng *rand.Rand) []float64 {
	if k > g.n {
		k = g.n
	}
	sources := rng.Perm(g.n)[:k]
	scale := float64(g.n) / float64(k)

	bet := make([]float64, g.n)
	for _, s := range sources {
		accumulate(g, s, bet, scale)
	}
	normalize(bet, g.n)
	return bet
}

// accumulate runs BFS from source s and back-propagates dependency scores
// into bet, multiplied by scale. This is the inner loop of Brandes' algorithm.
func accumulate(g *intGraph, s int, bet []float64, scale float64) {
	n := g.n
	pred := make([][]int, n)
	sigma := make([]float64, n)
	dist := make([]int, n)
	for i := range dist {
		dist[i] = -1
	}
	sigma[s] = 1.0
	dist[s] = 0

	queue := make([]int, 0, n)
	queue = append(queue, s)
	stack := make([]int, 0, n)

	for len(queue) > 0 {
		v := queue[0]
		queue = queue[1:]
		stack = append(stack, v)
		for _, w := range g.adj[v] {
			if dist[w] < 0 {
				dist[w] = dist[v] + 1
				queue = append(queue, w)
			}
			if dist[w] == dist[v]+1 {
				sigma[w] += sigma[v]
				pred[w] = append(pred[w], v)
			}
		}
	}

	delta := make([]float64, n)
	for len(stack) > 0 {
		w := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, v := range pred[w] {
			if sigma[w] > 0 {
				delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
			}
		}
		if w != s {
			bet[w] += delta[w] * scale
		}
	}
}

// normalize divides raw betweenness sums by the number of pairs (n-1)(n-2)/2
// for undirected graphs (Brandes counts each path in both directions).
func normalize(bet []float64, n int) {
	if n <= 2 {
		return
	}
	// Brandes accumulates each undirected shortest path twice (once from each
	// endpoint), so the normalization denominator is (n-1)*(n-2), not /2.
	norm := float64((n - 1) * (n - 2))
	for i := range bet {
		bet[i] /= norm
	}
}
