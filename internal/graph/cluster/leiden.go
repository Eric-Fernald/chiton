// Package cluster implements the Leiden community detection algorithm.
//
// Input is a slice of model.Node and model.Edge. Output is a map from node ID
// to integer community ID (0-based, compact). The algorithm treats the graph
// as undirected and unweighted (edge weight = 1).
//
// Reference: Traag, V.A., Waltman, L. & van Eck, N.J. (2019).
// "From Louvain to Leiden: guaranteeing well-connected communities."
// Scientific Reports 9, 5233. https://doi.org/10.1038/s41598-019-41695-z
package cluster

import (
	"math/rand"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

const eps = 1e-10

// Run executes the Leiden algorithm and returns a map from node ID to community
// ID. Resolution γ controls granularity: higher = smaller communities.
// γ=1.0 gives standard modularity-equivalent behaviour. Pass nil rng to use a
// fixed seed.
func Run(nodes []model.Node, edges []model.Edge, resolution float64, rng *rand.Rand) map[string]int {
	if len(nodes) == 0 {
		return nil
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	idToIdx := make(map[string]int, len(nodes))
	idxToID := make([]string, len(nodes))
	for i, n := range nodes {
		idToIdx[n.ID] = i
		idxToID[i] = n.ID
	}

	g := buildGraph(len(nodes), edges, idToIdx)
	comm := runLeiden(g, resolution, rng)

	result := make(map[string]int, len(nodes))
	for i, c := range comm {
		result[idxToID[i]] = c
	}
	return result
}

// ── internal graph ────────────────────────────────────────────────────────────

// graph is an undirected weighted graph with 0-based integer node indices.
// All edge weights are 1.0 (unweighted graphs).
type graph struct {
	n      int
	adj    [][]nbr
	deg    []float64 // deg[i] = sum of weights of edges incident to i
	totalW float64   // sum of all edge weights (each edge counted once)
}

type nbr struct {
	id     int
	weight float64
}

func buildGraph(n int, edges []model.Edge, idToIdx map[string]int) *graph {
	adj := make([][]nbr, n)
	deg := make([]float64, n)
	var totalW float64

	seen := make(map[[2]int]bool)
	for _, e := range edges {
		u, ok1 := idToIdx[e.Src]
		v, ok2 := idToIdx[e.Dst]
		if !ok1 || !ok2 || u == v {
			continue
		}
		// Deduplicate: store canonical (min, max) pair.
		key := [2]int{u, v}
		if u > v {
			key = [2]int{v, u}
		}
		if seen[key] {
			continue
		}
		seen[key] = true
		adj[u] = append(adj[u], nbr{v, 1.0})
		adj[v] = append(adj[v], nbr{u, 1.0})
		deg[u]++
		deg[v]++
		totalW++
	}
	return &graph{n: n, adj: adj, deg: deg, totalW: totalW}
}

// ── partition state ───────────────────────────────────────────────────────────

// pstate tracks per-community degree sums for efficient modularity gain computation.
type pstate struct {
	comm    []int
	commDeg []float64 // commDeg[c] = sum of node degrees in community c
}

func newState(g *graph, comm []int) *pstate {
	maxC := 0
	for _, c := range comm {
		if c > maxC {
			maxC = c
		}
	}
	st := &pstate{
		comm:    append([]int(nil), comm...),
		commDeg: make([]float64, maxC+1),
	}
	for i, c := range comm {
		st.commDeg[c] += g.deg[i]
	}
	return st
}

// ── local moving ──────────────────────────────────────────────────────────────

// localMoving runs greedy modularity-improving moves until stable.
// Returns whether any node was moved.
func localMoving(g *graph, st *pstate, γ float64, rng *rand.Rand) bool {
	if g.totalW == 0 {
		return false
	}
	twoM := 2 * g.totalW
	improved := false

	for iter := 0; iter < 20; iter++ {
		moved := false
		for _, v := range permutation(g.n, rng) {
			cOld := st.comm[v]
			kv := g.deg[v]

			nc := neighborComms(g.adj[v], st.comm)

			// Baseline score for staying in cOld (v excluded from cOld's degree).
			degOld := st.commDeg[cOld] - kv
			kvOld := nc[cOld]
			base := kvOld - γ*kv*degOld/twoM

			bestComm := cOld
			bestGain := 0.0
			for c, kvc := range nc {
				if c == cOld {
					continue
				}
				score := kvc - γ*kv*st.commDeg[c]/twoM
				if gain := score - base; gain > bestGain+eps {
					bestGain = gain
					bestComm = c
				}
			}
			if bestComm == cOld {
				continue
			}
			st.commDeg[cOld] -= kv
			st.commDeg[bestComm] += kv
			st.comm[v] = bestComm
			moved = true
			improved = true
		}
		if !moved {
			break
		}
	}
	return improved
}

// neighborComms returns the sum of edge weights from v to each adjacent community.
func neighborComms(adj []nbr, comm []int) map[int]float64 {
	nc := make(map[int]float64, len(adj))
	for _, nb := range adj {
		nc[comm[nb.id]] += nb.weight
	}
	return nc
}

// ── Leiden refinement ─────────────────────────────────────────────────────────

// refine runs constrained local moving within each coarse community, producing
// a finer partition. This is the step that distinguishes Leiden from Louvain:
// it guarantees that each final community was well-connected before aggregation.
func refine(g *graph, coarseComm []int, γ float64, rng *rand.Rand) []int {
	// Start with every node in its own singleton fine community.
	fineComm := make([]int, g.n)
	for i := range fineComm {
		fineComm[i] = i
	}
	st := newState(g, fineComm)
	// coarseOf maps a fine community ID to its coarse community.
	// Initially fine community i contains only node i, so coarseOf[i] = coarseComm[i].
	coarseOf := append([]int(nil), coarseComm...)

	if g.totalW == 0 {
		return fineComm
	}
	twoM := 2 * g.totalW

	for iter := 0; iter < 20; iter++ {
		moved := false
		for _, v := range permutation(g.n, rng) {
			cOld := st.comm[v]
			kv := g.deg[v]
			cv := coarseComm[v]

			nc := neighborComms(g.adj[v], st.comm)

			degOld := st.commDeg[cOld] - kv
			kvOld := nc[cOld]
			base := kvOld - γ*kv*degOld/twoM

			bestComm := cOld
			bestGain := 0.0
			for c, kvc := range nc {
				if c == cOld {
					continue
				}
				if coarseOf[c] != cv {
					continue // only merge within same coarse community
				}
				score := kvc - γ*kv*st.commDeg[c]/twoM
				if gain := score - base; gain > bestGain+eps {
					bestGain = gain
					bestComm = c
				}
			}
			if bestComm == cOld {
				continue
			}
			st.commDeg[cOld] -= kv
			st.commDeg[bestComm] += kv
			st.comm[v] = bestComm
			moved = true
		}
		if !moved {
			break
		}
	}
	return st.comm
}

// ── aggregation ───────────────────────────────────────────────────────────────

// aggregate builds a reduced graph where each community in comm is a super-node.
// Edge weights in the aggregate are sums of inter-community edge weights.
// The comm slice is compacted in-place to 0-based IDs before aggregation.
func aggregate(g *graph, comm []int) (*graph, []int) {
	comm = compactComm(comm)
	numComm := 0
	for _, c := range comm {
		if c+1 > numComm {
			numComm = c + 1
		}
	}

	type pair struct{ a, b int }
	edgeW := make(map[pair]float64)
	for u := 0; u < g.n; u++ {
		cu := comm[u]
		for _, nb := range g.adj[u] {
			cv := comm[nb.id]
			if cu == cv {
				continue
			}
			p := pair{cu, cv}
			if cu > cv {
				p = pair{cv, cu}
			}
			// Each undirected edge (u,v) appears in adj[u] and adj[v], so we
			// accumulate half the true weight per direction; dividing by 2 below
			// gives the correct inter-community edge weight.
			edgeW[p] += nb.weight
		}
	}

	aggAdj := make([][]nbr, numComm)
	aggDeg := make([]float64, numComm)
	var totalW float64
	for p, w := range edgeW {
		w /= 2
		aggAdj[p.a] = append(aggAdj[p.a], nbr{p.b, w})
		aggAdj[p.b] = append(aggAdj[p.b], nbr{p.a, w})
		aggDeg[p.a] += w
		aggDeg[p.b] += w
		totalW += w
	}
	return &graph{n: numComm, adj: aggAdj, deg: aggDeg, totalW: totalW}, comm
}

// ── main loop ─────────────────────────────────────────────────────────────────

func runLeiden(g0 *graph, γ float64, rng *rand.Rand) []int {
	n0 := g0.n

	// nodeToSuper[orig] = super-node index in the current-level graph.
	nodeToSuper := make([]int, n0)
	for i := range nodeToSuper {
		nodeToSuper[i] = i
	}

	g := g0
	for {
		comm := make([]int, g.n)
		for i := range comm {
			comm[i] = i
		}
		st := newState(g, comm)

		if !localMoving(g, st, γ, rng) {
			break
		}

		refined := refine(g, st.comm, γ, rng)
		agg, compactedRefined := aggregate(g, refined)

		// Update nodeToSuper: each original node's super-node is now the fine
		// community it belongs to, which becomes the aggregate node index.
		for orig := 0; orig < n0; orig++ {
			nodeToSuper[orig] = compactedRefined[nodeToSuper[orig]]
		}

		if agg.n >= g.n {
			break
		}
		g = agg
	}

	return compactComm(nodeToSuper)
}

// ── helpers ───────────────────────────────────────────────────────────────────

func permutation(n int, rng *rand.Rand) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = i
	}
	rng.Shuffle(n, func(i, j int) { p[i], p[j] = p[j], p[i] })
	return p
}

// compactComm remaps community IDs to a contiguous 0-based range, preserving
// relative order of first occurrence.
func compactComm(comm []int) []int {
	remap := make(map[int]int, len(comm))
	next := 0
	result := make([]int, len(comm))
	for i, c := range comm {
		if _, ok := remap[c]; !ok {
			remap[c] = next
			next++
		}
		result[i] = remap[c]
	}
	return result
}
