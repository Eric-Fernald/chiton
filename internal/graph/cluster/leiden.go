// Package cluster implements Leiden community detection (Traag et al. 2019) on the graph treated as
// undirected and unweighted, mapping each node ID to a compact 0-based community ID.
package cluster

import (
	"math/rand"
	"slices"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

const eps = 1e-10

// Run maps each node ID to a community ID. Higher resolution γ yields smaller communities (1.0 is standard
// modularity). A nil rng uses a fixed seed.
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
// Input edges weigh 1.0, and aggregation sums them.
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

// singletonState puts every node in its own community.
func singletonState(g *graph) *pstate {
	comm := make([]int, g.n)
	for i := range comm {
		comm[i] = i
	}
	return &pstate{comm: comm, commDeg: slices.Clone(g.deg)}
}

// seededState starts g's nodes in the communities comm gives, whose IDs must be below g.n.
func seededState(g *graph, comm []int) *pstate {
	commDeg := make([]float64, g.n)
	for v, c := range comm {
		commDeg[c] += g.deg[v]
	}
	return &pstate{comm: comm, commDeg: commDeg}
}

// commWeights sums edge weight per neighbouring community in reusable buffers, listing communities in
// first-seen order so tie-breaks between equal gains are deterministic.
type commWeights struct {
	w    []float64
	seen []bool
	list []int
}

func newCommWeights(numComm int) *commWeights {
	return &commWeights{w: make([]float64, numComm), seen: make([]bool, numComm)}
}

func (cw *commWeights) add(adj []nbr, comm []int) {
	for _, nb := range adj {
		c := comm[nb.id]
		if !cw.seen[c] {
			cw.seen[c] = true
			cw.list = append(cw.list, c)
		}
		cw.w[c] += nb.weight
	}
}

func (cw *commWeights) clear() {
	for _, c := range cw.list {
		cw.w[c], cw.seen[c] = 0, false
	}
	cw.list = cw.list[:0]
}

// ── local moving ──────────────────────────────────────────────────────────────

// moveNodes moves nodes to the neighbouring community with the best modularity gain for up to 20 sweeps.
// A non-nil coarse keeps each move inside the node's coarse community.
func moveNodes(g *graph, st *pstate, γ float64, rng *rand.Rand, coarse []int) {
	if g.totalW == 0 {
		return
	}
	twoM := 2 * g.totalW
	order := make([]int, g.n)
	nc := newCommWeights(len(st.commDeg))

	for iter := 0; iter < 20; iter++ {
		moved := false
		permutation(order, rng)
		for _, v := range order {
			cOld := st.comm[v]
			kv := g.deg[v]

			nc.clear()
			nc.add(g.adj[v], st.comm)

			// Baseline score for staying in cOld (v excluded from cOld's degree).
			degOld := st.commDeg[cOld] - kv
			base := nc.w[cOld] - γ*kv*degOld/twoM

			bestComm := cOld
			bestGain := 0.0
			for _, c := range nc.list {
				if c == cOld || (coarse != nil && coarse[c] != coarse[v]) {
					continue
				}
				score := nc.w[c] - γ*kv*st.commDeg[c]/twoM
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
}

// ── Leiden refinement ─────────────────────────────────────────────────────────

// refine re-runs local moving from singletons, merging only within each coarse community. This gives Leiden's
// guarantee, which Louvain lacks, that every community is well-connected before aggregation.
func refine(g *graph, coarseComm []int, γ float64, rng *rand.Rand) []int {
	st := singletonState(g)
	// Fine community c starts as the singleton {c}, so coarseComm[c] is its coarse community.
	moveNodes(g, st, γ, rng, coarseComm)
	return st.comm
}

// ── aggregation ───────────────────────────────────────────────────────────────

// aggregate builds a reduced graph whose super-nodes are the communities in comm, weighting each edge by the
// summed inter-community edge weight. It also returns comm compacted to 0-based IDs.
// A super-node's degree is its members' total, so its internal weight counts as a self-loop (twice) and
// totalW is unchanged; the self-loop stays out of adj because it moves with the node and cancels in every gain.
func aggregate(g *graph, comm []int) (*graph, []int) {
	comm = compactComm(comm)
	numComm := 0
	for _, c := range comm {
		numComm = max(numComm, c+1)
	}

	members := make([][]int, numComm)
	aggDeg := make([]float64, numComm)
	for u, c := range comm {
		members[c] = append(members[c], u)
		aggDeg[c] += g.deg[u]
	}

	aggAdj := make([][]nbr, numComm)
	cw := newCommWeights(numComm)
	for c, us := range members {
		cw.clear()
		for _, u := range us {
			cw.add(g.adj[u], comm)
		}
		for _, d := range cw.list {
			if d != c {
				aggAdj[c] = append(aggAdj[c], nbr{d, cw.w[d]})
			}
		}
	}
	return &graph{n: numComm, adj: aggAdj, deg: aggDeg, totalW: g.totalW}, comm
}

// ── main loop ─────────────────────────────────────────────────────────────────

// runLeiden alternates local moving, refinement, and aggregation until refinement stops merging nodes. Each
// aggregate level starts from the unrefined partition, as Leiden requires, not from singletons.
func runLeiden(g0 *graph, γ float64, rng *rand.Rand) []int {
	// nodeToSuper[orig] = super-node index in the current-level graph.
	nodeToSuper := make([]int, g0.n)
	for i := range nodeToSuper {
		nodeToSuper[i] = i
	}

	g := g0
	st := singletonState(g)
	for {
		moveNodes(g, st, γ, rng, nil)
		refined := refine(g, st.comm, γ, rng)
		agg, compactedRefined := aggregate(g, refined)
		if agg.n >= g.n {
			break // refinement merged nothing, so st.comm is final
		}
		for orig, s := range nodeToSuper {
			nodeToSuper[orig] = compactedRefined[s]
		}
		// Each refined community sits inside one unrefined community, which seeds its aggregate node.
		seed := make([]int, agg.n)
		for u, r := range compactedRefined {
			seed[r] = st.comm[u]
		}
		g = agg
		st = seededState(g, compactComm(seed))
	}

	out := make([]int, len(nodeToSuper))
	for orig, s := range nodeToSuper {
		out[orig] = st.comm[s]
	}
	return compactComm(out)
}

// ── helpers ───────────────────────────────────────────────────────────────────

// permutation fills p with a random permutation of 0..len(p)-1.
func permutation(p []int, rng *rand.Rand) {
	for i := range p {
		p[i] = i
	}
	rng.Shuffle(len(p), func(i, j int) { p[i], p[j] = p[j], p[i] })
}

// compactComm remaps community IDs to a contiguous 0-based range, preserving
// relative order of first occurrence.
func compactComm(comm []int) []int {
	maxC := -1
	for _, c := range comm {
		maxC = max(maxC, c)
	}
	remap := make([]int, maxC+1)
	for i := range remap {
		remap[i] = -1
	}
	next := 0
	result := make([]int, len(comm))
	for i, c := range comm {
		if remap[c] < 0 {
			remap[c] = next
			next++
		}
		result[i] = remap[c]
	}
	return result
}
