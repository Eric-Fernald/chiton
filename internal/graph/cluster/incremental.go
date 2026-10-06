package cluster

import (
	"maps"
	"math/rand"
	"slices"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/metrics"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// RegionDepth is how many hops around the touched nodes an incremental pass re-clusters.
const RegionDepth = 2

// Region is the part of the graph an incremental pass re-clusters.
type Region struct {
	Nodes   []model.Node    // nodes to re-cluster, in first-reached order
	Edges   []model.Edge    // edges with both ends in Nodes
	Degree  map[string]int  // whole-graph degree of each node whose adjacency was read; others keep prev's
	Removed map[string]bool // seeds no longer in the graph, dropped from the results
}

// NeighborFunc returns id's node (nil when absent), its incident edges, and its distinct neighbours other
// than itself. ingest backs it with GraphStore.Neighbors, so a re-cluster reads only the region.
type NeighborFunc func(id string) (self *model.Node, edges []model.Edge, nbrs []model.Node, err error)

// BuildRegion collects the nodes within depth hops of seeds and the edges among them. Only nodes closer than
// depth are expanded, so edges between two nodes at the hop limit are not read. External package stubs are
// leaves, which keeps a hub like pkg://fmt from pulling in every importer. Absent seeds land in Removed.
func BuildRegion(seeds []string, depth int, nb NeighborFunc) (Region, error) {
	r := Region{Degree: map[string]int{}, Removed: map[string]bool{}}
	in := map[string]bool{}
	var level []string
	for _, id := range seeds {
		if !in[id] {
			in[id] = true
			level = append(level, id)
		}
	}
	seenEdge := map[string]bool{}
	var edges []model.Edge
	for d := 0; len(level) > 0; d++ {
		var next []string
		for _, id := range level {
			self, inc, nbrs, err := nb(id)
			if err != nil {
				return Region{}, err
			}
			if self == nil { // only a seed can be absent; deeper nodes came from a neighbour list
				delete(in, id)
				r.Removed[id] = true
				continue
			}
			if d == 0 {
				r.Nodes = append(r.Nodes, *self)
			}
			if d >= depth || model.IsExternal(*self) {
				continue
			}
			r.Degree[id] = len(nbrs)
			for _, e := range inc {
				if !seenEdge[e.ID] {
					seenEdge[e.ID] = true
					edges = append(edges, e)
				}
			}
			for _, n := range nbrs {
				if in[n.ID] {
					continue
				}
				in[n.ID] = true
				r.Nodes = append(r.Nodes, n)
				if d+1 < depth && !model.IsExternal(n) {
					next = append(next, n.ID)
				}
			}
		}
		level = next
	}
	for _, e := range edges {
		if e.Src != e.Dst && in[e.Src] && in[e.Dst] {
			r.Edges = append(r.Edges, e)
		}
	}
	return r, nil
}

// ApplyRegion re-clusters r and merges it into prev, keeping every other node's prev result. External stubs
// already in prev keep their community, since the region sees only a few of a hub's edges.
func ApplyRegion(prev *Results, r Region, resolution float64, rng *rand.Rand) *Results {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}
	var sub []model.Node
	keep := make(map[string]bool, len(r.Nodes))
	for _, n := range r.Nodes {
		if _, had := prev.Nodes[n.ID]; had && model.IsExternal(n) {
			continue
		}
		sub = append(sub, n)
		keep[n.ID] = true
	}
	_, subEdges := induce(sub, r.Edges, keep)
	local := Run(sub, subEdges, resolution, rng)

	out := make(map[string]NodeResult, len(prev.Nodes)+len(sub))
	base := 0
	for id, pr := range prev.Nodes {
		if r.Removed[id] || keep[id] {
			continue
		}
		out[id] = pr
		base = max(base, pr.CommunityID+1)
	}
	for _, n := range r.Nodes {
		pr, had := prev.Nodes[n.ID]
		nr := NodeResult{Label: n.Label, SourceURI: n.SourceURI, CommunityID: pr.CommunityID,
			Betweenness: pr.Betweenness, Degree: pr.Degree}
		if c, ok := local[n.ID]; ok {
			nr.CommunityID = base + c
		}
		if d, ok := r.Degree[n.ID]; ok || !had {
			nr.Degree = d
		}
		out[n.ID] = nr
	}
	return compactResults(out)
}

// compactResults renumbers the communities in nodes to 0..k-1, keeping their relative order.
func compactResults(nodes map[string]NodeResult) *Results {
	used := map[int]bool{}
	for _, nr := range nodes {
		used[nr.CommunityID] = true
	}
	ids := slices.Sorted(maps.Keys(used))
	remap := make(map[int]int, len(ids))
	for i, c := range ids {
		remap[c] = i
	}
	for id, nr := range nodes {
		nr.CommunityID = remap[nr.CommunityID]
		nodes[id] = nr
	}
	return &Results{GeneratedAt: time.Now(), NodeCount: len(nodes), CommunityCount: len(ids), Nodes: nodes}
}

// RunIncremental is ApplyRegion over an in-memory graph: it re-clusters the RegionDepth-hop region around
// touched plus any node new since prev, and drops prev nodes absent from nodes. A nil or empty prev runs a
// full pass. Betweenness is global, so it comes from prev (new nodes get 0).
func RunIncremental(prev *Results, nodes []model.Node, edges []model.Edge, touched map[string]bool, resolution float64, rng *rand.Rand) *Results {
	if len(nodes) == 0 {
		return &Results{GeneratedAt: time.Now(), Nodes: map[string]NodeResult{}}
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}
	// On a cold start or a lost sidecar, do the same full pass as `chiton cluster`.
	if prev == nil || len(prev.Nodes) == 0 {
		return Full(nodes, edges, resolution, rng)
	}

	var seeds []string
	for _, n := range nodes {
		if _, had := prev.Nodes[n.ID]; touched[n.ID] || !had {
			seeds = append(seeds, n.ID)
		}
	}
	r, _ := BuildRegion(seeds, RegionDepth, memNeighbors(nodes, edges)) // memNeighbors never fails
	present := make(map[string]bool, len(nodes))
	for _, n := range nodes {
		present[n.ID] = true
	}
	for id := range prev.Nodes {
		if !present[id] {
			r.Removed[id] = true
		}
	}
	return ApplyRegion(prev, r, resolution, rng)
}

// memNeighbors is a NeighborFunc over an in-memory graph, ignoring edges to IDs not in nodes.
func memNeighbors(nodes []model.Node, edges []model.Edge) NeighborFunc {
	byID := make(map[string]*model.Node, len(nodes))
	for i := range nodes {
		byID[nodes[i].ID] = &nodes[i]
	}
	incident := map[string][]model.Edge{}
	for _, e := range edges {
		if e.Src == e.Dst || byID[e.Src] == nil || byID[e.Dst] == nil {
			continue
		}
		incident[e.Src] = append(incident[e.Src], e)
		incident[e.Dst] = append(incident[e.Dst], e)
	}
	return func(id string) (*model.Node, []model.Edge, []model.Node, error) {
		self := byID[id]
		if self == nil {
			return nil, nil, nil, nil
		}
		seen := map[string]bool{}
		var nbrs []model.Node
		for _, e := range incident[id] {
			other := e.Dst
			if other == id {
				other = e.Src
			}
			if !seen[other] {
				seen[other] = true
				nbrs = append(nbrs, *byID[other])
			}
		}
		return self, incident[id], nbrs, nil
	}
}

// Full computes Leiden communities, betweenness, and degree for every node in one whole-graph pass. A nil rng
// uses Run's fixed seed.
func Full(nodes []model.Node, edges []model.Edge, resolution float64, rng *rand.Rand) *Results {
	comm := Run(nodes, edges, resolution, rng)
	met := metrics.Compute(nodes, edges)
	bet := make(map[string]float64, len(met))
	degree := make(map[string]int, len(met))
	for id, m := range met {
		bet[id] = m.Betweenness
		degree[id] = m.Degree
	}
	return buildResults(nodes, comm, bet, degree)
}

// induce returns the subgraph induced by the node-ID set keep: nodes whose ID
// is in keep, and edges with both endpoints in keep.
func induce(nodes []model.Node, edges []model.Edge, keep map[string]bool) ([]model.Node, []model.Edge) {
	var sn []model.Node
	for _, n := range nodes {
		if keep[n.ID] {
			sn = append(sn, n)
		}
	}
	var se []model.Edge
	for _, e := range edges {
		if keep[e.Src] && keep[e.Dst] {
			se = append(se, e)
		}
	}
	return sn, se
}

// buildResults assembles a *Results from per-node community, betweenness, and
// degree maps. Missing betweenness/degree default to zero.
func buildResults(nodes []model.Node, comm map[string]int, bet map[string]float64, degree map[string]int) *Results {
	commSet := map[int]struct{}{}
	nodeResults := make(map[string]NodeResult, len(nodes))
	for _, n := range nodes {
		c := comm[n.ID]
		commSet[c] = struct{}{}
		nodeResults[n.ID] = NodeResult{
			Label:       n.Label,
			SourceURI:   n.SourceURI,
			CommunityID: c,
			Betweenness: bet[n.ID],
			Degree:      degree[n.ID],
		}
	}
	return &Results{
		GeneratedAt:    time.Now(),
		NodeCount:      len(nodes),
		CommunityCount: len(commSet),
		Nodes:          nodeResults,
	}
}
