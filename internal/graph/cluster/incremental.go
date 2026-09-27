package cluster

import (
	"math/rand"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/metrics"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// RunIncremental re-clusters only the ≤2-hop neighborhood of touched, keeping other nodes' prev communities;
// prev nil/empty runs a full pass. Betweenness is global, so it is carried over from prev (new nodes get 0).
func RunIncremental(prev *Results, nodes []model.Node, edges []model.Edge, touched map[string]bool, resolution float64, rng *rand.Rand) *Results {
	if len(nodes) == 0 {
		return &Results{GeneratedAt: time.Now(), Nodes: map[string]NodeResult{}}
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}
	// Cold start (or sidecar lost): do the same full pass as `chiton cluster`.
	if prev == nil || len(prev.Nodes) == 0 {
		return Full(nodes, edges, resolution, rng)
	}

	adj := buildAdjacency(nodes, edges)
	degree := make(map[string]int, len(nodes))
	for _, n := range nodes {
		degree[n.ID] = len(adj[n.ID])
	}

	// 2-hop neighborhood of the touched nodes (present nodes only).
	hood := neighborhood(adj, touched, 2)

	// Induced subgraph over the neighborhood, clustered fresh.
	subNodes, subEdges := induce(nodes, edges, hood)
	localComm := Run(subNodes, subEdges, resolution, rng)

	// Retained communities for nodes outside the neighborhood.
	comm := make(map[string]int, len(nodes))
	maxRetained := -1
	for _, n := range nodes {
		if hood[n.ID] {
			continue
		}
		pr, ok := prev.Nodes[n.ID]
		if !ok {
			continue // new node outside the hood (rare) — assigned a singleton below
		}
		comm[n.ID] = pr.CommunityID
		maxRetained = max(maxRetained, pr.CommunityID)
	}

	base := maxRetained + 1
	maxLocal := -1
	for _, c := range localComm {
		maxLocal = max(maxLocal, c)
	}
	for id, c := range localComm {
		comm[id] = base + c
	}
	// Any present node still unassigned (e.g. a brand-new isolated node not in
	// the neighborhood) becomes its own singleton community.
	next := base + maxLocal + 1
	for _, n := range nodes {
		if _, ok := comm[n.ID]; !ok {
			comm[n.ID] = next
			next++
		}
	}

	comm = compactByNodeOrder(nodes, comm)

	// Carry betweenness forward for surviving nodes; new nodes default to 0.
	bet := make(map[string]float64, len(nodes))
	for _, n := range nodes {
		if pr, ok := prev.Nodes[n.ID]; ok {
			bet[n.ID] = pr.Betweenness
		}
	}

	return buildResults(nodes, comm, bet, degree)
}

// Full is a whole-graph pass: Leiden communities plus betweenness and degree for every node. A nil rng
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

// buildAdjacency builds an undirected, deduplicated adjacency map restricted to
// node IDs present in nodes. Self-loops are dropped.
func buildAdjacency(nodes []model.Node, edges []model.Edge) map[string][]string {
	present := make(map[string]struct{}, len(nodes))
	for _, n := range nodes {
		present[n.ID] = struct{}{}
	}
	adj := make(map[string]map[string]struct{}, len(nodes))
	add := func(a, b string) {
		if adj[a] == nil {
			adj[a] = map[string]struct{}{}
		}
		adj[a][b] = struct{}{}
	}
	for _, e := range edges {
		if e.Src == e.Dst {
			continue
		}
		if _, ok := present[e.Src]; !ok {
			continue
		}
		if _, ok := present[e.Dst]; !ok {
			continue
		}
		add(e.Src, e.Dst)
		add(e.Dst, e.Src)
	}
	out := make(map[string][]string, len(adj))
	for id, set := range adj {
		nb := make([]string, 0, len(set))
		for n := range set {
			nb = append(nb, n)
		}
		out[id] = nb
	}
	return out
}

// neighborhood returns the IDs within depth hops of the seeds. Every seed is included, even one with no
// adjacency entry (isolated or deleted); induce later drops IDs that are not graph nodes.
func neighborhood(adj map[string][]string, seeds map[string]bool, depth int) map[string]bool {
	hood := make(map[string]bool)
	var frontier []string
	for id, ok := range seeds {
		if !ok {
			continue
		}
		// Include the seed itself; even an isolated node (no adj entry) belongs
		// to its own re-cluster set so its community is refreshed.
		if !hood[id] {
			hood[id] = true
			frontier = append(frontier, id)
		}
	}
	for d := 0; d < depth && len(frontier) > 0; d++ {
		var next []string
		for _, id := range frontier {
			for _, nb := range adj[id] {
				if !hood[nb] {
					hood[nb] = true
					next = append(next, nb)
				}
			}
		}
		frontier = next
	}
	return hood
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

// compactByNodeOrder remaps community IDs to a contiguous 0-based range,
// assigning new IDs in order of first appearance across nodes (deterministic).
func compactByNodeOrder(nodes []model.Node, comm map[string]int) map[string]int {
	remap := make(map[int]int)
	nextID := 0
	out := make(map[string]int, len(comm))
	for _, n := range nodes {
		c, ok := comm[n.ID]
		if !ok {
			continue
		}
		mapped, seen := remap[c]
		if !seen {
			mapped = nextID
			remap[c] = mapped
			nextID++
		}
		out[n.ID] = mapped
	}
	return out
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
