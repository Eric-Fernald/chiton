package cluster

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// karateEdges is Zachary's karate club (34 members, 78 ties), 1-based as published.
var karateEdges = [][2]int{
	{1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}, {1, 9}, {1, 11}, {1, 12}, {1, 13}, {1, 14},
	{1, 18}, {1, 20}, {1, 22}, {1, 32}, {2, 3}, {2, 4}, {2, 8}, {2, 14}, {2, 18}, {2, 20}, {2, 22}, {2, 31},
	{3, 4}, {3, 8}, {3, 9}, {3, 10}, {3, 14}, {3, 28}, {3, 29}, {3, 33}, {4, 8}, {4, 13}, {4, 14},
	{5, 7}, {5, 11}, {6, 7}, {6, 11}, {6, 17}, {7, 17}, {9, 31}, {9, 33}, {9, 34}, {10, 34}, {14, 34},
	{15, 33}, {15, 34}, {16, 33}, {16, 34}, {19, 33}, {19, 34}, {20, 34}, {21, 33}, {21, 34}, {23, 33},
	{23, 34}, {24, 26}, {24, 28}, {24, 30}, {24, 33}, {24, 34}, {25, 26}, {25, 28}, {25, 32}, {26, 32},
	{27, 30}, {27, 34}, {28, 34}, {29, 32}, {29, 34}, {30, 33}, {30, 34}, {31, 33}, {31, 34}, {32, 33},
	{32, 34}, {33, 34},
}

// graphOf builds nodes "n0".."n<n-1>" and one edge per pair.
func graphOf(n int, pairs [][2]int) ([]model.Node, []model.Edge) {
	nodes := make([]model.Node, n)
	for i := range nodes {
		id := fmt.Sprint("n", i)
		nodes[i] = makeNode(id, id)
	}
	edges := make([]model.Edge, len(pairs))
	for i, p := range pairs {
		edges[i] = makeEdge(fmt.Sprint("n", p[0]), fmt.Sprint("n", p[1]))
	}
	return nodes, edges
}

// cliques returns k cliques of size nodes each; ring adds one edge from each clique to the next.
func cliques(k, size int, ring bool) [][2]int {
	var pairs [][2]int
	for c := range k {
		base := c * size
		for i := range size {
			for j := i + 1; j < size; j++ {
				pairs = append(pairs, [2]int{base + i, base + j})
			}
		}
		if ring {
			pairs = append(pairs, [2]int{base, ((c + 1) % k) * size})
		}
	}
	return pairs
}

// modularity computes Newman's Q (γ = 1) of comm over the undirected, deduplicated graph.
func modularity(nodes []model.Node, edges []model.Edge, comm map[string]int) float64 {
	idx := make(map[string]int, len(nodes))
	for i, n := range nodes {
		idx[n.ID] = i
	}
	g := buildGraph(len(nodes), edges, idx)
	twoM := 2 * g.totalW
	var internal float64
	tot := map[int]float64{}
	for u := range g.n {
		cu := comm[nodes[u].ID]
		tot[cu] += g.deg[u]
		for _, nb := range g.adj[u] {
			if comm[nodes[nb.id].ID] == cu {
				internal += nb.weight
			}
		}
	}
	q := internal / twoM
	for _, t := range tot {
		q -= (t / twoM) * (t / twoM)
	}
	return q
}

// assertBlocks checks each consecutive run of size nodes shares one community and no two runs share one.
func assertBlocks(t *testing.T, comm map[string]int, k, size int) {
	t.Helper()
	owner := map[int]int{}
	for c := range k {
		want := comm[fmt.Sprint("n", c*size)]
		for i := range size {
			if got := comm[fmt.Sprint("n", c*size+i)]; got != want {
				t.Fatalf("clique %d split: n%d in %d, n%d in %d", c, c*size, want, c*size+i, got)
			}
		}
		if prev, dup := owner[want]; dup {
			t.Fatalf("cliques %d and %d merged into community %d", prev, c, want)
		}
		owner[want] = c
	}
}

// TestTwoCliquesJoinedByOneEdge: a single bridge must not merge two dense cliques.
func TestTwoCliquesJoinedByOneEdge(t *testing.T) {
	pairs := append(cliques(2, 5, false), [2]int{0, 5})
	nodes, edges := graphOf(10, pairs)
	for seed := range int64(10) {
		assertBlocks(t, Run(nodes, edges, 1.0, rand.New(rand.NewSource(seed))), 2, 5)
	}
}

// TestRingOfCliques: ten K5s in a ring (below modularity's resolution limit) give one community per clique.
func TestRingOfCliques(t *testing.T) {
	nodes, edges := graphOf(50, cliques(10, 5, true))
	for seed := range int64(10) {
		assertBlocks(t, Run(nodes, edges, 1.0, rand.New(rand.NewSource(seed))), 10, 5)
	}
}

// TestKarateClubModularity: published Louvain/Leiden partitions of the karate club reach Q ≈ 0.42.
func TestKarateClubModularity(t *testing.T) {
	pairs := make([][2]int, len(karateEdges))
	for i, e := range karateEdges {
		pairs[i] = [2]int{e[0] - 1, e[1] - 1}
	}
	nodes, edges := graphOf(34, pairs)
	for seed := range int64(20) {
		if q := modularity(nodes, edges, Run(nodes, edges, 1.0, rand.New(rand.NewSource(seed)))); q < 0.37 {
			t.Errorf("seed %d: modularity %.4f, want >= 0.37", seed, q)
		}
	}
}
