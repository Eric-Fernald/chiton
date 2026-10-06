package metrics

import (
	"math"
	"math/rand"
	"testing"
)

// randomGraph returns a seeded n-node graph: a ring (so it is connected) plus extra random chords.
func randomGraph(n, chords int, seed int64) *intGraph {
	rng := rand.New(rand.NewSource(seed))
	adj := make([][]int, n)
	seen := map[[2]int]bool{}
	link := func(u, v int) {
		if u == v {
			return
		}
		key := [2]int{min(u, v), max(u, v)}
		if seen[key] {
			return
		}
		seen[key] = true
		adj[u] = append(adj[u], v)
		adj[v] = append(adj[v], u)
	}
	for i := range n {
		link(i, (i+1)%n)
	}
	for range chords {
		link(rng.Intn(n), rng.Intn(n))
	}
	return &intGraph{n: n, adj: adj}
}

// TestBetweennessParallelMatchesSerial checks splitting sources across workers changes no score beyond float
// rounding, in exact (every source) and sampled (scaled subset) mode, and that a worker count is repeatable.
func TestBetweennessParallelMatchesSerial(t *testing.T) {
	g := randomGraph(600, 1500, 7)
	all := make([]int, g.n)
	for i := range all {
		all[i] = i
	}
	sampled := rand.New(rand.NewSource(42)).Perm(g.n)[:50]
	cases := []struct {
		name    string
		sources []int
		scale   float64
	}{
		{"exact", all, 1},
		{"sampled", sampled, float64(g.n) / 50},
	}
	for _, tc := range cases {
		want := betweennessWorkers(g, tc.sources, tc.scale, 1)
		for _, workers := range []int{2, 3, 8, 64} {
			got := betweennessWorkers(g, tc.sources, tc.scale, workers)
			for i := range want {
				if math.Abs(got[i]-want[i]) > 1e-9 {
					t.Fatalf("%s, %d workers: node %d = %.15f, serial %.15f", tc.name, workers, i, got[i], want[i])
				}
			}
			again := betweennessWorkers(g, tc.sources, tc.scale, workers)
			for i := range got {
				if again[i] != got[i] {
					t.Fatalf("%s, %d workers: node %d differs between runs (%v vs %v)", tc.name, workers, i, got[i], again[i])
				}
			}
		}
	}
}
