package metrics

import (
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

// TestBetweennessParallelMatchesSerial checks splitting sources across workers changes no score bit
// (exact ==), in exact (every source) and sampled (scaled subset) mode, and that a worker count is repeatable.
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
				if got[i] != want[i] {
					t.Fatalf("%s, %d workers: node %d = %.17g, one worker %.17g", tc.name, workers, i, got[i], want[i])
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

// TestBetweennessSymmetricTiesExact checks that mathematically tied nodes get bit-equal scores at any worker count.
func TestBetweennessSymmetricTiesExact(t *testing.T) {
	const n = 200
	ring := randomGraph(n, 0, 1)
	star := &intGraph{n: n, adj: make([][]int, n)}
	for i := 1; i < n; i++ {
		star.adj[0] = append(star.adj[0], i)
		star.adj[i] = []int{0}
	}
	all := make([]int, n)
	for i := range all {
		all[i] = i
	}
	for name, g := range map[string]*intGraph{"ring": ring, "star": star} {
		for _, workers := range []int{1, 2, 3, 8} {
			got := betweennessWorkers(g, all, 1, workers)
			for i := 2; i < n; i++ {
				if got[i] != got[1] {
					t.Fatalf("%s, %d workers: node %d = %.17g, node 1 = %.17g", name, workers, i, got[i], got[1])
				}
			}
		}
	}
}

// TestBetweennessDegenerate covers no sources and graphs of at most two nodes.
func TestBetweennessDegenerate(t *testing.T) {
	g := randomGraph(5, 0, 1)
	for _, workers := range []int{1, 4} {
		for i, v := range betweennessWorkers(g, nil, 1, workers) {
			if v != 0 {
				t.Fatalf("no sources, %d workers: node %d = %v", workers, i, v)
			}
		}
	}
	for n := 0; n <= 2; n++ {
		g := &intGraph{n: n, adj: make([][]int, n)}
		if n == 2 {
			g.adj[0], g.adj[1] = []int{1}, []int{0}
		}
		src := make([]int, n)
		for i := range src {
			src[i] = i
		}
		for _, v := range betweennessWorkers(g, src, 1, 4) {
			if v != 0 {
				t.Fatalf("n=%d: score %v, want 0", n, v)
			}
		}
	}
}
