package hnswlib

import (
	"cmp"
	"fmt"
	"math/rand"
	"reflect"
	"slices"
	"sort"
	"testing"
)

func newTestGraph() *Graph[int] {
	g := New[int]()
	g.M = 6
	g.Metric = Euclidean
	return g
}

func lineGraph(n int) *Graph[int] {
	g := newTestGraph()
	for i := range n {
		g.Insert(i, []float32{float32(i)})
	}
	return g
}

func keys[K cmp.Ordered](hits []Hit[K]) []K {
	out := make([]K, len(hits))
	for i, h := range hits {
		out[i] = h.Key
	}
	return out
}

func randVec(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

// checkInvariants verifies the adjacency lists: ids in range, no self-links, link caps, links only to
// nodes present at that level, and a live entry point on the top level.
func checkInvariants[K cmp.Ordered](t *testing.T, g *Graph[K]) {
	t.Helper()
	for id, levels := range g.links {
		for l, list := range levels {
			if len(list) > g.maxLinks(l) {
				t.Fatalf("node %d level %d has %d links, cap %d", id, l, len(list), g.maxLinks(l))
			}
			for _, nb := range list {
				switch {
				case nb < 0 || int(nb) >= len(g.keys):
					t.Fatalf("node %d level %d links to out-of-range %d", id, l, nb)
				case int(nb) == id:
					t.Fatalf("node %d links to itself at level %d", id, l)
				case len(g.links[nb]) <= l:
					t.Fatalf("node %d level %d links to %d, which has no level %d", id, l, nb, l)
				}
			}
		}
	}
	if len(g.index) > 0 && (g.dead[g.entry] || len(g.links[g.entry])-1 != g.top) {
		t.Fatalf("entry %d is dead or not on the top level %d", g.entry, g.top)
	}
}

func TestNearestOrdersByDistanceThenKey(t *testing.T) {
	g := lineGraph(128)
	// 64 and 65 are 0.5 away, then 63 and 66 at 1.5.
	if got := keys(g.Nearest([]float32{64.5}, 4)); !slices.Equal(got, []int{64, 65, 63, 66}) {
		t.Fatalf("Nearest = %v, want [64 65 63 66]", got)
	}
	checkInvariants(t, g)
}

func TestNearestCosine(t *testing.T) {
	g := New[int]()
	g.Insert(1, []float32{1, 1})
	g.Insert(2, []float32{0, 1})
	g.Insert(3, []float32{1, -1})
	if got := g.Nearest([]float32{0.5, 0.5}, 1); len(got) != 1 || got[0].Key != 1 || got[0].Dist > 1e-6 {
		t.Fatalf("Nearest = %v, want key 1 at distance 0", got)
	}
}

func TestNearestOnLine(t *testing.T) {
	g := lineGraph(100)
	if res := keys(g.Nearest([]float32{50.5}, 1)); len(res) != 1 || (res[0] != 50 && res[0] != 51) {
		t.Fatalf("nearest to 50.5 = %v, want 50 or 51", res)
	}
	if res := keys(g.Nearest([]float32{0}, 3)); !slices.Equal(res, []int{0, 1, 2}) {
		t.Fatalf("nearest 3 to 0 = %v, want [0 1 2]", res)
	}
}

func TestRemoveHidesNodes(t *testing.T) {
	g := lineGraph(128)
	for i := 0; i < 128; i += 2 {
		if !g.Remove(i) {
			t.Fatalf("Remove(%d) = false", i)
		}
	}
	if g.Len() != 64 || g.Remove(-1) || g.Remove(0) {
		t.Fatalf("Len = %d after removing evens, or a repeat remove succeeded", g.Len())
	}
	for q := range 128 {
		for _, k := range keys(g.Nearest([]float32{float32(q)}, 5)) {
			if k%2 == 0 {
				t.Fatalf("query %d returned removed key %d", q, k)
			}
		}
	}
	if _, ok := g.Vector(4); ok {
		t.Fatal("Vector found a removed key")
	}
	if res := keys(g.Nearest([]float32{64}, 2)); !slices.Equal(res, []int{63, 65}) {
		t.Fatalf("nearest to 64 after removes = %v, want [63 65]", res)
	}
	checkInvariants(t, g)
}

func TestRemoveAllThenReuse(t *testing.T) {
	g := lineGraph(50)
	for i := range 50 {
		g.Remove(i)
		checkInvariants(t, g)
	}
	if g.Len() != 0 || g.Dim() != 0 || g.Nearest([]float32{1}, 3) != nil {
		t.Fatal("emptied graph should have no nodes, no dimension, and no results")
	}
	g.Insert(7, []float32{1, 2})
	if got := keys(g.Nearest([]float32{1, 2}, 1)); !slices.Equal(got, []int{7}) || g.Dim() != 2 {
		t.Fatalf("reused graph: Nearest = %v Dim = %d", got, g.Dim())
	}
}

func TestInsertReplacesKey(t *testing.T) {
	g := lineGraph(200)
	for i := 0; i < 200; i += 3 {
		g.Insert(i, []float32{float32(i) + 1000})
	}
	if g.Len() != 200 {
		t.Fatalf("Len = %d, want 200", g.Len())
	}
	if v, ok := g.Vector(3); !ok || v[0] != 1003 {
		t.Fatalf("Vector(3) = %v, %v, want 1003", v, ok)
	}
	for _, q := range []float32{3, 1003, 600} {
		for _, h := range g.Nearest([]float32{q}, 10) {
			if cur, _ := g.Vector(h.Key); h.Dist != Euclidean.Distance([]float32{q}, cur) {
				t.Fatalf("query %v: key %d at distance %v does not match its current vector %v", q, h.Key, h.Dist, cur)
			}
		}
	}
	if got := keys(g.Nearest([]float32{1003}, 1)); !slices.Equal(got, []int{3}) {
		t.Fatalf("nearest to 1003 = %v, want [3]", got)
	}
	checkInvariants(t, g)
}

func TestNearestIsRepeatable(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	g := New[int]()
	g.EfSearch = 4 // a narrow search makes the entry point matter
	for i := range 3000 {
		g.Insert(i, randVec(rng, 16))
	}
	q := randVec(rng, 16)
	first := g.Nearest(q, 10)
	for i := range 200 {
		if got := g.Nearest(q, 10); !reflect.DeepEqual(got, first) {
			t.Fatalf("call %d returned different neighbours", i)
		}
	}
}

func TestBuildIsReproducible(t *testing.T) {
	build := func() *Graph[int] {
		rng := rand.New(rand.NewSource(9))
		g := New[int]()
		for i := range 500 {
			g.Insert(i, randVec(rng, 8))
		}
		return g
	}
	a, b := build(), build()
	if !reflect.DeepEqual(a.links, b.links) || a.entry != b.entry {
		t.Fatal("two builds with the same seed and input produced different graphs")
	}
}

func TestRecall(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	g := New[int]()
	vecs := make([][]float32, 3000)
	for i := range vecs {
		vecs[i] = randVec(rng, 24)
		g.Insert(i, vecs[i])
	}
	checkInvariants(t, g)
	var hit, total int
	for range 100 {
		q := randVec(rng, 24)
		exact := make([]int, len(vecs))
		for i := range exact {
			exact[i] = i
		}
		sort.SliceStable(exact, func(a, b int) bool {
			return Cosine.Distance(q, vecs[exact[a]]) < Cosine.Distance(q, vecs[exact[b]])
		})
		want := map[int]bool{}
		for _, k := range exact[:10] {
			want[k] = true
		}
		for _, k := range keys(g.Nearest(q, 10)) {
			if want[k] {
				hit++
			}
		}
		total += 10
	}
	if r := float64(hit) / float64(total); r < 0.95 {
		t.Fatalf("recall@10 = %.3f, want >= 0.95", r)
	}
}

// TestInsertBatchParallel builds one graph on many goroutines and checks it is well formed and as
// accurate as a sequential build. Under `go test -race` it is also the data-race check for the build.
func TestInsertBatchParallel(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	n := 4000
	ids, vecs := make([]int, n), make([][]float32, n)
	for i := range n {
		ids[i], vecs[i] = i, randVec(rng, 24)
	}
	g := New[int]()
	g.Workers = 8
	g.InsertBatch(ids[:1], vecs[:1]) // a lone first node takes the entry-point path
	g.InsertBatch(ids[1:], vecs[1:])
	if g.Len() != n || g.locks != nil {
		t.Fatalf("Len = %d, locks left set: %v", g.Len(), g.locks != nil)
	}
	checkInvariants(t, g)
	for id, levels := range g.links {
		for l, list := range levels {
			if len(slices.Compact(slices.Sorted(slices.Values(list)))) != len(list) {
				t.Fatalf("node %d level %d has a duplicate link: %v", id, l, list)
			}
		}
	}
	var hit int
	for range 100 {
		q := randVec(rng, 24)
		exact := slices.Clone(ids)
		slices.SortStableFunc(exact, func(a, b int) int { return cmp.Compare(Cosine.Distance(q, vecs[a]), Cosine.Distance(q, vecs[b])) })
		want := map[int]bool{}
		for _, k := range exact[:10] {
			want[k] = true
		}
		for _, k := range keys(g.Nearest(q, 10)) {
			if want[k] {
				hit++
			}
		}
	}
	if r := float64(hit) / 1000; r < 0.95 {
		t.Fatalf("recall@10 = %.3f, want >= 0.95", r)
	}
}

func TestInsertBatchOneWorkerMatchesInserts(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	keys, vecs := make([]int, 500), make([][]float32, 500)
	for i := range keys {
		keys[i], vecs[i] = i, randVec(rng, 8)
	}
	a, b := New[int](), New[int]()
	for i := range keys {
		a.Insert(keys[i], vecs[i])
	}
	b.Workers = 1
	b.InsertBatch(keys, vecs)
	if !reflect.DeepEqual(a.links, b.links) || a.entry != b.entry {
		t.Fatal("a one-worker batch built a different graph than the same Inserts")
	}
}

func TestInsertBatchDuplicateKeysKeepLast(t *testing.T) {
	g := lineGraph(20)
	g.Workers = 4
	batch := []int{3, 50, 3, 51}
	vecs := [][]float32{{100}, {50}, {300}, {51}}
	g.InsertBatch(batch, vecs)
	if v, _ := g.Vector(3); g.Len() != 22 || v[0] != 300 {
		t.Fatalf("Len = %d, Vector(3) = %v; want 22 and the last vector, 300", g.Len(), v)
	}
	if got := keys(g.Nearest([]float32{299}, 1)); !slices.Equal(got, []int{3}) {
		t.Fatalf("nearest to 299 = %v, want [3]", got)
	}
	checkInvariants(t, g)
}

func TestInsertBatchMismatchPanicsUnchanged(t *testing.T) {
	g := lineGraph(5)
	defer func() {
		if recover() == nil {
			t.Fatal("a batch with a wrong-dimension vector did not panic")
		}
		if g.Len() != 5 {
			t.Fatalf("Len = %d after a rejected batch, want 5", g.Len())
		}
	}()
	g.InsertBatch([]int{10, 11}, [][]float32{{1}, {1, 2}})
}

func TestDimensionMismatchPanics(t *testing.T) {
	g := New[int]()
	g.Insert(1, []float32{1, 0})
	for name, fn := range map[string]func(){
		"insert":  func() { g.Insert(2, []float32{1, 0, 0}) },
		"nearest": func() { g.Nearest([]float32{1}, 1) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s with the wrong dimension did not panic", name)
				}
			}()
			fn()
		}()
	}
}

func TestConcurrentNearest(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	g := New[string]()
	for i := range 2000 {
		g.Insert(fmt.Sprint(i), randVec(rng, 8))
	}
	q := randVec(rng, 8)
	want := g.Nearest(q, 5)
	done := make(chan bool)
	for range 8 {
		go func() {
			ok := true
			for range 200 {
				ok = ok && reflect.DeepEqual(g.Nearest(q, 5), want)
			}
			done <- ok
		}()
	}
	for range 8 {
		if !<-done {
			t.Fatal("a concurrent search returned a different result")
		}
	}
}
