package vector

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sync"
	"testing"
)

func TestHNSWSkipsInvalidVectors(t *testing.T) {
	h := NewHNSW(HNSWConfig{})
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	h.Add([]Item{
		{ID: "ok", Vec: []float32{1, 0, 0}},
		{ID: "short", Vec: []float32{1, 0}},
		{ID: "long", Vec: []float32{1, 0, 0, 0}},
		{ID: "empty", Vec: nil},
		{ID: "zero", Vec: []float32{0, 0, 0}},
		{ID: "nan", Vec: []float32{nan, 1, 0}},
		{ID: "inf", Vec: []float32{inf, 1, 0}},
		{ID: "ok2", Vec: []float32{0, 1, 0}},
	})
	if h.Len() != 2 || h.Skipped() != 6 || h.Dim() != 3 {
		t.Fatalf("Len=%d Skipped=%d Dim=%d, want 2, 6, 3", h.Len(), h.Skipped(), h.Dim())
	}
	for name, q := range map[string][]float32{"nan": {nan, 0, 0}, "inf": {inf, 0, 0}, "short": {1, 0}} {
		if got := h.Search(q, 2); got != nil {
			t.Errorf("%s query returned %v, want nil", name, got)
		}
	}
}

func TestHNSWConfiguredDimRejectsOthers(t *testing.T) {
	h := NewHNSW(HNSWConfig{Dim: 2})
	h.Add([]Item{{ID: "three", Vec: []float32{1, 0, 0}}, {ID: "two", Vec: []float32{1, 0}}})
	if h.Len() != 1 || h.Skipped() != 1 || h.Dim() != 2 {
		t.Fatalf("Len=%d Skipped=%d Dim=%d, want 1, 1, 2", h.Len(), h.Skipped(), h.Dim())
	}
}

// TestHNSWInvalidReplacementKeepsOld matches Flat: a skipped vector never evicts the stored one.
func TestHNSWInvalidReplacementKeepsOld(t *testing.T) {
	h := NewHNSW(HNSWConfig{})
	h.Add([]Item{{ID: "a", Vec: []float32{1, 0}}})
	h.Add([]Item{{ID: "a", Vec: []float32{0, 0}}})
	if res := h.Search([]float32{1, 0}, 1); len(res) != 1 || res[0].ID != "a" || res[0].Score < 0.999 {
		t.Fatalf("old vector lost after invalid replace: %v", res)
	}
}

// TestHNSWScoresMatchFlat checks every returned score is the exact cosine Flat reports for that ID.
func TestHNSWScoresMatchFlat(t *testing.T) {
	items := randomItems(rand.New(rand.NewSource(3)), 2000, 32)
	f, h := NewFlat(), NewHNSW(HNSWConfig{Seed: 1})
	f.Add(items)
	h.Add(items)
	exact := map[string]float32{}
	q := randomVec(rand.New(rand.NewSource(4)), 32)
	for _, r := range f.Search(q, f.Len()) {
		exact[r.ID] = r.Score
	}
	res := h.Search(q, 20)
	if len(res) != 20 {
		t.Fatalf("got %d results, want 20", len(res))
	}
	for _, r := range res {
		if d := math.Abs(float64(r.Score - exact[r.ID])); d > 1e-5 {
			t.Fatalf("%s score %v, flat %v", r.ID, r.Score, exact[r.ID])
		}
	}
	if !slices.IsSortedFunc(res, rankOrder) {
		t.Fatalf("results not in rank order: %v", res)
	}
}

// TestHNSWRecall is the T2 acceptance check: recall@8 against exact search at the default parameters, on
// random embedding-like vectors.
func TestHNSWRecall(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a 20k-vector index")
	}
	gen := newEmbeddingLike(rand.New(rand.NewSource(5)), 128, 200)
	items := gen.items(20000)
	f, h := NewFlat(), NewHNSW(HNSWConfig{Seed: 1})
	f.Add(items)
	h.Add(items)
	if r := recallAt(f, h, gen.queries(200), 8); r < 0.95 {
		t.Fatalf("recall@8 = %.3f, want >= 0.95", r)
	}
}

// TestHNSWRecallUnstructured pins the worst case: iid Gaussian vectors have no neighbourhood structure, so
// recall is lower. It guards against regressions rather than promising embedding-level recall.
func TestHNSWRecallUnstructured(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a 20k-vector index")
	}
	rng := rand.New(rand.NewSource(5))
	items := randomItems(rng, 20000, 64)
	f, h := NewFlat(), NewHNSW(HNSWConfig{Seed: 1})
	f.Add(items)
	h.Add(items)
	queries := make([][]float32, 200)
	for i := range queries {
		queries[i] = randomVec(rng, 64)
	}
	if r := recallAt(f, h, queries, 8); r < 0.85 {
		t.Fatalf("recall@8 = %.3f, want >= 0.85", r)
	}
}

// TestHNSWConcurrentSearch runs many readers against a frozen index. Under `go test -race` it is the R6
// thread-safety check; without the race detector it still checks every concurrent result is well formed.
func TestHNSWConcurrentSearch(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	h := NewHNSW(HNSWConfig{Seed: 1})
	h.Add(randomItems(rng, 3000, 16))
	queries := make([][]float32, 64)
	for i := range queries {
		queries[i] = randomVec(rng, 16)
	}
	var wg sync.WaitGroup
	errs := make(chan string, 16)
	for g := range 16 {
		wg.Go(func() {
			for i := range 100 {
				res := h.Search(queries[(g+i)%len(queries)], 8)
				if len(res) != 8 || !slices.IsSortedFunc(res, rankOrder) {
					errs <- fmt.Sprintf("goroutine %d: bad result %v", g, res)
					return
				}
			}
		})
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}
}

func randomVec(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

func randomItems(rng *rand.Rand, n, dim int) []Item {
	items := make([]Item, n)
	for i := range items {
		items[i] = Item{ID: fmt.Sprintf("n%07d", i), Vec: randomVec(rng, dim)}
	}
	return items
}

// recallAt is the fraction of exact's top-k IDs that approx also returns, averaged over queries.
func recallAt(exact, approx Index, queries [][]float32, k int) float64 {
	var hit, total int
	for _, q := range queries {
		want := map[string]bool{}
		for _, r := range exact.Search(q, k) {
			want[r.ID] = true
		}
		for _, r := range approx.Search(q, k) {
			if want[r.ID] {
				hit++
			}
		}
		total += len(want)
	}
	return float64(hit) / float64(total)
}
