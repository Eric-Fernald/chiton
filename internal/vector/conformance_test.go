package vector

import "testing"

// backends lists every Index implementation so behaviour shared with Flat is checked once for all of them.
// The indexes here are tiny, so HNSW search is exhaustive and its results are exact.
var backends = []struct {
	name string
	new  func() Index
}{
	{"flat", func() Index { return NewFlat() }},
	{"hnsw", func() Index { return NewHNSW(HNSWConfig{}) }},
}

func forEachBackend(t *testing.T, fn func(t *testing.T, idx Index)) {
	t.Helper()
	for _, b := range backends {
		t.Run(b.name, func(t *testing.T) { fn(t, b.new()) })
	}
}

func TestIndexOrdersByCosine(t *testing.T) {
	forEachBackend(t, func(t *testing.T, idx Index) {
		idx.Add([]Item{
			{ID: "x", Vec: []float32{1, 0, 0}},
			{ID: "y", Vec: []float32{0, 1, 0}},
			{ID: "xy", Vec: []float32{1, 1, 0}},
		})
		if idx.Len() != 3 {
			t.Fatalf("Len = %d, want 3", idx.Len())
		}
		res := idx.Search([]float32{2, 0, 0}, 3)
		if len(res) != 3 || res[0].ID != "x" || res[1].ID != "xy" || res[2].ID != "y" {
			t.Fatalf("order = %v, want x, xy, y", res)
		}
		if res[0].Score < 0.999 || res[1].Score < 0.70 || res[1].Score > 0.71 || res[2].Score > 1e-6 {
			t.Fatalf("scores = %v, want ~1, ~0.707, ~0", res)
		}
	})
}

func TestIndexTiesBreakByID(t *testing.T) {
	forEachBackend(t, func(t *testing.T, idx Index) {
		idx.Add([]Item{
			{ID: "c", Vec: []float32{1, 0}},
			{ID: "a", Vec: []float32{2, 0}},
			{ID: "b", Vec: []float32{3, 0}},
			{ID: "d", Vec: []float32{0, 1}},
		})
		res := idx.Search([]float32{1, 0}, 3)
		if len(res) != 3 || res[0].ID != "a" || res[1].ID != "b" || res[2].ID != "c" {
			t.Fatalf("tied results = %v, want a, b, c", res)
		}
	})
}

func TestIndexSearchK(t *testing.T) {
	forEachBackend(t, func(t *testing.T, idx Index) {
		idx.Add([]Item{
			{ID: "a", Vec: []float32{1, 0}},
			{ID: "b", Vec: []float32{0.9, 0.1}},
			{ID: "c", Vec: []float32{0, 1}},
		})
		if res := idx.Search([]float32{1, 0}, 2); len(res) != 2 {
			t.Fatalf("k=2 returned %d results", len(res))
		}
		if res := idx.Search([]float32{1, 0}, 10); len(res) != 3 {
			t.Fatalf("k=10 over 3 vectors returned %d results", len(res))
		}
	})
}

func TestIndexReplaceByID(t *testing.T) {
	forEachBackend(t, func(t *testing.T, idx Index) {
		idx.Add([]Item{{ID: "a", Vec: []float32{1, 0}}, {ID: "b", Vec: []float32{1, 1}}})
		idx.Add([]Item{{ID: "a", Vec: []float32{0, 1}}})
		if idx.Len() != 2 {
			t.Fatalf("Len = %d, want 2 after replace", idx.Len())
		}
		res := idx.Search([]float32{0, 1}, 1)
		if len(res) != 1 || res[0].ID != "a" || res[0].Score < 0.999 {
			t.Fatalf("replaced vector not used: %v", res)
		}
	})
}

func TestIndexDelete(t *testing.T) {
	forEachBackend(t, func(t *testing.T, idx Index) {
		idx.Add([]Item{
			{ID: "a", Vec: []float32{1, 0}},
			{ID: "b", Vec: []float32{0, 1}},
			{ID: "c", Vec: []float32{1, 1}},
		})
		idx.Delete([]string{"b", "missing"})
		if idx.Len() != 2 {
			t.Fatalf("Len = %d, want 2 after delete", idx.Len())
		}
		for _, r := range idx.Search([]float32{0, 1}, 5) {
			if r.ID == "b" {
				t.Fatal("deleted id b still returned")
			}
		}
		if res := idx.Search([]float32{1, 0}, 1); len(res) != 1 || res[0].ID != "a" {
			t.Fatalf("after delete nearest to (1,0) = %v, want a", res)
		}
		idx.Delete([]string{"a", "c"})
		if idx.Len() != 0 || idx.Search([]float32{1, 0}, 3) != nil {
			t.Fatal("index emptied by delete should be empty and return nil")
		}
	})
}

func TestIndexEdgeCases(t *testing.T) {
	forEachBackend(t, func(t *testing.T, idx Index) {
		if got := idx.Search([]float32{1, 0}, 3); got != nil {
			t.Fatal("search on empty index should return nil")
		}
		idx.Add([]Item{{ID: "a", Vec: []float32{1, 0}}})
		idx.Add([]Item{{ID: "z", Vec: nil}})
		if idx.Len() != 1 {
			t.Fatalf("Len = %d, want 1 (nil vec skipped)", idx.Len())
		}
		for name, q := range map[string][]float32{"nil": nil, "zero": {0, 0}, "wrong dim": {1, 0, 0}} {
			if got := idx.Search(q, 3); got != nil {
				t.Errorf("%s query returned %v, want nil", name, got)
			}
		}
		for _, k := range []int{0, -1} {
			if got := idx.Search([]float32{1, 0}, k); got != nil {
				t.Errorf("k=%d returned %v, want nil", k, got)
			}
		}
	})
}
