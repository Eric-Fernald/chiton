package vector

import (
	"sync"
	"testing"
)

func TestResolve(t *testing.T) {
	for _, tc := range []struct {
		opts Options
		n    int
		want string
	}{
		{Options{}, DefaultHNSWThreshold - 1, "flat"},
		{Options{Backend: "auto", Threshold: 5}, 5, "hnsw"},
		{Options{Backend: "Flat"}, 1 << 30, "flat"},
		{Options{Backend: "hnsw"}, 1, "hnsw"},
	} {
		if got, err := Resolve(tc.opts, tc.n); err != nil || got != tc.want {
			t.Errorf("Resolve(%+v, %d) = %q, %v; want %q", tc.opts, tc.n, got, err, tc.want)
		}
	}
	if _, err := Resolve(Options{Backend: "nope"}, 1); err == nil {
		t.Error("Resolve accepted an unknown backend")
	}
}

func TestSwappableDelegatesAndSwaps(t *testing.T) {
	flat := NewFlat()
	flat.Add([]Item{{ID: "a", Vec: []float32{1, 0}}})
	sw := NewSwappable(flat)
	if sw.Len() != 1 || BackendName(sw) != "flat" {
		t.Fatalf("Len %d, backend %s; want 1, flat", sw.Len(), BackendName(sw))
	}
	if res := sw.Search([]float32{1, 0}, 1); len(res) != 1 || res[0].ID != "a" {
		t.Fatalf("Search before swap = %v", res)
	}

	h := NewHNSW(HNSWConfig{})
	h.Add([]Item{{ID: "a", Vec: []float32{1, 0}}, {ID: "b", Vec: []float32{0, 1}}})
	sw.Swap(h)
	if sw.Len() != 2 || BackendName(sw) != "hnsw" {
		t.Fatalf("after swap: Len %d, backend %s; want 2, hnsw", sw.Len(), BackendName(sw))
	}
	if res := sw.Search([]float32{0, 1}, 1); len(res) != 1 || res[0].ID != "b" {
		t.Fatalf("Search after swap = %v", res)
	}
	sw.Delete([]string{"b"})
	sw.Add([]Item{{ID: "c", Vec: []float32{1, 1}}})
	if sw.Len() != 2 {
		t.Fatalf("Add/Delete not delegated: Len %d", sw.Len())
	}
}

// TestSwappableSwapDuringSearch is a race-detector check: searches keep running while the backend swaps.
func TestSwappableSwapDuringSearch(t *testing.T) {
	items := []Item{{ID: "a", Vec: []float32{1, 0}}, {ID: "b", Vec: []float32{0, 1}}}
	flat := NewFlat()
	flat.Add(items)
	sw := NewSwappable(flat)
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 500 {
				if res := sw.Search([]float32{1, 0}, 1); len(res) != 1 || res[0].ID != "a" {
					t.Errorf("bad result during swap: %v", res)
					return
				}
			}
		})
	}
	for range 20 {
		h := NewHNSW(HNSWConfig{})
		h.Add(items)
		sw.Swap(h)
	}
	wg.Wait()
}
