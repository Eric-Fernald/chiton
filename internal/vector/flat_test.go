package vector

import "testing"

func TestFlatSearchOrdersByCosine(t *testing.T) {
	f := NewFlat()
	f.Add([]Item{
		{ID: "x", Vec: []float32{1, 0, 0}},
		{ID: "y", Vec: []float32{0, 1, 0}},
		{ID: "xy", Vec: []float32{1, 1, 0}},
	})
	if f.Len() != 3 {
		t.Fatalf("Len = %d, want 3", f.Len())
	}

	res := f.Search([]float32{1, 0, 0}, 3)
	if len(res) != 3 {
		t.Fatalf("expected 3 results, got %d", len(res))
	}
	// x is identical → score 1; xy is 45° → ~0.707; y is orthogonal → 0.
	if res[0].ID != "x" {
		t.Fatalf("nearest = %q, want x", res[0].ID)
	}
	if res[1].ID != "xy" {
		t.Fatalf("second = %q, want xy", res[1].ID)
	}
	if res[2].ID != "y" {
		t.Fatalf("third = %q, want y", res[2].ID)
	}
	if res[0].Score < 0.999 {
		t.Fatalf("identical vector score = %v, want ~1", res[0].Score)
	}
}

func TestFlatSearchK(t *testing.T) {
	f := NewFlat()
	f.Add([]Item{
		{ID: "a", Vec: []float32{1, 0}},
		{ID: "b", Vec: []float32{0.9, 0.1}},
		{ID: "c", Vec: []float32{0, 1}},
	})
	res := f.Search([]float32{1, 0}, 2)
	if len(res) != 2 {
		t.Fatalf("expected 2 results with k=2, got %d", len(res))
	}
}

func TestFlatReplaceByID(t *testing.T) {
	f := NewFlat()
	f.Add([]Item{{ID: "a", Vec: []float32{1, 0}}})
	f.Add([]Item{{ID: "a", Vec: []float32{0, 1}}}) // replace
	if f.Len() != 1 {
		t.Fatalf("Len = %d, want 1 after replace", f.Len())
	}
	res := f.Search([]float32{0, 1}, 1)
	if res[0].Score < 0.999 {
		t.Fatalf("replaced vector not used: score=%v", res[0].Score)
	}
}

func TestFlatDelete(t *testing.T) {
	f := NewFlat()
	f.Add([]Item{
		{ID: "a", Vec: []float32{1, 0}},
		{ID: "b", Vec: []float32{0, 1}},
		{ID: "c", Vec: []float32{1, 1}},
	})
	f.Delete([]string{"b", "missing"})
	if f.Len() != 2 {
		t.Fatalf("Len = %d, want 2 after delete", f.Len())
	}
	for _, r := range f.Search([]float32{0, 1}, 5) {
		if r.ID == "b" {
			t.Fatal("deleted id b still returned")
		}
	}
	// Surviving ids must still resolve.
	res := f.Search([]float32{1, 0}, 1)
	if res[0].ID != "a" {
		t.Fatalf("after delete nearest to (1,0) = %q, want a", res[0].ID)
	}
}

func TestFlatEdgeCases(t *testing.T) {
	f := NewFlat()
	if got := f.Search([]float32{1, 0}, 3); got != nil {
		t.Fatal("search on empty index should return nil")
	}
	f.Add([]Item{{ID: "a", Vec: []float32{1, 0}}})
	f.Add([]Item{{ID: "z", Vec: nil}}) // zero-length skipped
	if f.Len() != 1 {
		t.Fatalf("Len = %d, want 1 (nil vec skipped)", f.Len())
	}
	if got := f.Search([]float32{1, 0}, 0); got != nil {
		t.Fatal("k=0 should return nil")
	}
	if got := f.Search(nil, 3); got != nil {
		t.Fatal("nil query should return nil")
	}
	// Dimension mismatch is skipped, not panicked.
	if got := f.Search([]float32{1, 0, 0}, 3); got != nil {
		t.Fatal("mismatched-dim query should yield no results")
	}
}
