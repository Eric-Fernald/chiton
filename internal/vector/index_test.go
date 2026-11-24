package vector

import "testing"

func TestNewIndexSelectsBackend(t *testing.T) {
	for _, tc := range []struct {
		backend   string
		threshold int
		n         int
		want      string
	}{
		{"", 0, DefaultHNSWThreshold - 1, "flat"},
		{"auto", 0, DefaultHNSWThreshold, "hnsw"},
		{"AUTO", 10, 9, "flat"},
		{"auto", 10, 10, "hnsw"},
		{"flat", 10, 1_000_000, "flat"},
		{"hnsw", 0, 3, "hnsw"},
		{" HNSW ", 0, 3, "hnsw"},
	} {
		idx, err := NewIndex(Options{Backend: tc.backend, Threshold: tc.threshold}, tc.n)
		if err != nil {
			t.Fatalf("%+v: %v", tc, err)
		}
		if got := BackendName(idx); got != tc.want {
			t.Errorf("backend %q threshold %d n %d: got %s, want %s", tc.backend, tc.threshold, tc.n, got, tc.want)
		}
	}
}

func TestNewIndexRejectsUnknownBackend(t *testing.T) {
	if _, err := NewIndex(Options{Backend: "faiss"}, 10); err == nil {
		t.Fatal("unknown backend accepted")
	}
}

func TestNewIndexPassesHNSWConfig(t *testing.T) {
	idx, err := NewIndex(Options{Backend: "hnsw", HNSW: HNSWConfig{Dim: 3, M: 8, EfSearch: 40, EfConstruction: 90}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	h := idx.(*HNSW)
	if h.Dim() != 3 || h.g.M != 8 || h.g.EfSearch != 40 || h.g.EfConstruction != 90 {
		t.Fatalf("config not applied: dim %d M %d ef %d efC %d", h.Dim(), h.g.M, h.g.EfSearch, h.g.EfConstruction)
	}
}
