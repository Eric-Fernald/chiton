package main

import (
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/config"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

func embsOf(dims ...int) []store.Embedding {
	embs := make([]store.Embedding, len(dims))
	for i, d := range dims {
		v := make([]float32, d)
		if d > 0 {
			v[i%d] = 1
		}
		embs[i] = store.Embedding{NodeID: string(rune('a' + i)), Vec: v}
	}
	return embs
}

func TestBuildIndexSelectsBackend(t *testing.T) {
	for _, tc := range []struct {
		name  string
		cfg   config.VectorConfig
		exact bool
		want  string
	}{
		{"auto below threshold", config.VectorConfig{Index: "auto", HNSWThreshold: 10}, false, "flat"},
		{"auto at threshold", config.VectorConfig{Index: "auto", HNSWThreshold: 4}, false, "hnsw"},
		{"forced hnsw", config.VectorConfig{Index: "hnsw"}, false, "hnsw"},
		{"exact overrides hnsw", config.VectorConfig{Index: "hnsw"}, true, "flat"},
	} {
		idx, rep, err := buildIndex(tc.cfg, embsOf(3, 3, 3, 3), tc.exact)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if rep.Backend != tc.want || idx.Len() != 4 || rep.Vectors != 4 {
			t.Errorf("%s: backend %s, %d indexed, report %+v; want %s with 4", tc.name, rep.Backend, idx.Len(), rep, tc.want)
		}
	}
}

func TestBuildIndexKeepsMajorityDimension(t *testing.T) {
	embs := embsOf(4, 2, 4, 4, 2, 0)
	idx, rep, err := buildIndex(config.VectorConfig{Index: "hnsw"}, embs, false)
	if err != nil {
		t.Fatal(err)
	}
	// The three 4-dim vectors win; both 2-dim vectors and the empty one are skipped.
	if idx.Len() != 3 || rep.Skipped != 3 {
		t.Fatalf("indexed %d, skipped %d; want 3 and 3", idx.Len(), rep.Skipped)
	}
	for i, e := range embs {
		if e.Vec != nil {
			t.Fatalf("embs[%d].Vec not released after indexing", i)
		}
	}
}

func TestBuildIndexRejectsUnknownBackend(t *testing.T) {
	_, _, err := buildIndex(config.VectorConfig{Index: "annoy"}, embsOf(2), false)
	if err == nil || !strings.Contains(err.Error(), "annoy") {
		t.Fatalf("err = %v, want one naming the bad backend", err)
	}
}

func TestIndexReportString(t *testing.T) {
	rep := indexReport{Backend: "hnsw", Vectors: 1200, Skipped: 3}
	if s := rep.String(); !strings.Contains(s, "1200 embeddings") || !strings.Contains(s, "hnsw index") ||
		!strings.Contains(s, "3 skipped") {
		t.Fatalf("String() = %q", s)
	}
	if s := (indexReport{Backend: "flat", Vectors: 5}).String(); strings.Contains(s, "skipped") {
		t.Fatalf("String() = %q mentions skips when there were none", s)
	}
}
