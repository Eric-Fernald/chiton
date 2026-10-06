package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/config"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/vector"
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

func TestServeIndexBelowThresholdIsFlat(t *testing.T) {
	var logs []string
	logf := func(f string, a ...any) { logs = append(logs, fmt.Sprintf(f, a...)) }
	reload := func(context.Context) ([]store.Embedding, error) {
		t.Fatal("reload called for a flat index")
		return nil, nil
	}
	idx, ready, err := serveIndex(context.Background(), config.VectorConfig{Index: "auto", HNSWThreshold: 10},
		embsOf(3, 3, 3), reload, logf)
	if err != nil {
		t.Fatal(err)
	}
	<-ready
	if vector.BackendName(idx) != "flat" || idx.Len() != 3 || len(logs) != 1 || !strings.Contains(logs[0], "flat index") {
		t.Fatalf("backend %s, Len %d, logs %q", vector.BackendName(idx), idx.Len(), logs)
	}
}

func TestServeIndexBuildsHNSWInBackground(t *testing.T) {
	var mu sync.Mutex
	var logs []string
	logf := func(f string, a ...any) { mu.Lock(); logs = append(logs, fmt.Sprintf(f, a...)); mu.Unlock() }
	release := make(chan struct{})
	reloads := 0
	reload := func(context.Context) ([]store.Embedding, error) {
		reloads++
		<-release // hold the background build until the flat phase has been checked
		return embsOf(3, 3, 3, 3), nil
	}
	idx, ready, err := serveIndex(context.Background(), config.VectorConfig{Index: "auto", HNSWThreshold: 4},
		embsOf(3, 3, 3, 3), reload, logf)
	if err != nil {
		t.Fatal(err)
	}
	if vector.BackendName(idx) != "flat" || len(idx.Search([]float32{1, 0, 0}, 2)) != 2 {
		t.Fatalf("before the build: backend %s; want a searchable flat index", vector.BackendName(idx))
	}
	close(release)
	<-ready
	mu.Lock()
	defer mu.Unlock()
	if vector.BackendName(idx) != "hnsw" || idx.Len() != 4 || reloads != 1 {
		t.Fatalf("after the build: backend %s, Len %d, reloads %d", vector.BackendName(idx), idx.Len(), reloads)
	}
	if len(logs) != 2 || !strings.Contains(logs[0], "background") || !strings.Contains(logs[1], "hnsw index ready") {
		t.Fatalf("logs = %q", logs)
	}
}

func TestServeIndexKeepsFlatWhenReloadFails(t *testing.T) {
	var logs []string
	logf := func(f string, a ...any) { logs = append(logs, fmt.Sprintf(f, a...)) }
	reload := func(context.Context) ([]store.Embedding, error) { return nil, errors.New("store closed") }
	idx, ready, err := serveIndex(context.Background(), config.VectorConfig{Index: "hnsw"}, embsOf(3, 3), reload, logf)
	if err != nil {
		t.Fatal(err)
	}
	<-ready
	if vector.BackendName(idx) != "flat" || idx.Len() != 2 {
		t.Fatalf("backend %s, Len %d; want the flat index kept", vector.BackendName(idx), idx.Len())
	}
	if !strings.Contains(logs[len(logs)-1], "store closed") {
		t.Fatalf("logs = %q, want the reload error", logs)
	}
}

func TestServeIndexRejectsUnknownBackend(t *testing.T) {
	_, _, err := serveIndex(context.Background(), config.VectorConfig{Index: "x"}, embsOf(2), nil,
		func(string, ...any) {})
	if err == nil {
		t.Fatal("unknown backend accepted")
	}
}

func TestOneShotSearchIsExactUnlessHNSWForced(t *testing.T) {
	for index, want := range map[string]bool{"": true, "auto": true, " AUTO ": true, "flat": true, "hnsw": false} {
		if got := oneShotExact(config.VectorConfig{Index: index}); got != want {
			t.Errorf("oneShotExact(%q) = %v, want %v", index, got, want)
		}
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
