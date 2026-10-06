package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
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
		embsOf(3, 3, 3), reload, savedIndex{}, logf)
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
		embsOf(3, 3, 3, 3), reload, savedIndex{}, logf)
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
	idx, ready, err := serveIndex(context.Background(), config.VectorConfig{Index: "hnsw"}, embsOf(3, 3), reload, savedIndex{}, logf)
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
	_, _, err := serveIndex(context.Background(), config.VectorConfig{Index: "x"}, embsOf(2), nil, savedIndex{},
		func(string, ...any) {})
	if err == nil {
		t.Fatal("unknown backend accepted")
	}
}

// fakeEmbeddings is an EmbeddingStore whose AllEmbeddings returns a fresh copy each call, as a real store does.
type fakeEmbeddings struct {
	embs  []store.Embedding
	loads int
}

func (f *fakeEmbeddings) UpsertEmbeddings(context.Context, []store.Embedding) error { return nil }

func (f *fakeEmbeddings) AllEmbeddings(context.Context) ([]store.Embedding, error) {
	f.loads++
	out := make([]store.Embedding, len(f.embs))
	for i, e := range f.embs {
		out[i] = e
		out[i].Vec = slices.Clone(e.Vec)
	}
	return out, nil
}

func randomEmbs(n, dim int) []store.Embedding {
	rng := rand.New(rand.NewSource(21))
	embs := make([]store.Embedding, n)
	for i := range embs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		embs[i] = store.Embedding{NodeID: fmt.Sprintf("sym:%04d", i), Vec: v}
	}
	return embs
}

func TestSavedIndexServesWithoutBuilding(t *testing.T) {
	cfg := config.VectorConfig{Index: "auto", HNSWThreshold: 100}
	es := &fakeEmbeddings{embs: randomEmbs(300, 16)}
	saved := savedIndex{Path: filepath.Join(t.TempDir(), "hnsw.idx"), Key: bytes.Repeat([]byte{3}, 32)}
	var logs []string
	logf := func(f string, a ...any) { logs = append(logs, fmt.Sprintf(f, a...)) }

	built, err := saveIndex(context.Background(), cfg, es, "hash", saved, logf)
	if err != nil || built == nil || vector.BackendName(built) != "hnsw" {
		t.Fatalf("saveIndex = %v, %v", built, err)
	}
	if again, err := saveIndex(context.Background(), cfg, es, "hash", saved, logf); err != nil || again == nil ||
		!strings.Contains(logs[len(logs)-1], "is current") {
		t.Fatalf("second saveIndex = %v, %v, logs %q; want the saved index reused", again, err, logs)
	}

	logs = nil
	reload := func(context.Context) ([]store.Embedding, error) {
		t.Fatal("serve rebuilt despite a current saved index")
		return nil, nil
	}
	embs, _ := es.AllEmbeddings(context.Background())
	idx, ready, err := serveIndex(context.Background(), cfg, embs, reload, saved, logf)
	if err != nil {
		t.Fatal(err)
	}
	<-ready
	if vector.BackendName(idx) != "hnsw" || idx.Len() != 300 || len(logs) != 1 || !strings.Contains(logs[0], "loaded") {
		t.Fatalf("backend %s, Len %d, logs %q", vector.BackendName(idx), idx.Len(), logs)
	}
	for i, e := range embs {
		if e.Vec != nil {
			t.Fatalf("embs[%d].Vec not released after loading", i)
		}
	}
	q := randomEmbs(1, 16)[0].Vec
	if a, b := built.Search(q, 5), idx.Search(q, 5); !slices.Equal(a, b) {
		t.Fatalf("loaded index answers differently:\n%v\n%v", a, b)
	}
}

// TestSavedIndexLeavesOutOtherModels checks embed saves an index over the configured model's vectors only, and
// that serve, filtering the same way, loads it instead of rebuilding and warns about the rest.
func TestSavedIndexLeavesOutOtherModels(t *testing.T) {
	cfg := config.VectorConfig{Index: "auto", HNSWThreshold: 100}
	mine, theirs := randomEmbs(300, 16), randomEmbs(40, 16)
	for i := range mine {
		mine[i].Model = "m"
	}
	for i := range theirs {
		theirs[i].NodeID, theirs[i].Model = fmt.Sprintf("other:%04d", i), "x"
	}
	es := &fakeEmbeddings{embs: append(mine, theirs...)}
	saved := savedIndex{Path: filepath.Join(t.TempDir(), "hnsw.idx")}
	if built, err := saveIndex(context.Background(), cfg, es, "m", saved, func(string, ...any) {}); err != nil || built.Len() != 300 {
		t.Fatalf("saveIndex = %v, %v; want 300 vectors of model m", built, err)
	}

	var logs []string
	logf := func(f string, a ...any) { logs = append(logs, fmt.Sprintf(f, a...)) }
	all, _ := es.AllEmbeddings(context.Background())
	embs := keepModel(all, "m", "serve", logf)
	reload := func(context.Context) ([]store.Embedding, error) {
		t.Fatal("serve rebuilt despite a current saved index")
		return nil, nil
	}
	idx, ready, err := serveIndex(context.Background(), cfg, embs, reload, saved, logf)
	if err != nil {
		t.Fatal(err)
	}
	<-ready
	if idx.Len() != 300 || len(logs) != 2 || !strings.Contains(logs[0], `40 embedding(s) made with model "x"`) ||
		!strings.Contains(logs[1], "loaded") {
		t.Fatalf("Len %d, logs %q", idx.Len(), logs)
	}
}

func TestStaleSavedIndexFallsBackToBackgroundBuild(t *testing.T) {
	cfg := config.VectorConfig{Index: "hnsw"}
	es := &fakeEmbeddings{embs: randomEmbs(50, 8)}
	saved := savedIndex{Path: filepath.Join(t.TempDir(), "hnsw.idx")}
	if _, err := saveIndex(context.Background(), cfg, es, "hash", saved, func(string, ...any) {}); err != nil {
		t.Fatal(err)
	}
	es.embs[10].Vec[0] += 1 // the store changed after the save

	var mu sync.Mutex
	var logs []string
	logf := func(f string, a ...any) { mu.Lock(); logs = append(logs, fmt.Sprintf(f, a...)); mu.Unlock() }
	embs, _ := es.AllEmbeddings(context.Background())
	idx, ready, err := serveIndex(context.Background(), cfg, embs, es.AllEmbeddings, saved, logf)
	if err != nil {
		t.Fatal(err)
	}
	<-ready
	mu.Lock()
	defer mu.Unlock()
	if vector.BackendName(idx) != "hnsw" || idx.Len() != 50 || !strings.Contains(logs[0], "out of date") {
		t.Fatalf("backend %s, Len %d, logs %q", vector.BackendName(idx), idx.Len(), logs)
	}
}

func TestUnreadableSavedIndexReloadsForFlat(t *testing.T) {
	cfg := config.VectorConfig{Index: "hnsw"}
	es := &fakeEmbeddings{embs: randomEmbs(40, 8)}
	saved := savedIndex{Path: filepath.Join(t.TempDir(), "hnsw.idx")}
	if _, err := saveIndex(context.Background(), cfg, es, "hash", saved, func(string, ...any) {}); err != nil {
		t.Fatal(err)
	}
	// Truncate the topology after the header: the fingerprint matches, so the load takes vectors and then fails.
	data, err := os.ReadFile(saved.Path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(saved.Path, data[:len(data)-20], 0o600); err != nil {
		t.Fatal(err)
	}
	embs, _ := es.AllEmbeddings(context.Background())
	loadsBefore := es.loads
	idx, ready, err := serveIndex(context.Background(), cfg, embs, es.AllEmbeddings, saved, func(string, ...any) {})
	if err != nil {
		t.Fatal(err)
	}
	<-ready
	if idx.Len() != 40 || es.loads-loadsBefore != 2 { // one reload for Flat, one for the background build
		t.Fatalf("Len %d, %d reloads; want 40 and 2", idx.Len(), es.loads-loadsBefore)
	}
}

func TestSaveIndexSkipsSmallStores(t *testing.T) {
	es := &fakeEmbeddings{embs: randomEmbs(10, 4)}
	saved := savedIndex{Path: filepath.Join(t.TempDir(), "hnsw.idx")}
	idx, err := saveIndex(context.Background(), config.VectorConfig{Index: "auto", HNSWThreshold: 100}, es, "hash", saved,
		func(string, ...any) {})
	if err != nil || idx != nil {
		t.Fatalf("saveIndex = %v, %v; want nothing for a store under the threshold", idx, err)
	}
	if _, err := os.Stat(saved.Path); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("index file written for a small store: %v", err)
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
