package ingest

import (
	"context"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules" // register deterministic rules
	"github.com/Eric-Fernald/chiton/internal/cache"
	"github.com/Eric-Fernald/chiton/internal/embed"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

type harness struct {
	dir      string // source tree
	graphDir string
	gs       *badgerstore.Store
	cache    *cache.Cache
	eng      *Engine
}

func newHarness(t *testing.T, reembed, reanalyze bool) *harness {
	t.Helper()
	root := t.TempDir()
	src := filepath.Join(root, "src")
	if err := os.MkdirAll(src, 0o755); err != nil {
		t.Fatal(err)
	}
	graphDir := filepath.Join(root, "chiton-out")
	gs, err := badgerstore.Open(filepath.Join(graphDir, "graph"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { gs.Close() })

	ch, err := cache.Load(filepath.Join(graphDir, "cache.json"))
	if err != nil {
		t.Fatal(err)
	}

	var emb embed.Embedder
	if reembed {
		emb = embed.NewHashEmbedder(64)
	}
	var az *analyze.Analyzer
	if reanalyze {
		az = analyze.New(gs, nil)
	}
	eng := New(gs, ch, Config{
		GraphPath: filepath.Join(graphDir, "graph"),
		Reembed:   reembed,
		Reanalyze: reanalyze,
	}, emb, az, func(string, ...any) {})

	return &harness{dir: src, graphDir: graphDir, gs: gs, cache: ch, eng: eng}
}

func (h *harness) write(t *testing.T, name, content string) string {
	t.Helper()
	p := filepath.Join(h.dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func (h *harness) nodesForURI(t *testing.T, uri string) map[string]model.Node {
	t.Helper()
	all, err := h.gs.AllNodes(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]model.Node{}
	for _, n := range all {
		if n.SourceURI == uri {
			out[n.Label] = n
		}
	}
	return out
}

func (h *harness) hasEdge(t *testing.T, src, dst string, kind model.EdgeKind) bool {
	t.Helper()
	edges, err := h.gs.AllEdges(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range edges {
		if e.Src == src && e.Dst == dst && e.Kind == kind {
			return true
		}
	}
	return false
}

func (h *harness) edgeTouches(t *testing.T, id string) bool {
	t.Helper()
	edges, err := h.gs.AllEdges(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range edges {
		if e.Src == id || e.Dst == id {
			return true
		}
	}
	return false
}

// TestSyncFileReconciles is the core test: add, modify (renaming a symbol), and
// delete a file, verifying the graph reflects each state with no stale nodes or
// dangling edges.
func TestSyncFileReconciles(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()

	p := h.write(t, "foo.go", "package src\nfunc Foo() { Bar() }\nfunc Bar() {}\n")
	uri := uriFor(p)
	fooID := model.NewNodeID(uri, "Foo")
	barID := model.NewNodeID(uri, "Bar")
	bazID := model.NewNodeID(uri, "Baz")

	if _, err := h.eng.SyncFile(ctx, p); err != nil {
		t.Fatalf("SyncFile: %v", err)
	}
	nodes := h.nodesForURI(t, uri)
	if _, ok := nodes["Foo"]; !ok {
		t.Error("Foo symbol missing after initial sync")
	}
	if _, ok := nodes["Bar"]; !ok {
		t.Error("Bar symbol missing after initial sync")
	}
	if !h.hasEdge(t, fooID, barID, model.EdgeCalls) {
		t.Error("expected CALLS edge Foo->Bar")
	}

	// Modify: rename Bar to Baz, Foo now calls Baz.
	h.write(t, "foo.go", "package src\nfunc Foo() { Baz() }\nfunc Baz() {}\n")
	if _, err := h.eng.SyncFile(ctx, p); err != nil {
		t.Fatalf("SyncFile modify: %v", err)
	}
	nodes = h.nodesForURI(t, uri)
	if _, ok := nodes["Bar"]; ok {
		t.Error("stale Bar symbol still present after rename")
	}
	if _, ok := nodes["Baz"]; !ok {
		t.Error("Baz symbol missing after rename")
	}
	if !h.hasEdge(t, fooID, bazID, model.EdgeCalls) {
		t.Error("expected CALLS edge Foo->Baz after rename")
	}
	if h.edgeTouches(t, barID) {
		t.Error("dangling edge referencing removed Bar node")
	}

	// Delete the file.
	if err := os.Remove(p); err != nil {
		t.Fatal(err)
	}
	if _, err := h.eng.DeleteFile(ctx, p); err != nil {
		t.Fatalf("DeleteFile: %v", err)
	}
	if n := h.nodesForURI(t, uri); len(n) != 0 {
		t.Errorf("expected all nodes gone after delete, got %d", len(n))
	}
	if h.edgeTouches(t, fooID) {
		t.Error("edges remained after deleting the file")
	}
}

// TestSyncFileSkipsUnchanged verifies the content cache short-circuits a
// no-change re-sync.
func TestSyncFileSkipsUnchanged(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	p := h.write(t, "a.go", "package src\nfunc A() {}\n")

	if out, err := h.eng.SyncFile(ctx, p); err != nil || out.Skipped {
		t.Fatalf("first sync: out=%+v err=%v", out, err)
	}
	out, err := h.eng.SyncFile(ctx, p)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Skipped || out.Reason != "unchanged" {
		t.Errorf("second sync should skip unchanged, got %+v", out)
	}
}

// TestReembed verifies a documented symbol gets an embedding when --embed is on.
func TestReembed(t *testing.T) {
	h := newHarness(t, true, false)
	ctx := context.Background()
	p := h.write(t, "doc.go", "package src\n// Greet says hello.\nfunc Greet() {}\n")

	out, err := h.eng.SyncFile(ctx, p)
	if err != nil {
		t.Fatal(err)
	}
	if out.Reembedded == 0 {
		t.Fatal("expected at least one node re-embedded")
	}
	es := store.EmbeddingStore(h.gs)
	embs, err := es.AllEmbeddings(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(embs) == 0 {
		t.Error("no embeddings persisted after re-embed")
	}
}

// TestReanalyze verifies deterministic findings are created and pruned on sync.
func TestReanalyze(t *testing.T) {
	h := newHarness(t, false, true)
	ctx := context.Background()
	p := h.write(t, "todo.go", "package src\n// TODO: refactor\nfunc T() {}\n")
	uri := uriFor(p)

	if _, err := h.eng.SyncFile(ctx, p); err != nil {
		t.Fatal(err)
	}
	views, _ := analyze.List(ctx, h.gs, analyze.ListOptions{Scope: uri})
	if len(findingsWithRule(views, "quality/todo-comment")) != 1 {
		t.Fatalf("expected 1 todo finding, got %d", len(views))
	}

	h.write(t, "todo.go", "package src\nfunc T() {}\n")
	if _, err := h.eng.SyncFile(ctx, p); err != nil {
		t.Fatal(err)
	}
	views, _ = analyze.List(ctx, h.gs, analyze.ListOptions{Scope: uri})
	if got := len(findingsWithRule(views, "quality/todo-comment")); got != 0 {
		t.Errorf("expected todo finding pruned after fix, got %d", got)
	}
}

// TestSyncDirPrunesRemovedFiles verifies one-shot reconciliation removes nodes
// for files that no longer exist.
func TestSyncDirPrunesRemovedFiles(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	pa := h.write(t, "a.go", "package src\nfunc A() {}\n")
	pb := h.write(t, "b.go", "package src\nfunc B() {}\n")

	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	if len(h.nodesForURI(t, uriFor(pa))) == 0 || len(h.nodesForURI(t, uriFor(pb))) == 0 {
		t.Fatal("both files should be indexed after SyncDir")
	}

	if err := os.Remove(pb); err != nil {
		t.Fatal(err)
	}
	sum, err := h.eng.SyncDir(ctx, h.dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if sum.Deleted != 1 {
		t.Errorf("expected 1 pruned file, got %d", sum.Deleted)
	}
	if n := len(h.nodesForURI(t, uriFor(pb))); n != 0 {
		t.Errorf("b.go nodes should be pruned, got %d", n)
	}
	if len(h.nodesForURI(t, uriFor(pa))) == 0 {
		t.Error("a.go nodes should remain")
	}
}

// TestSyncDirPruneKeepsPresentExcludedFiles verifies prune deletes only files
// genuinely gone from disk — a file that still exists but was excluded from the
// walk (here, because it lives under an ignored directory that build indexed)
// must NOT be pruned.
func TestSyncDirPruneKeepsPresentExcludedFiles(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()

	// Index a file that lives under a dir SyncDir's walk ignores (vendor),
	// simulating a prior `chiton build` that indexed it (build doesn't skip it).
	vendorDir := filepath.Join(h.dir, "vendor")
	if err := os.MkdirAll(vendorDir, 0o755); err != nil {
		t.Fatal(err)
	}
	vendored := filepath.Join(vendorDir, "v.go")
	if err := os.WriteFile(vendored, []byte("package vendor\nfunc V() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := h.eng.SyncFile(ctx, vendored); err != nil {
		t.Fatal(err)
	}

	pa := h.write(t, "a.go", "package src\nfunc A() {}\n")

	// SyncDir walks h.dir, skips vendor/, but must not prune v.go (still on disk).
	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	if len(h.nodesForURI(t, uriFor(vendored))) == 0 {
		t.Error("vendored file (present on disk, excluded from walk) was wrongly pruned")
	}
	if len(h.nodesForURI(t, uriFor(pa))) == 0 {
		t.Error("a.go should be indexed")
	}

	// Now actually delete the vendored file; prune SHOULD remove it.
	if err := os.Remove(vendored); err != nil {
		t.Fatal(err)
	}
	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	if n := len(h.nodesForURI(t, uriFor(vendored))); n != 0 {
		t.Errorf("truly-removed vendored file should be pruned, got %d nodes", n)
	}
}

// TestIncrementalReclusterUpdatesSidecar seeds a sidecar, then verifies a sync
// updates it to include a newly added file's nodes.
func TestIncrementalReclusterUpdatesSidecar(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	pa := h.write(t, "a.go", "package src\nfunc A() {}\n")
	if _, err := h.eng.SyncFile(ctx, pa); err != nil {
		t.Fatal(err)
	}

	// Seed a full cluster sidecar (mirrors `chiton cluster`).
	nodes, _ := h.gs.AllNodes(ctx)
	edges, _ := h.gs.AllEdges(ctx)
	seed := cluster.RunIncremental(nil, nodes, edges, map[string]bool{}, 1.0, rand.New(rand.NewSource(1)))
	sidecar := cluster.SidecarPath(h.eng.cfg.GraphPath)
	if err := cluster.WriteSidecar(sidecar, seed); err != nil {
		t.Fatal(err)
	}
	seedCount := seed.NodeCount

	// Add a new file and sync it as a batch (drives an incremental recluster).
	pb := h.write(t, "b.go", "package src\nfunc B() {}\n")
	sum, err := h.eng.SyncChanges(ctx, []Change{{Path: pb, Op: OpUpsert}})
	if err != nil {
		t.Fatal(err)
	}
	if !sum.Reclustered {
		t.Error("expected SyncChanges to re-cluster when a sidecar exists")
	}

	updated, err := cluster.LoadSidecar(sidecar)
	if err != nil {
		t.Fatal(err)
	}
	if updated.NodeCount <= seedCount {
		t.Errorf("sidecar node count did not grow: seed=%d updated=%d", seedCount, updated.NodeCount)
	}
	bID := model.NewNodeID(uriFor(pb), "B")
	if _, ok := updated.Nodes[bID]; !ok {
		t.Error("new symbol B missing from updated sidecar")
	}
}

// TestWatchGracefulShutdown drives the full watch lifecycle: start, observe a
// new file get indexed, cancel the context, and confirm the store reopens
// cleanly with the change persisted. This is the supported shutdown path
// (Ctrl-C → context cancel → drain → store.Close); a hard kill is not.
func TestWatchGracefulShutdown(t *testing.T) {
	root := t.TempDir()
	src := filepath.Join(root, "src")
	if err := os.MkdirAll(src, 0o755); err != nil {
		t.Fatal(err)
	}
	graphPath := filepath.Join(root, "chiton-out", "graph")

	gs, err := badgerstore.Open(graphPath)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	ch, err := cache.Load(filepath.Join(root, "chiton-out", "cache.json"))
	if err != nil {
		t.Fatal(err)
	}
	eng := New(gs, ch, Config{
		GraphPath: graphPath,
		Debounce:  100 * time.Millisecond,
	}, nil, nil, func(string, ...any) {})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- eng.Watch(ctx, root) }()

	// Let the watcher register its directory watches before we write — fsnotify
	// only reports events that occur after a watch is added.
	time.Sleep(500 * time.Millisecond)

	p := filepath.Join(src, "w.go")
	if err := os.WriteFile(p, []byte("package src\nfunc W() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	uri := uriFor(p)

	// Wait until W appears in the store (watcher processed the write).
	indexed := false
	deadline := time.After(5 * time.Second)
poll:
	for {
		nodes, _ := gs.AllNodes(ctx)
		for _, n := range nodes {
			if n.SourceURI == uri && n.Label == "W" {
				indexed = true
				break poll
			}
		}
		select {
		case <-deadline:
			break poll
		case <-time.After(20 * time.Millisecond):
		}
	}

	// Always stop the watcher and close the store before asserting, so a
	// failure can't leave a goroutine holding the store open (blocking cleanup).
	cancel()
	var watchErr error
	select {
	case watchErr = <-done:
	case <-time.After(35 * time.Second):
		t.Fatal("Watch did not return after context cancel")
	}
	if err := gs.Close(); err != nil {
		t.Fatalf("store close: %v", err)
	}

	if !indexed {
		t.Fatal("watcher never indexed the new file")
	}
	if watchErr != nil {
		t.Fatalf("Watch returned error: %v", watchErr)
	}

	// Graceful path: the store must reopen cleanly with the change persisted.
	reopened, err := badgerstore.Open(graphPath)
	if err != nil {
		t.Fatalf("store did not reopen cleanly after graceful shutdown: %v", err)
	}
	defer reopened.Close()
	nodes, err := reopened.AllNodes(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, n := range nodes {
		if n.SourceURI == uri && n.Label == "W" {
			found = true
		}
	}
	if !found {
		t.Error("change not persisted across graceful shutdown")
	}
}

func findingsWithRule(views []analyze.FindingView, ruleID string) []analyze.FindingView {
	var out []analyze.FindingView
	for _, v := range views {
		if v.RuleID == ruleID {
			out = append(out, v)
		}
	}
	return out
}
