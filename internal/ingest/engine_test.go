package ingest

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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
	"github.com/Eric-Fernald/chiton/internal/walk"
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

	ch, err := cache.Load(filepath.Join(graphDir, "cache.json"), gs.Epoch())
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

// TestSyncFileReconciles adds, modifies (renaming a symbol), and deletes a file, checking the graph has
// no stale nodes or dangling edges at each step.
func TestSyncFileReconciles(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()

	p := h.write(t, "foo.go", "package src\nfunc Foo() { Bar() }\nfunc Bar() {}\n")
	uri := model.FileURI(p)
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
	uri := model.FileURI(p)

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
	if len(h.nodesForURI(t, model.FileURI(pa))) == 0 || len(h.nodesForURI(t, model.FileURI(pb))) == 0 {
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
	if n := len(h.nodesForURI(t, model.FileURI(pb))); n != 0 {
		t.Errorf("b.go nodes should be pruned, got %d", n)
	}
	if len(h.nodesForURI(t, model.FileURI(pa))) == 0 {
		t.Error("a.go nodes should remain")
	}
}

// TestSyncDirPruneKeepsPresentExcludedFiles verifies prune deletes only files gone from disk, not ones
// still present but excluded from the walk (here, under an ignored dir that build indexed).
func TestSyncDirPruneKeepsPresentExcludedFiles(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()

	// Index a file under a dir SyncDir's walk ignores (vendor), simulating a store written before the ignore
	// policy existed.
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
	if len(h.nodesForURI(t, model.FileURI(vendored))) == 0 {
		t.Error("vendored file (present on disk, excluded from walk) was wrongly pruned")
	}
	if len(h.nodesForURI(t, model.FileURI(pa))) == 0 {
		t.Error("a.go should be indexed")
	}

	// Now actually delete the vendored file; prune SHOULD remove it.
	if err := os.Remove(vendored); err != nil {
		t.Fatal(err)
	}
	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	if n := len(h.nodesForURI(t, model.FileURI(vendored))); n != 0 {
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
	bID := model.NewNodeID(model.FileURI(pb), "B")
	if _, ok := updated.Nodes[bID]; !ok {
		t.Error("new symbol B missing from updated sidecar")
	}
}

// TestSyncReportsReclustered checks that BatchSummary.Reclustered is set exactly when a sidecar exists after
// the batch, including a batch that changed nothing.
func TestSyncReportsReclustered(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	p := h.write(t, "a.go", "package src\nfunc A() {}\n")
	sync := func(want bool) {
		t.Helper()
		sum, err := h.eng.SyncDir(ctx, h.dir, true)
		if err != nil {
			t.Fatal(err)
		}
		if sum.Reclustered != want {
			t.Fatalf("Reclustered = %v, want %v (summary %+v)", sum.Reclustered, want, sum)
		}
	}
	sync(false) // changed, no sidecar yet
	sync(false) // unchanged, no sidecar

	nodes, _ := h.gs.AllNodes(ctx)
	edges, _ := h.gs.AllEdges(ctx)
	if err := cluster.WriteSidecar(cluster.SidecarPath(h.eng.cfg.GraphPath), cluster.Full(nodes, edges, 1.0, nil)); err != nil {
		t.Fatal(err)
	}
	sync(true) // unchanged, sidecar exists
	h.write(t, "a.go", "package src\nfunc A() {}\nfunc B() {}\n")
	sync(true) // changed, re-clustered
	if err := os.Remove(p); err != nil {
		t.Fatal(err)
	}
	sync(true) // pruned, re-clustered
}

// TestWatchGracefulShutdown starts Watch, indexes a new file, and cancels, then checks the store reopens
// cleanly with the change persisted. The supported shutdown path is Ctrl-C, cancel, drain, then Close.
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
	ch, err := cache.Load(filepath.Join(root, "chiton-out", "cache.json"), gs.Epoch())
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
	uri := model.FileURI(p)

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

// TestSyncDirHonoursIgnorePolicy checks SyncDir skips the default directory list and the root's .chitonignore.
func TestSyncDirHonoursIgnorePolicy(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	for _, d := range []string{"gen", ".venv"} {
		if err := os.MkdirAll(filepath.Join(h.dir, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	kept := h.write(t, "a.go", "package src\nfunc A() {}\n")
	gen := h.write(t, filepath.Join("gen", "g.go"), "package gen\nfunc G() {}\n")
	venv := h.write(t, filepath.Join(".venv", "v.py"), "def v():\n    pass\n")
	h.write(t, ".chitonignore", "gen/\n")

	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	if len(h.nodesForURI(t, model.FileURI(kept))) == 0 {
		t.Error("a.go should be indexed")
	}
	for _, p := range []string{gen, venv} {
		if n := len(h.nodesForURI(t, model.FileURI(p))); n != 0 {
			t.Errorf("%s should be skipped, got %d node(s)", p, n)
		}
	}
}

// TestResyncKeepsForeignEdges reproduces the lost-linkage bug: an edit keeps a meeting's edges on a surviving
// symbol, and renaming the symbol drops them with it.
func TestResyncKeepsForeignEdges(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	p := h.write(t, "foo.go", "package src\nfunc Hello() {}\n")
	helloID := model.NewNodeID(model.FileURI(p), "Hello")
	if _, err := h.eng.SyncFile(ctx, p); err != nil {
		t.Fatal(err)
	}
	const decID, mtgID = "dec", "mtg"
	if err := h.gs.UpsertNodes(ctx, []model.Node{
		{ID: decID, Kind: model.KindDecision, Label: "keep Hello", SourceURI: "decision://meeting://standup"},
		{ID: mtgID, Kind: model.KindMeeting, Label: "standup", SourceURI: "meeting://standup"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := h.gs.UpsertEdges(ctx, []model.Edge{
		{ID: "d", Src: decID, Dst: helloID, Kind: model.EdgeDecidedIn, Provenance: model.ProvInferred},
		{ID: "m", Src: helloID, Dst: mtgID, Kind: model.EdgeDiscussedIn, Provenance: model.ProvInferred},
	}); err != nil {
		t.Fatal(err)
	}

	h.write(t, "foo.go", "package src\nfunc Hello() {}\nfunc World() {}\n")
	if _, err := h.eng.SyncFile(ctx, p); err != nil {
		t.Fatal(err)
	}
	if !h.hasEdge(t, decID, helloID, model.EdgeDecidedIn) || !h.hasEdge(t, helloID, mtgID, model.EdgeDiscussedIn) {
		t.Fatal("editing foo.go deleted the meeting edges on the surviving Hello symbol")
	}

	h.write(t, "foo.go", "package src\nfunc Greet() {}\n")
	if _, err := h.eng.SyncFile(ctx, p); err != nil {
		t.Fatal(err)
	}
	if h.edgeTouches(t, helloID) {
		t.Error("edges still reference the renamed-away Hello symbol")
	}
	if n, _ := h.gs.GetNode(ctx, decID); n == nil {
		t.Error("the decision node itself must survive")
	}
}

// TestSyncChangesStopsOnCancel checks a cancelled context stops a batch before it writes anything.
func TestSyncChangesStopsOnCancel(t *testing.T) {
	h := newHarness(t, false, false)
	p := h.write(t, "a.go", "package src\nfunc A() {}\n")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := h.eng.SyncChanges(ctx, []Change{{Path: p, Op: OpUpsert}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("SyncChanges err = %v, want context.Canceled", err)
	}
	if n := len(h.nodesForURI(t, model.FileURI(p))); n != 0 {
		t.Errorf("cancelled batch wrote %d node(s)", n)
	}
}

// TestSyncDirCountsUnreadable checks one unreadable directory is counted and skipped, not fatal.
func TestSyncDirCountsUnreadable(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("chmod cannot make a directory unreadable on Windows")
	}
	if os.Geteuid() == 0 {
		t.Skip("root can read any directory")
	}
	h := newHarness(t, false, false)
	ctx := context.Background()
	locked := filepath.Join(h.dir, "locked")
	if err := os.MkdirAll(locked, 0o755); err != nil {
		t.Fatal(err)
	}
	h.write(t, filepath.Join("locked", "x.go"), "package locked\nfunc X() {}\n")
	ok := h.write(t, "a.go", "package src\nfunc A() {}\n")
	if err := os.Chmod(locked, 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(locked, 0o755) })

	sum, err := h.eng.SyncDir(ctx, h.dir, true)
	if err != nil {
		t.Fatalf("SyncDir aborted on an unreadable dir: %v", err)
	}
	if sum.Unreadable != 1 {
		t.Errorf("Unreadable = %d, want 1", sum.Unreadable)
	}
	if len(h.nodesForURI(t, model.FileURI(ok))) == 0 {
		t.Error("a.go should still be indexed")
	}
}

// countingStore records every ReplaceSources call and each node it writes.
type countingStore struct {
	*badgerstore.Store
	calls      int
	nodeWrites map[string]int
}

func (c *countingStore) ReplaceSources(ctx context.Context, updates []store.SourceUpdate) ([]int, error) {
	c.calls++
	for _, u := range updates {
		for _, n := range u.Nodes {
			c.nodeWrites[n.ID]++
		}
	}
	return c.Store.ReplaceSources(ctx, updates)
}

// TestSyncDirBatchesWrites checks SyncDir writes many extracted files in one batched call, reports scan and
// unchanged counts, and skips every file on a re-run.
func TestSyncDirBatchesWrites(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	cs := &countingStore{Store: h.gs, nodeWrites: map[string]int{}}
	eng := New(cs, h.cache, Config{GraphPath: h.eng.cfg.GraphPath}, nil, nil, nil)
	const files = 30
	for i := range files {
		h.write(t, fmt.Sprintf("f%02d.go", i), fmt.Sprintf("package src\nfunc F%02d() {}\n", i))
	}

	sum, err := eng.SyncDir(ctx, h.dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if sum.Scanned != files || sum.Upserted != files || sum.Nodes != 2*files {
		t.Fatalf("summary %+v, want %d scanned and upserted, %d nodes", sum, files, 2*files)
	}
	if cs.calls != 1 {
		t.Errorf("ReplaceSources calls = %d, want 1 batched write", cs.calls)
	}

	sum, err = eng.SyncDir(ctx, h.dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if sum.Unchanged != files || sum.Upserted != 0 {
		t.Errorf("re-run summary %+v, want all %d unchanged", sum, files)
	}
}

// TestSyncDirPruneKeepsUnstatableFiles checks prune deletes only files whose stat says they do not exist, so a
// file hidden by a permission error (an unreadable directory) keeps its nodes.
func TestSyncDirPruneKeepsUnstatableFiles(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	p := h.write(t, "a.go", "package src\nfunc A() {}\n")
	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(p); err != nil {
		t.Fatal(err)
	}
	h.eng.stat = func(string) (fs.FileInfo, error) { return nil, fs.ErrPermission }
	sum, err := h.eng.SyncDir(ctx, h.dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if sum.Deleted != 0 || len(h.nodesForURI(t, model.FileURI(p))) == 0 {
		t.Fatalf("a file whose stat failed with a permission error was pruned (summary %+v)", sum)
	}

	h.eng.stat = os.Stat
	if sum, err = h.eng.SyncDir(ctx, h.dir, true); err != nil || sum.Deleted != 1 {
		t.Fatalf("SyncDir = %+v, %v; want the gone file pruned", sum, err)
	}
}

// TestNoReclusterSkipsTouchedSet checks a NoRecluster engine records no touched IDs for a re-cluster it never runs.
func TestNoReclusterSkipsTouchedSet(t *testing.T) {
	h := newHarness(t, false, false)
	eng := New(h.gs, h.cache, Config{GraphPath: h.eng.cfg.GraphPath, NoRecluster: true}, nil, nil, nil)
	h.write(t, "a.go", "package src\nfunc A() {}\n")
	if _, err := eng.SyncDir(context.Background(), h.dir, true); err != nil {
		t.Fatal(err)
	}
	if touched, dirty := eng.takeTouched(); dirty || len(touched) != 0 {
		t.Errorf("touched = %d IDs, dirty = %v; want none", len(touched), dirty)
	}
}

// symlinkInfo is a FileInfo whose mode marks a symlink, so the check runs on hosts that cannot create one.
type symlinkInfo struct{ fs.FileInfo }

func (symlinkInfo) Mode() fs.FileMode { return fs.ModeSymlink | 0o777 }

// TestSyncFileRefusesNonRegular checks prepare skips whatever Lstat reports as a symlink without reading it.
func TestSyncFileRefusesNonRegular(t *testing.T) {
	h := newHarness(t, false, false)
	p := h.write(t, "notes.md", "SECRET MATERIAL\n")
	h.eng.lstat = func(name string) (fs.FileInfo, error) {
		fi, err := os.Lstat(name)
		if err != nil {
			return nil, err
		}
		return symlinkInfo{fi}, nil
	}
	out, err := h.eng.SyncFile(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Skipped || out.Reason != "not a regular file" {
		t.Errorf("SyncFile = %+v, want skipped as not a regular file", out)
	}
	if n := len(h.nodesForURI(t, model.FileURI(p))); n != 0 {
		t.Errorf("symlink indexed with %d node(s)", n)
	}
}

// TestSyncSkipsSymlinks checks neither a direct sync nor a reconcile reads a real symlink, so a link to a file
// outside the repo (a key, a .env) never reaches the graph.
func TestSyncSkipsSymlinks(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	secret := filepath.Join(t.TempDir(), "id_rsa.md")
	if err := os.WriteFile(secret, []byte("PRIVATE KEY MATERIAL\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(h.dir, "notes.md")
	if err := os.Symlink(secret, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	out, err := h.eng.SyncFile(ctx, link)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Skipped || out.Reason != "not a regular file" {
		t.Errorf("SyncFile(symlink) = %+v, want skipped as not a regular file", out)
	}
	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	if n := len(h.nodesForURI(t, model.FileURI(link))); n != 0 {
		t.Errorf("symlinked file indexed with %d node(s)", n)
	}
}

// TestSyncDirWritesSharedStubsOnce checks a package stub every file references is written once per run.
func TestSyncDirWritesSharedStubsOnce(t *testing.T) {
	h := newHarness(t, false, false)
	cs := &countingStore{Store: h.gs, nodeWrites: map[string]int{}}
	eng := New(cs, h.cache, Config{GraphPath: h.eng.cfg.GraphPath}, nil, nil, nil)
	for _, name := range []string{"A", "B", "C"} {
		h.write(t, name+".go", "package src\nimport \"fmt\"\nfunc "+name+"() { fmt.Println() }\n")
	}
	if _, err := eng.SyncDir(context.Background(), h.dir, true); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{model.NewNodeID("pkg://fmt", "fmt"), model.NewNodeID("pkg://fmt", "Println")} {
		if got := cs.nodeWrites[id]; got != 1 {
			t.Errorf("stub %s written %d times, want 1", id, got)
		}
	}
}

// TestSyncDirPrunesCaseAliases checks a cache entry an older build left under a lower-cased URI is dropped
// once the file is re-indexed under its canonical URI.
func TestSyncDirPrunesCaseAliases(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("path case aliases only arise on case-insensitive Windows paths")
	}
	h := newHarness(t, false, false)
	ctx := context.Background()
	p := h.write(t, "a.go", "package src\nfunc A() {}\n")
	alias := "file://" + filepath.ToSlash(strings.ToLower(p))
	if err := h.gs.UpsertNodes(ctx, []model.Node{{ID: model.NewNodeID(alias, alias), Kind: model.KindFile,
		Label: "a.go", SourceURI: alias}}); err != nil {
		t.Fatal(err)
	}
	h.cache.Mark(alias, "stale")

	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	if n := len(h.nodesForURI(t, alias)); n != 0 {
		t.Errorf("alias %s kept %d node(s)", alias, n)
	}
	if len(h.nodesForURI(t, model.FileURI(p))) == 0 {
		t.Error("the canonical URI should be indexed")
	}
}

// TestRescanAfterOverflow checks the overflow rescan enqueues changed, new, and vanished files, and that
// processing them reconciles the graph.
func TestRescanAfterOverflow(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	pa := h.write(t, "a.go", "package src\nfunc A() {}\n")
	pb := h.write(t, "b.go", "package src\nfunc B() {}\n")
	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	// Changes the watcher dropped: edit a.go, delete b.go, add c.go.
	h.write(t, "a.go", "package src\nfunc A2() {}\n")
	if err := os.Remove(pb); err != nil {
		t.Fatal(err)
	}
	pc := h.write(t, "c.go", "package src\nfunc C() {}\n")

	policy, err := walk.LoadPolicy(h.dir)
	if err != nil {
		t.Fatal(err)
	}
	var queued []string
	if err := h.eng.rescan(ctx, h.dir, policy, func(p string) { queued = append(queued, p) }); err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, p := range queued {
		got[model.FileURI(p)] = true
	}
	for _, want := range []string{pa, pb, pc} {
		if !got[model.FileURI(want)] {
			t.Errorf("rescan did not enqueue %s (queued %v)", want, queued)
		}
	}

	for _, p := range queued {
		if err := h.eng.syncKey(ctx, p); err != nil {
			t.Fatal(err)
		}
	}
	if _, ok := h.nodesForURI(t, model.FileURI(pa))["A2"]; !ok {
		t.Error("edited a.go not re-synced")
	}
	if n := len(h.nodesForURI(t, model.FileURI(pb))); n != 0 {
		t.Errorf("deleted b.go still has %d node(s)", n)
	}
	if len(h.nodesForURI(t, model.FileURI(pc))) == 0 {
		t.Error("new c.go not indexed")
	}
}

// syncFails is a store whose Sync always fails.
type syncFails struct{ store.GraphStore }

func (syncFails) Sync() error { return errors.New("disk gone") }

// TestCacheNotSavedWhenStoreSyncFails checks the cache is never persisted ahead of an unsynced store.
func TestCacheNotSavedWhenStoreSyncFails(t *testing.T) {
	h := newHarness(t, false, false)
	eng := New(syncFails{h.gs}, h.cache, Config{GraphPath: filepath.Join(h.graphDir, "graph")}, nil, nil, nil)
	p := h.write(t, "a.go", "package a\n\nfunc A() {}\n")
	if _, err := eng.SyncChanges(context.Background(), []Change{{Path: p, Op: OpUpsert}}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(h.graphDir, "cache.json")); !os.IsNotExist(err) {
		t.Fatalf("cache.json written although the store sync failed (stat err %v)", err)
	}
}

// TestSyncDirReportsStoreSyncFailure checks the build path (SyncDir) also skips the cache save and returns the
// store sync error rather than reporting success.
func TestSyncDirReportsStoreSyncFailure(t *testing.T) {
	h := newHarness(t, false, false)
	eng := New(syncFails{h.gs}, h.cache, Config{GraphPath: filepath.Join(h.graphDir, "graph"), NoRecluster: true}, nil, nil, nil)
	h.write(t, "a.go", "package a\n\nfunc A() {}\n")
	if _, err := eng.SyncDir(context.Background(), h.dir, true); err == nil || !strings.Contains(err.Error(), "disk gone") {
		t.Fatalf("SyncDir err = %v, want the store sync failure", err)
	}
	if _, err := os.Stat(filepath.Join(h.graphDir, "cache.json")); !os.IsNotExist(err) {
		t.Fatalf("cache.json written although the store sync failed (stat err %v)", err)
	}
}
