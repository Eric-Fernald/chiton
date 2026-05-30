// Package ingest is chiton's live-sync engine (Phase 3). It keeps the graph in
// step with the source as files change, driven by three front-ends that all
// funnel into the same incremental Updater:
//
//   - a debounced fsnotify watcher (Engine.Watch),
//   - git post-commit / post-checkout hooks (Engine.SyncChanges over a git diff),
//   - one-shot directory reconciliation (Engine.SyncDir).
//
// Per changed file the engine: re-extracts AST nodes/edges, deletes the file's
// stale nodes (DeleteBySourceURI — correct because chiton emits no cross-file
// structural edges into a file's symbols), upserts the fresh ones, and
// optionally re-embeds and re-runs deterministic file rules. After a batch
// settles it incrementally re-clusters the touched 2-hop neighborhood (see
// cluster.RunIncremental), updating the sidecar without a full pass.
package ingest

import (
	"context"
	"crypto/sha256"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/cache"
	"github.com/Eric-Fernald/chiton/internal/embed"
	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/ingest/fswatch"
	gitpkg "github.com/Eric-Fernald/chiton/internal/ingest/git"
	"github.com/Eric-Fernald/chiton/internal/jobs"
)

// Op is the kind of change applied to a file.
type Op int

const (
	OpUpsert Op = iota
	OpDelete
)

// Change is one file transition handed to SyncChanges. Path is absolute.
type Change struct {
	Path string
	Op   Op
}

// Logger is a minimal printf-style sink; nil disables logging.
type Logger func(format string, args ...any)

// Config tunes the engine. Zero values get sane defaults via withDefaults.
type Config struct {
	GraphPath       string        // for locating the cluster sidecar
	Resolution      float64       // Leiden resolution for incremental re-cluster
	Reembed         bool          // re-embed a changed file's embeddable nodes
	Reanalyze       bool          // re-run deterministic FileRules on change
	Workers         int           // watch-mode worker concurrency
	Debounce        time.Duration // per-file debounce in watch mode
	ClusterDebounce time.Duration // idle window before an incremental re-cluster
}

func (c Config) withDefaults() Config {
	if c.Resolution <= 0 {
		c.Resolution = 1.0
	}
	if c.Workers <= 0 {
		c.Workers = 4
	}
	if c.Debounce <= 0 {
		c.Debounce = 300 * time.Millisecond
	}
	if c.ClusterDebounce <= 0 {
		c.ClusterDebounce = 3 * time.Second
	}
	return c
}

// Engine performs incremental graph updates. It is safe for concurrent use:
// store writes are individually transactional, the cache is mutex-guarded, and
// the cluster pass is serialised by clusterMu.
type Engine struct {
	store    store.GraphStore
	cache    *cache.Cache
	embedder embed.Embedder    // nil ⇒ no re-embed
	analyzer *analyze.Analyzer // nil ⇒ no re-analyze
	cfg      Config
	log      Logger

	clusterMu sync.Mutex

	touchedMu sync.Mutex
	touched   map[string]bool
	dirty     bool
}

// New builds an Engine. embedder and analyzer may be nil to disable re-embed /
// re-analyze regardless of the Config flags.
func New(s store.GraphStore, c *cache.Cache, cfg Config, embedder embed.Embedder, analyzer *analyze.Analyzer, log Logger) *Engine {
	if log == nil {
		log = func(string, ...any) {}
	}
	return &Engine{
		store:    s,
		cache:    c,
		embedder: embedder,
		analyzer: analyzer,
		cfg:      cfg.withDefaults(),
		log:      log,
		touched:  map[string]bool{},
	}
}

// Outcome reports what one file sync did.
type Outcome struct {
	Path          string
	Op            Op
	Skipped       bool
	Reason        string
	NodesUpserted int
	EdgesUpserted int
	NodesDeleted  int
	Reembedded    int
	TouchedIDs    []string
}

// BatchSummary aggregates a SyncChanges / SyncDir run.
type BatchSummary struct {
	Upserted    int
	Deleted     int
	Skipped     int
	Reembedded  int
	Reclustered bool
}

// SyncFile (re-)extracts a single file into the graph. A vanished file is
// delegated to DeleteFile. Read/parse failures are reported as skips (not
// errors) so a watcher never dies on one bad file; only store failures error.
func (e *Engine) SyncFile(ctx context.Context, path string) (Outcome, error) {
	ex := extast.ForFile(path)
	if ex == nil {
		return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: "unsupported file type"}, nil
	}
	content, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return e.DeleteFile(ctx, path)
		}
		return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: "read: " + err.Error()}, nil
	}

	uri := uriFor(path)
	sum := sha256.Sum256(content)
	hexSum := fmt.Sprintf("%x", sum[:])
	if !e.cache.Changed(uri, hexSum) {
		return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: "unchanged"}, nil
	}

	nodes, edges, err := ex.ExtractFile(ctx, path)
	if err != nil {
		return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: "extract: " + err.Error()}, nil
	}

	deleted, err := e.store.DeleteBySourceURI(ctx, uri)
	if err != nil {
		return Outcome{}, fmt.Errorf("delete stale nodes for %s: %w", uri, err)
	}
	if err := e.store.UpsertNodes(ctx, nodes); err != nil {
		return Outcome{}, fmt.Errorf("upsert nodes for %s: %w", uri, err)
	}
	if err := e.store.UpsertEdges(ctx, edges); err != nil {
		return Outcome{}, fmt.Errorf("upsert edges for %s: %w", uri, err)
	}
	e.cache.Mark(uri, hexSum)

	ids := make([]string, 0, len(nodes))
	for _, n := range nodes {
		ids = append(ids, n.ID)
	}

	out := Outcome{
		Path:          path,
		Op:            OpUpsert,
		NodesUpserted: len(nodes),
		EdgesUpserted: len(edges),
		NodesDeleted:  deleted,
		TouchedIDs:    ids,
	}

	if e.cfg.Reembed && e.embedder != nil {
		n, err := e.reembed(ctx, nodes)
		if err != nil {
			e.log("sync: %s: re-embed failed: %v", path, err)
		}
		out.Reembedded = n
	}
	if e.cfg.Reanalyze && e.analyzer != nil {
		if _, err := e.analyzer.AnalyzeFile(ctx, path); err != nil {
			e.log("sync: %s: re-analyze failed: %v", path, err)
		}
	}

	e.markTouched(ids)
	return out, nil
}

// DeleteFile removes a file's nodes/edges (and its findings) from the graph.
// Idempotent: deleting an already-absent file is a no-op.
func (e *Engine) DeleteFile(ctx context.Context, path string) (Outcome, error) {
	return e.deleteURI(ctx, uriFor(path))
}

// SyncChange dispatches one Change to SyncFile or DeleteFile.
func (e *Engine) SyncChange(ctx context.Context, ch Change) (Outcome, error) {
	if ch.Op == OpDelete {
		return e.DeleteFile(ctx, ch.Path)
	}
	return e.SyncFile(ctx, ch.Path)
}

// SyncChanges applies a batch of changes synchronously, then incrementally
// re-clusters and flushes the cache. Used by the git hooks and `sync git`.
func (e *Engine) SyncChanges(ctx context.Context, changes []Change) (*BatchSummary, error) {
	// Persist the cache even if a later change errors, so files already written
	// to the store are not needlessly re-extracted on the next run.
	defer e.saveCache()
	sum := &BatchSummary{}
	for _, ch := range changes {
		out, err := e.SyncChange(ctx, ch)
		if err != nil {
			return sum, err
		}
		e.tally(sum, out)
	}
	e.flush(ctx, sum)
	return sum, nil
}

// SyncDir reconciles a whole directory tree in one shot: every supported file
// is (re-)synced, and — when prune is true — files previously indexed under the
// tree that are now gone are deleted. Then it re-clusters and flushes.
func (e *Engine) SyncDir(ctx context.Context, root string, prune bool) (*BatchSummary, error) {
	defer e.saveCache()
	sum := &BatchSummary{}
	seen := map[string]bool{}

	walkErr := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if path != root && DefaultIgnoreDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if extast.ForFile(path) == nil {
			return nil
		}
		out, err := e.SyncFile(ctx, path)
		if err != nil {
			return err
		}
		seen[uriFor(path)] = true
		e.tally(sum, out)
		return nil
	})
	if walkErr != nil {
		return sum, walkErr
	}

	if prune {
		rootURI := uriFor(root)
		for _, uri := range e.cache.URIs() {
			if uri != rootURI && !strings.HasPrefix(uri, rootURI+"/") {
				continue
			}
			if seen[uri] {
				continue
			}
			// Only prune URIs whose file is genuinely gone from disk. A file that
			// still exists but was excluded this walk (ignored dir, unsupported
			// type, narrower root) must NOT be deleted — that would make prune a
			// surprising, destructive default.
			if p := pathForURI(uri); p != "" {
				if _, err := os.Stat(p); err == nil {
					continue
				}
			}
			out, err := e.deleteURI(ctx, uri)
			if err != nil {
				return sum, err
			}
			e.tally(sum, out)
		}
	}

	e.flush(ctx, sum)
	return sum, nil
}

// Watch starts a debounced fsnotify watcher over root, re-syncing changed files
// and incrementally re-clustering once activity settles. It blocks until ctx is
// cancelled, then drains in-flight work, runs a final re-cluster, and flushes.
//
// The queues are started on context.Background(), NOT the watch ctx, so that
// cancelling the watch (Ctrl-C) stops the fsnotify loop without aborting the
// queues mid-flight — Drain() then processes files edited just before shutdown
// rather than dropping them.
func (e *Engine) Watch(ctx context.Context, root string) error {
	// A single-worker, longer-debounced queue serialises cluster passes.
	clusterQ := jobs.New(func(c context.Context, _ string) error {
		return e.Recluster(c)
	}, jobs.Options{Workers: 1, Debounce: e.cfg.ClusterDebounce})
	clusterQ.Start(context.Background())

	fileQ := jobs.New(e.syncKey, jobs.Options{
		Workers:  e.cfg.Workers,
		Debounce: e.cfg.Debounce,
		OnError: func(key string, err error) {
			e.log("sync: %s: %v", key, err)
		},
		OnDone: func(key string, _ error) {
			clusterQ.Enqueue("cluster")
		},
	})
	fileQ.Start(context.Background())

	w, err := fswatch.New(root, fileQ.Enqueue, fswatch.Options{
		IgnoreDir: DefaultIgnoreDir,
		Accept:    func(p string) bool { return extast.ForFile(p) != nil },
		OnError:   func(err error) { e.log("watch: %v", err) },
		// A renamed/removed directory yields no per-child events, so prune the
		// graph for every file previously indexed under it.
		OnDirRemove: func(dir string) {
			if n := e.pruneDir(context.Background(), dir); n > 0 {
				e.log("watch: directory %s removed — pruned %d file(s)", dir, n)
				clusterQ.Enqueue("cluster")
			}
		},
	})
	if err != nil {
		fileQ.Stop()
		clusterQ.Stop()
		return err
	}

	e.log("watch: watching %s (%d dirs) — Ctrl-C to stop", root, w.Watched())
	runErr := w.Run(ctx) // blocks until ctx cancelled
	_ = w.Close()

	// Graceful shutdown: drain queued file work first (its OnDone re-arms the
	// cluster trigger), then drain the cluster queue, then a final flush.
	fileQ.Drain()
	clusterQ.Drain()

	flushCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := e.Recluster(flushCtx); err != nil {
		e.log("watch: final re-cluster failed: %v", err)
	}
	if err := e.cache.Save(); err != nil {
		e.log("watch: cache save failed: %v", err)
	}

	if runErr != nil && runErr != context.Canceled {
		return runErr
	}
	return nil
}

// pruneDir removes every file previously indexed under dir (used when a watched
// directory is renamed or deleted, which yields no per-child fs events). Returns
// the number of files pruned.
func (e *Engine) pruneDir(ctx context.Context, dir string) int {
	prefix := uriFor(dir)
	pruned := 0
	for _, uri := range e.cache.URIs() {
		if uri == prefix || strings.HasPrefix(uri, prefix+"/") {
			if _, err := e.deleteURI(ctx, uri); err != nil {
				e.log("sync: prune %s failed: %v", uri, err)
				continue
			}
			pruned++
		}
	}
	return pruned
}

// syncKey is the watch-queue handler: it stats the path to decide upsert vs
// delete, so a create-then-delete burst collapses to the right final action.
func (e *Engine) syncKey(ctx context.Context, path string) error {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			_, derr := e.DeleteFile(ctx, path)
			return derr
		}
		return err
	}
	_, err := e.SyncFile(ctx, path)
	return err
}

// Recluster runs one incremental cluster pass over the accumulated touched set,
// updating the sidecar in place. It is a no-op when nothing changed or when no
// sidecar exists yet (run `chiton cluster` once to seed it). Serialised by
// clusterMu so concurrent triggers can't corrupt the sidecar.
func (e *Engine) Recluster(ctx context.Context) error {
	e.clusterMu.Lock()
	defer e.clusterMu.Unlock()

	touched, dirty := e.takeTouched()
	if !dirty {
		return nil
	}

	sidecarPath := cluster.SidecarPath(e.cfg.GraphPath)
	prev, err := cluster.LoadSidecar(sidecarPath)
	if err != nil {
		return fmt.Errorf("load cluster sidecar: %w", err)
	}
	if prev == nil {
		// No prior full clustering — don't surprise the user with a full pass
		// during sync. `chiton cluster` seeds the sidecar; after that, sync
		// keeps it incrementally fresh.
		e.log("sync: skipping re-cluster (no cluster.json yet — run 'chiton cluster' once)")
		return nil
	}

	nodes, err := e.store.AllNodes(ctx)
	if err != nil {
		return fmt.Errorf("load nodes for re-cluster: %w", err)
	}
	edges, err := e.store.AllEdges(ctx)
	if err != nil {
		return fmt.Errorf("load edges for re-cluster: %w", err)
	}

	rng := rand.New(rand.NewSource(42))
	res := cluster.RunIncremental(prev, nodes, edges, touched, e.cfg.Resolution, rng)
	if err := cluster.WriteSidecar(sidecarPath, res); err != nil {
		return err
	}
	e.log("sync: re-clustered %d touched node(s) → %d communities over %d nodes",
		len(touched), res.CommunityCount, res.NodeCount)
	return nil
}

// reembed embeds the embeddable nodes among the given set and upserts vectors.
func (e *Engine) reembed(ctx context.Context, nodes []model.Node) (int, error) {
	es, ok := e.store.(store.EmbeddingStore)
	if !ok {
		return 0, nil
	}
	var work []model.Node
	var texts []string
	for _, n := range nodes {
		if embed.Embeddable(n) {
			work = append(work, n)
			texts = append(texts, embed.TextFor(n))
		}
	}
	if len(work) == 0 {
		return 0, nil
	}
	vecs, err := e.embedder.Embed(ctx, texts)
	if err != nil {
		return 0, err
	}
	embs := make([]store.Embedding, 0, len(work))
	for i, n := range work {
		if i < len(vecs) {
			embs = append(embs, store.Embedding{NodeID: n.ID, Vec: vecs[i]})
		}
	}
	if err := es.UpsertEmbeddings(ctx, embs); err != nil {
		return 0, err
	}
	return len(embs), nil
}

// deleteURI deletes by a known source_uri. It records the deleted file's
// surviving neighbors as touched (so the incremental re-cluster refreshes
// communities whose connectivity changed) BEFORE the edges vanish, then removes
// the nodes/edges, cache entry, and all findings for the file.
func (e *Engine) deleteURI(ctx context.Context, uri string) (Outcome, error) {
	e.markSurvivorsTouched(ctx, uri)
	deleted, err := e.store.DeleteBySourceURI(ctx, uri)
	if err != nil {
		return Outcome{}, fmt.Errorf("delete %s: %w", uri, err)
	}
	e.cache.Delete(uri)
	if e.analyzer != nil {
		if _, err := e.analyzer.PruneFileURI(ctx, uri); err != nil {
			e.log("sync: prune findings for %s failed: %v", uri, err)
		}
	}
	return Outcome{Path: uri, Op: OpDelete, NodesDeleted: deleted}, nil
}

// markSurvivorsTouched records a file's nodes and their ≤2-hop neighbors as
// touched before a delete. Cross-file structural edges only ever point OUT to
// shared package nodes, so those shared neighbors are exactly the survivors
// whose community could shift when this file's edges disappear. Marking them
// (and harmlessly the soon-deleted nodes, which RunIncremental ignores once
// absent) keeps the incremental re-cluster honest. Always sets the dirty flag.
func (e *Engine) markSurvivorsTouched(ctx context.Context, uri string) {
	fileID := model.NewNodeID(uri, uri)
	sub, err := e.store.Neighbors(ctx, fileID, 2, nil)
	if err != nil {
		e.markDirty()
		return
	}
	ids := make([]string, 0, len(sub.Nodes))
	for _, n := range sub.Nodes {
		ids = append(ids, n.ID)
	}
	e.markTouched(ids) // sets dirty even when ids is empty
}

// flush re-clusters after a batch. The cache is persisted separately by a
// deferred saveCache in the caller, so it is saved even when a change errors.
func (e *Engine) flush(ctx context.Context, sum *BatchSummary) {
	if err := e.Recluster(ctx); err != nil {
		e.log("sync: re-cluster failed: %v", err)
		return
	}
	sidecar := cluster.SidecarPath(e.cfg.GraphPath)
	if prev, _ := cluster.LoadSidecar(sidecar); prev != nil {
		sum.Reclustered = true
	}
}

// saveCache persists the content cache, logging (not returning) any error so it
// is safe to use in a deferred best-effort save.
func (e *Engine) saveCache() {
	if err := e.cache.Save(); err != nil {
		e.log("sync: cache save failed: %v", err)
	}
}

// pathForURI converts a file:// source URI back to an on-disk path, or "" if it
// is not a file URI.
func pathForURI(uri string) string {
	const p = "file://"
	if !strings.HasPrefix(uri, p) {
		return ""
	}
	return filepath.FromSlash(strings.TrimPrefix(uri, p))
}

func (e *Engine) tally(sum *BatchSummary, out Outcome) {
	switch {
	case out.Skipped:
		sum.Skipped++
	case out.Op == OpDelete:
		sum.Deleted++
	default:
		sum.Upserted++
	}
	sum.Reembedded += out.Reembedded
}

func (e *Engine) markTouched(ids []string) {
	e.touchedMu.Lock()
	defer e.touchedMu.Unlock()
	for _, id := range ids {
		e.touched[id] = true
	}
	e.dirty = true
}

func (e *Engine) markDirty() {
	e.touchedMu.Lock()
	defer e.touchedMu.Unlock()
	e.dirty = true
}

// takeTouched atomically returns and resets the accumulated touched set and
// dirty flag.
func (e *Engine) takeTouched() (map[string]bool, bool) {
	e.touchedMu.Lock()
	defer e.touchedMu.Unlock()
	if !e.dirty {
		return nil, false
	}
	t := e.touched
	e.touched = map[string]bool{}
	e.dirty = false
	return t, true
}

// uriFor returns the canonical file:// URI for a path, matching the AST
// extractors' convention exactly.
func uriFor(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		abs = path
	}
	return "file://" + filepath.ToSlash(abs)
}

// DefaultIgnoreDir reports whether a directory base name should be skipped by
// the watcher and one-shot reconciler. Includes the chiton output dir so the
// store/cache writes never feed back into the watcher.
func DefaultIgnoreDir(name string) bool {
	switch name {
	case ".git", ".chiton", "chiton-out", "node_modules", "vendor", ".hg", ".svn":
		return true
	}
	return false
}

// GitChanges adapts git-package changes (from a diff) into engine Changes.
func GitChanges(in []gitpkg.Change) []Change {
	out := make([]Change, 0, len(in))
	for _, c := range in {
		op := OpUpsert
		if c.Op == gitpkg.OpDelete {
			op = OpDelete
		}
		out = append(out, Change{Path: c.Path, Op: op})
	}
	return out
}
