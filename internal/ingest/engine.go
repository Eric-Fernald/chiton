// Package ingest is chiton's live-sync engine. It re-extracts changed files and replaces their nodes by diff
// (store.ReplaceSources), so edges other sources attached to surviving symbols stay.
package ingest

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"math/rand"
	"os"
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
	"github.com/Eric-Fernald/chiton/internal/walk"
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

// Engine performs incremental graph updates. It is safe for concurrent use because each store write is
// transactional, the cache is mutex-guarded, and clusterMu serialises the cluster pass.
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
	Unreadable  int // walk entries that could not be read, logged and skipped (SyncDir only)
	Reembedded  int
	Reclustered bool
}

// SyncFile (re-)extracts a single file into the graph and hands a vanished file to DeleteFile.
// Read and parse failures are skips, not errors, so a watcher never stops on one bad file.
func (e *Engine) SyncFile(ctx context.Context, path string) (Outcome, error) {
	ex := extast.ForFile(path)
	if ex == nil {
		return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: "unsupported file type"}, nil
	}
	uri := model.FileURI(path)
	content, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return e.deleteURI(ctx, uri)
		}
		return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: "read: " + err.Error()}, nil
	}

	hexSum := model.ContentSHA(content)
	if !e.cache.Changed(uri, hexSum) {
		return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: "unchanged"}, nil
	}

	// Extract the same bytes that were hashed, so the cached hash always matches the extracted content.
	nodes, edges, err := ex.ExtractSource(ctx, path, content)
	if err != nil {
		return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: "extract: " + err.Error()}, nil
	}

	// Read the file's findings while the AFFECTS edges into its nodes still exist; the replace drops those on
	// nodes the new extraction lacks.
	var prior *analyze.PriorFindings
	if e.cfg.Reanalyze && e.analyzer != nil {
		if prior, err = e.analyzer.ReadPriorFindings(ctx, uri); err != nil {
			e.log("sync: %s: %v", path, err) // a nil prior makes the re-analyze query the store instead
		}
	}
	deleted, err := store.ReplaceSources(ctx, e.store, store.SourceUpdate{URI: uri, Nodes: nodes, Edges: edges})
	if err != nil {
		return Outcome{}, fmt.Errorf("replace %s: %w", uri, err)
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
		NodesDeleted:  deleted[0],
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
		if _, err := e.analyzer.ReanalyzeExtracted(ctx, path, content, nodes, edges, prior); err != nil {
			e.log("sync: %s: re-analyze failed: %v", path, err)
		}
	}

	e.markTouched(ids)
	return out, nil
}

// DeleteFile removes a file's nodes/edges (and its findings) from the graph.
// Idempotent: deleting an already-absent file is a no-op.
func (e *Engine) DeleteFile(ctx context.Context, path string) (Outcome, error) {
	return e.deleteURI(ctx, model.FileURI(path))
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
		if err := ctx.Err(); err != nil {
			return sum, err
		}
		out, err := e.SyncChange(ctx, ch)
		if err != nil {
			return sum, err
		}
		e.tally(sum, out)
	}
	e.flush(ctx, sum)
	return sum, nil
}

// SyncDir reconciles a directory tree in one pass. It (re-)syncs every supported file the walk policy keeps
// and, when prune is true, deletes indexed files under the tree that are gone from disk.
func (e *Engine) SyncDir(ctx context.Context, root string, prune bool) (*BatchSummary, error) {
	defer e.saveCache()
	sum := &BatchSummary{}
	seen := map[string]bool{}
	policy, err := walk.LoadPolicy(root)
	if err != nil {
		return sum, err
	}

	stats, walkErr := walk.Files(ctx, root, policy, func(path string) error {
		if extast.ForFile(path) == nil {
			return nil
		}
		out, err := e.SyncFile(ctx, path)
		if err != nil {
			return err
		}
		seen[model.FileURI(path)] = true
		e.tally(sum, out)
		return nil
	}, func(path string, err error) { e.log("sync: skip %s: %v", path, err) })
	sum.Unreadable = stats.Unreadable
	if walkErr != nil {
		return sum, walkErr
	}

	if prune {
		rootURI := model.FileURI(root)
		for _, uri := range e.cache.URIs() {
			if err := ctx.Err(); err != nil {
				return sum, err
			}
			if uri != rootURI && !strings.HasPrefix(uri, rootURI+"/") {
				continue
			}
			if seen[uri] {
				continue
			}
			// Prune only files that are gone from disk. A file this walk excluded (ignored dir, unsupported
			// type, or narrower root) must survive.
			if p, ok := model.PathFromFileURI(uri); ok {
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

// Watch re-syncs changed files under root until ctx is cancelled, then drains in-flight work and re-clusters.
// Queues run on context.Background() so a cancel stops fsnotify but Drain still processes queued edits.
func (e *Engine) Watch(ctx context.Context, root string) error {
	policy, err := walk.LoadPolicy(root)
	if err != nil {
		return err
	}
	// A single-worker, longer-debounced queue serialises cluster passes.
	clusterQ := jobs.New(func(c context.Context, _ string) error {
		_, err := e.recluster(c)
		return err
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
		Skip:    policy.Skip,
		Accept:  func(p string) bool { return extast.ForFile(p) != nil },
		OnError: func(err error) { e.log("watch: %v", err) },
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
	if _, err := e.recluster(flushCtx); err != nil {
		e.log("watch: final re-cluster failed: %v", err)
	}
	if err := e.cache.Save(); err != nil {
		e.log("watch: cache save failed: %v", err)
	}

	if runErr != nil && !errors.Is(runErr, context.Canceled) {
		return runErr
	}
	return nil
}

// pruneDir removes every file indexed under dir (a renamed or deleted directory yields no per-child
// events) and returns how many it pruned.
func (e *Engine) pruneDir(ctx context.Context, dir string) int {
	prefix := model.FileURI(dir)
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
		if errors.Is(err, fs.ErrNotExist) {
			_, derr := e.DeleteFile(ctx, path)
			return derr
		}
		return err
	}
	_, err := e.SyncFile(ctx, path)
	return err
}

// recluster runs one incremental cluster pass over the touched set under clusterMu, unless nothing changed or
// no sidecar exists yet (`chiton cluster` seeds it). haveSidecar reports whether a sidecar exists afterwards.
func (e *Engine) recluster(ctx context.Context) (haveSidecar bool, err error) {
	e.clusterMu.Lock()
	defer e.clusterMu.Unlock()

	sidecarPath := cluster.SidecarPath(e.cfg.GraphPath)
	touched, dirty := e.takeTouched()
	if !dirty {
		_, statErr := os.Stat(sidecarPath)
		return statErr == nil, nil
	}

	prev, err := cluster.LoadSidecar(sidecarPath)
	if err != nil {
		return false, fmt.Errorf("load cluster sidecar: %w", err)
	}
	if prev == nil {
		// With no prior full clustering, don't surprise the user with a full pass during sync.
		e.log("sync: skipping re-cluster (no cluster.json yet — run 'chiton cluster' once)")
		return false, nil
	}

	nodes, err := e.store.AllNodes(ctx)
	if err != nil {
		return false, fmt.Errorf("load nodes for re-cluster: %w", err)
	}
	edges, err := e.store.AllEdges(ctx)
	if err != nil {
		return false, fmt.Errorf("load edges for re-cluster: %w", err)
	}

	rng := rand.New(rand.NewSource(42))
	res := cluster.RunIncremental(prev, nodes, edges, touched, e.cfg.Resolution, rng)
	if err := cluster.WriteSidecar(sidecarPath, res); err != nil {
		return false, err
	}
	e.log("sync: re-clustered %d touched node(s) → %d communities over %d nodes",
		len(touched), res.CommunityCount, res.NodeCount)
	return true, nil
}

// reembed embeds the embeddable nodes among the given set and upserts vectors.
func (e *Engine) reembed(ctx context.Context, nodes []model.Node) (int, error) {
	es, ok := e.store.(store.EmbeddingStore)
	if !ok {
		return 0, nil
	}
	embs, _, err := embed.NodeEmbeddings(ctx, e.embedder, nodes)
	if err != nil || len(embs) == 0 {
		return 0, err
	}
	if err := es.UpsertEmbeddings(ctx, embs); err != nil {
		return 0, err
	}
	return len(embs), nil
}

// deleteURI removes a source_uri's nodes, edges, cache entry, and findings, first marking its surviving
// neighbors touched (before the edges vanish) so the re-cluster refreshes their communities.
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

// markSurvivorsTouched marks a file's nodes and their neighbors within 2 hops touched, and always sets dirty.
// Cross-file edges only point out to shared package nodes, which are the survivors whose community may shift.
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
	haveSidecar, err := e.recluster(ctx)
	if err != nil {
		e.log("sync: re-cluster failed: %v", err)
		return
	}
	sum.Reclustered = haveSidecar
}

// saveCache persists the content cache, logging (not returning) any error so it
// is safe to use in a deferred best-effort save.
func (e *Engine) saveCache() {
	if err := e.cache.Save(); err != nil {
		e.log("sync: cache save failed: %v", err)
	}
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
