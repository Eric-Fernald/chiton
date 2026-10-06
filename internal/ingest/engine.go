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
	"runtime"
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
	NoRecluster     bool          // skip the post-batch re-cluster (build leaves clustering to 'chiton cluster')
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
	stat     func(string) (fs.FileInfo, error) // os.Stat; prune's existence check, swappable in tests

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
	// SyncDir's walker and writer, and Watch's workers, log concurrently, so serialise the sink.
	var logMu sync.Mutex
	sink := log
	log = func(format string, args ...any) {
		logMu.Lock()
		defer logMu.Unlock()
		sink(format, args...)
	}
	return &Engine{
		store:    s,
		cache:    c,
		embedder: embedder,
		analyzer: analyzer,
		cfg:      cfg.withDefaults(),
		log:      log,
		stat:     os.Stat,
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
	Scanned     int // supported files the walk found (SyncDir only)
	Upserted    int
	Deleted     int
	Skipped     int // unchanged plus unreadable or unparsable files
	Unchanged   int // files the content cache skipped
	Unreadable  int // walk entries that could not be read, logged and skipped (SyncDir only)
	Nodes       int // nodes written
	Edges       int // edges written
	Reembedded  int
	Reclustered bool
}

const (
	syncQueue       = 256    // buffered paths and results between SyncDir's walker, pool, and writer
	batchRecords    = 20_000 // nodes plus edges per batched ReplaceSources call
	reasonUnchanged = "unchanged"
)

// prepared is one file read, hashed, and extracted off the writer goroutine.
type prepared struct {
	path, uri, sha string
	content        []byte // kept only when re-analyze needs it
	nodes          []model.Node
	edges          []model.Edge
	prior          *analyze.PriorFindings // read by apply just before the replace; nil makes re-analyze query the store
	gone           bool                   // the file vanished, so the writer deletes its URI
	out            Outcome                // Skipped when there is nothing to write
}

func skipped(path, reason string) Outcome {
	return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: reason}
}

// prepare reads, hashes, and extracts path without touching the store, so a pool can run it concurrently.
// Read and parse failures come back as skips.
func (e *Engine) prepare(ctx context.Context, path string) prepared {
	p := prepared{path: path, uri: model.FileURI(path)}
	ex := extast.ForFile(path)
	if ex == nil {
		p.out = skipped(path, "unsupported file type")
		return p
	}
	content, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			p.gone = true
		} else {
			p.out = skipped(path, "read: "+err.Error())
		}
		return p
	}
	p.sha = model.ContentSHA(content)
	if !e.cache.Changed(p.uri, p.sha) {
		p.out = skipped(path, reasonUnchanged)
		return p
	}
	// Extract the same bytes that were hashed, so the cached hash always matches the extracted content.
	if p.nodes, p.edges, err = ex.ExtractSource(ctx, path, content); err != nil {
		p.nodes, p.edges = nil, nil
		p.out = skipped(path, "extract: "+err.Error())
		return p
	}
	if e.reanalyzing() {
		p.content = content
	}
	return p
}

func (e *Engine) reanalyzing() bool { return e.cfg.Reanalyze && e.analyzer != nil }

// apply writes a batch of extracted files (distinct URIs) in one ReplaceSources call, then marks the cache and
// runs each file's follow-ups. Every extraction includes its file node, so a file whose node is absent is Fresh.
func (e *Engine) apply(ctx context.Context, batch []prepared) ([]Outcome, error) {
	fileIDs := make([]string, len(batch))
	for i, p := range batch {
		fileIDs[i] = model.NewNodeID(p.uri, p.uri)
	}
	existing, err := store.GetNodes(ctx, e.store, fileIDs)
	if err != nil {
		return nil, fmt.Errorf("look up file nodes: %w", err)
	}
	indexed := make(map[string]bool, len(existing))
	for _, n := range existing {
		indexed[n.ID] = true
	}
	updates := make([]store.SourceUpdate, len(batch))
	for i, p := range batch {
		updates[i] = store.SourceUpdate{URI: p.uri, Nodes: p.nodes, Edges: p.edges, Fresh: !indexed[fileIDs[i]]}
	}
	// Read each file's findings while the AFFECTS edges into its nodes still exist; the replace drops those on
	// nodes the new extraction lacks.
	if e.reanalyzing() {
		for i := range batch {
			if batch[i].prior, err = e.analyzer.ReadPriorFindings(ctx, batch[i].uri); err != nil {
				e.log("sync: %s: %v", batch[i].path, err)
			}
		}
	}
	deleted, err := store.ReplaceSources(ctx, e.store, updates...)
	if err != nil {
		return nil, fmt.Errorf("replace %d file(s): %w", len(batch), err)
	}
	outs := make([]Outcome, len(batch))
	for i, p := range batch {
		e.cache.Mark(p.uri, p.sha)
		outs[i] = e.finish(ctx, p, deleted[i])
	}
	return outs, nil
}

// finish re-embeds and re-analyzes a written file when configured and reports its outcome.
func (e *Engine) finish(ctx context.Context, p prepared, deleted int) Outcome {
	ids := make([]string, 0, len(p.nodes))
	for _, n := range p.nodes {
		ids = append(ids, n.ID)
	}
	out := Outcome{Path: p.path, Op: OpUpsert, NodesUpserted: len(p.nodes), EdgesUpserted: len(p.edges),
		NodesDeleted: deleted, TouchedIDs: ids}
	if e.cfg.Reembed && e.embedder != nil {
		n, err := e.reembed(ctx, p.nodes)
		if err != nil {
			e.log("sync: %s: re-embed failed: %v", p.path, err)
		}
		out.Reembedded = n
	}
	if e.reanalyzing() {
		if _, err := e.analyzer.ReanalyzeExtracted(ctx, p.path, p.content, p.nodes, p.edges, p.prior); err != nil {
			e.log("sync: %s: re-analyze failed: %v", p.path, err)
		}
	}
	if !e.cfg.NoRecluster {
		e.markTouched(ids)
	}
	return out
}

// SyncFile (re-)extracts a single file into the graph and hands a vanished file to DeleteFile.
// Read and parse failures are skips, not errors, so a watcher never stops on one bad file.
func (e *Engine) SyncFile(ctx context.Context, path string) (Outcome, error) {
	p := e.prepare(ctx, path)
	switch {
	case p.gone:
		return e.deleteURI(ctx, p.uri)
	case p.out.Skipped:
		return p.out, nil
	}
	outs, err := e.apply(ctx, []prepared{p})
	if err != nil {
		return Outcome{}, err
	}
	return outs[0], nil
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

// SyncDir reconciles a directory tree: GOMAXPROCS workers read and extract each supported file the walk policy
// keeps, this goroutine writes results in batches, and prune deletes indexed files gone from disk.
func (e *Engine) SyncDir(ctx context.Context, root string, prune bool) (sum *BatchSummary, err error) {
	sum = &BatchSummary{}
	defer func() {
		if serr := e.persistCache(); serr != nil && err == nil {
			err = serr
		}
	}()
	policy, err := walk.LoadPolicy(root)
	if err != nil {
		return sum, err
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Only the walker writes seen, scanned, stats, and walkErr; closing paths and then results orders every
	// write before the reads below.
	seen := map[string]bool{}
	var scanned int
	var stats walk.Stats
	var walkErr error
	paths := make(chan string, syncQueue)
	go func() {
		defer close(paths)
		stats, walkErr = walk.Files(ctx, root, policy, func(path string) error {
			if extast.ForFile(path) == nil {
				return nil
			}
			seen[model.FileURI(path)] = true
			scanned++
			select {
			case paths <- path:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}, func(path string, err error) { e.log("sync: skip %s: %v", path, err) })
	}()

	results := make(chan prepared, syncQueue)
	var wg sync.WaitGroup
	for range runtime.GOMAXPROCS(0) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range paths {
				if ctx.Err() == nil {
					results <- e.prepare(ctx, path)
				}
			}
		}()
	}
	go func() {
		wg.Wait()
		close(results)
	}()

	if err := e.writeAll(ctx, results, sum); err != nil {
		cancel()
		for range results {
		}
		return sum, err
	}
	sum.Scanned, sum.Unreadable = scanned, stats.Unreadable
	if walkErr != nil {
		return sum, walkErr
	}
	if prune {
		if err := e.prune(ctx, root, seen, sum); err != nil {
			return sum, err
		}
	}
	e.flush(ctx, sum)
	return sum, nil
}

// writeAll applies prepared files in batches of about batchRecords nodes and edges, deleting vanished files and
// tallying skips as they arrive.
func (e *Engine) writeAll(ctx context.Context, results <-chan prepared, sum *BatchSummary) error {
	var batch []prepared
	records := 0
	writeBatch := func() error {
		if len(batch) == 0 {
			return nil
		}
		outs, err := e.apply(ctx, batch)
		if err != nil {
			return err
		}
		for _, out := range outs {
			e.tally(sum, out)
		}
		batch, records = nil, 0
		return nil
	}
	for p := range results {
		switch {
		case p.gone:
			out, err := e.deleteURI(ctx, p.uri)
			if err != nil {
				return err
			}
			e.tally(sum, out)
		case p.out.Skipped:
			e.logSkip(p.out)
			e.tally(sum, p.out)
		default:
			batch = append(batch, p)
			if records += len(p.nodes) + len(p.edges); records >= batchRecords {
				if err := writeBatch(); err != nil {
					return err
				}
			}
		}
	}
	return writeBatch()
}

// logSkip reports a file skipped for any reason but an unchanged hash.
func (e *Engine) logSkip(out Outcome) {
	if out.Reason != reasonUnchanged {
		e.log("sync: skip %s: %s", out.Path, out.Reason)
	}
}

// prune deletes indexed files under root that the walk did not see and that are gone from disk. A file the walk
// excluded (ignored dir, unsupported type, narrower root) but still present survives.
func (e *Engine) prune(ctx context.Context, root string, seen map[string]bool, sum *BatchSummary) error {
	rootURI := model.FileURI(root)
	for _, uri := range e.cache.URIs() {
		if err := ctx.Err(); err != nil {
			return err
		}
		if (uri != rootURI && !strings.HasPrefix(uri, rootURI+"/")) || seen[uri] {
			continue
		}
		if p, ok := model.PathFromFileURI(uri); ok {
			// Only a file known to be gone is pruned: a stat error such as EACCES says nothing about whether it exists.
			if _, err := e.stat(p); !errors.Is(err, fs.ErrNotExist) {
				if err != nil {
					e.log("sync: keep %s: %v", p, err)
				}
				continue
			}
		}
		out, err := e.deleteURI(ctx, uri)
		if err != nil {
			return err
		}
		e.tally(sum, out)
	}
	return nil
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
	e.saveCache()

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
	if !e.cfg.NoRecluster {
		e.markSurvivorsTouched(ctx, uri)
	}
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

// flush re-clusters after a batch unless NoRecluster. The cache is persisted separately by the caller's
// deferred save, so it is saved even when a change errors.
func (e *Engine) flush(ctx context.Context, sum *BatchSummary) {
	if e.cfg.NoRecluster {
		return
	}
	haveSidecar, err := e.recluster(ctx)
	if err != nil {
		e.log("sync: re-cluster failed: %v", err)
		return
	}
	sum.Reclustered = haveSidecar
}

// persistCache saves the content cache. It is the one place the cache reaches disk, so a store sync that must
// precede the save (making the writes the cache vouches for durable) belongs here.
func (e *Engine) persistCache() error {
	if err := e.cache.Save(); err != nil {
		return fmt.Errorf("save cache: %w", err)
	}
	return nil
}

// saveCache is persistCache for deferred best-effort use: it logs rather than returns the error.
func (e *Engine) saveCache() {
	if err := e.persistCache(); err != nil {
		e.log("sync: %v", err)
	}
}

func (e *Engine) tally(sum *BatchSummary, out Outcome) {
	switch {
	case out.Skipped:
		sum.Skipped++
		if out.Reason == reasonUnchanged {
			sum.Unchanged++
		}
	case out.Op == OpDelete:
		sum.Deleted++
	default:
		sum.Upserted++
		sum.Nodes += out.NodesUpserted
		sum.Edges += out.EdgesUpserted
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
