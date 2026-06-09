package browser

import (
	"context"
	"path/filepath"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/ingest/fswatch"
	"github.com/Eric-Fernald/chiton/internal/jobs"
)

// defaultWatchDebounce is the idle window before re-ingesting a History file.
// Chromium writes to History continuously while browsing, so the window is
// deliberately wide to coalesce a burst of writes into one re-ingest.
const defaultWatchDebounce = 5 * time.Second

// Watch watches the directories holding the given History databases and
// re-ingests a database when its file changes, until ctx is cancelled. It reuses
// the same fswatch + jobs.Queue machinery as the Phase 3 live-sync watcher:
// events are debounced and coalesced per database, and a database already being
// ingested is never ingested concurrently with itself.
//
// On ctx cancellation it drains in-flight work and closes its watchers cleanly.
func (ing *Ingester) Watch(ctx context.Context, dbs []DetectedDB, opts Options, debounce time.Duration, logf func(string, ...any)) error {
	if len(dbs) == 0 {
		return nil
	}
	if debounce <= 0 {
		debounce = defaultWatchDebounce
	}
	if logf == nil {
		logf = func(string, ...any) {}
	}

	// Map each History file path to the browser that owns it, so the queue
	// handler (keyed on the path) knows which label to record.
	byPath := make(map[string]string, len(dbs))
	for _, d := range dbs {
		byPath[filepath.Clean(d.Path)] = d.Browser
	}

	queue := jobs.New(func(ctx context.Context, key string) error {
		rep, err := ing.IngestDB(ctx, key, byPath[key], opts)
		if err != nil {
			return err
		}
		logf("browser: re-ingested %s (%s) — %d pages", rep.Browser, key, rep.NodesUpserted)
		return nil
	}, jobs.Options{
		Workers:  1, // ingests touch the shared store; serialise them
		Debounce: debounce,
		OnError: func(key string, err error) {
			logf("browser: ingest %s failed: %v", key, err)
		},
	})
	// Start the queue on a background context (NOT the watch ctx) so a Ctrl-C
	// cancel does not hard-stop it before Drain() can flush the final debounced
	// re-ingest — mirroring the Phase 3 live-sync engine's queue lifecycle.
	queue.Start(context.Background())

	// One fswatch per unique parent directory, in WatchRootOnly mode so only the
	// profile directory itself is watched (the "User Data/<profile>" tree is huge
	// — caches, blobs — and we only care about the History file and its
	// journal/WAL sidecars sitting directly in it).
	parents := uniqueParents(byPath)
	var watchers []*fswatch.Watcher
	for parent := range parents {
		w, err := fswatch.New(parent, func(path string) {
			if key, ok := matchHistoryKey(path, byPath); ok {
				queue.Enqueue(key)
			}
		}, fswatch.Options{
			WatchRootOnly: true,
			Accept:        func(p string) bool { return isHistoryEvent(p) },
			OnError:       func(err error) { logf("browser watch: %v", err) },
		})
		if err != nil {
			// Best-effort: clean up watchers already started, then fail.
			for _, prev := range watchers {
				_ = prev.Close()
			}
			queue.Stop()
			return err
		}
		watchers = append(watchers, w)
	}

	// Run every watcher until ctx is cancelled.
	for _, w := range watchers {
		go func(w *fswatch.Watcher) { _ = w.Run(ctx) }(w)
	}

	<-ctx.Done()
	for _, w := range watchers {
		_ = w.Close()
	}
	queue.Drain() // process any debounced re-ingest before returning
	return nil
}

// uniqueParents returns the set of directories containing the watched DB files.
func uniqueParents(byPath map[string]string) map[string]struct{} {
	parents := make(map[string]struct{})
	for p := range byPath {
		parents[filepath.Dir(p)] = struct{}{}
	}
	return parents
}

// isHistoryEvent reports whether an event path concerns a History file (the DB
// itself or its -journal / -wal / -shm sidecars Chromium writes alongside it).
func isHistoryEvent(path string) bool {
	return strings.HasPrefix(filepath.Base(path), "History")
}

// matchHistoryKey maps an event path (possibly a -wal/-shm/-journal sidecar)
// back to a watched History DB key, if any.
func matchHistoryKey(path string, byPath map[string]string) (string, bool) {
	clean := filepath.Clean(path)
	if _, ok := byPath[clean]; ok {
		return clean, true
	}
	// Sidecar write (History-wal, History-shm, History-journal): strip the
	// suffix and match the base DB.
	for _, suffix := range []string{"-wal", "-shm", "-journal"} {
		if base, ok := strings.CutSuffix(clean, suffix); ok {
			if _, ok := byPath[base]; ok {
				return base, true
			}
		}
	}
	return "", false
}
