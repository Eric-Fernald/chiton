package browser

import (
	"context"
	"path/filepath"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/ingest/fswatch"
	"github.com/Eric-Fernald/chiton/internal/jobs"
)

// defaultWatchDebounce is deliberately wide: Chromium writes History continuously while browsing, so a
// burst of writes should coalesce into one re-ingest.
const defaultWatchDebounce = 5 * time.Second

// Watch re-ingests each History database when its file changes until ctx is cancelled, then drains.
// Events are debounced per database, and a database is never ingested concurrently with itself.
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
	// Start on a background context, not ctx, so a cancel can't stop the queue before Drain flushes
	// the final debounced re-ingest.
	queue.Start(context.Background())

	// One root-only fswatch per profile dir: its subtree is huge (caches, blobs) and only the History
	// file and its sidecars, sitting directly in it, matter.
	parents := uniqueParents(byPath)
	var watchers []*fswatch.Watcher
	for parent := range parents {
		w, err := fswatch.New(parent, func(path string) {
			if key, ok := matchHistoryKey(path, byPath); ok {
				queue.Enqueue(key)
			}
		}, fswatch.Options{
			WatchRootOnly: true,
			Accept:        isHistoryEvent,
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
	for _, suffix := range sidecarSuffixes {
		if base, ok := strings.CutSuffix(clean, suffix); ok {
			if _, ok := byPath[base]; ok {
				return base, true
			}
		}
	}
	return "", false
}
