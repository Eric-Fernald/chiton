// Package fswatch is a thin recursive wrapper over fsnotify. fsnotify watches a
// single directory (not its subtree), so this package walks the tree, watches
// every directory, and adds watches for directories created later. Each event
// of interest is handed to an enqueue callback; debouncing and the actual
// re-extraction live in the live-sync engine, not here.
package fswatch

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/fsnotify/fsnotify"
)

// Options customise which paths the watcher cares about.
type Options struct {
	// IgnoreDir reports whether a directory (by base name) should not be
	// watched or descended into (e.g. ".git", "node_modules").
	IgnoreDir func(name string) bool
	// Accept reports whether a file path is of interest (e.g. has a supported
	// extractor). Events for files Accept rejects are dropped.
	Accept func(path string) bool
	// OnError, if set, receives non-fatal watcher errors.
	OnError func(error)
	// OnDirRemove, if set, is called when a previously-watched directory is
	// removed or renamed away. fsnotify delivers no per-child events for this,
	// so the consumer must prune anything it indexed under that directory.
	OnDirRemove func(dir string)
	// WatchRootOnly, when true, watches only the root directory itself and does
	// not descend into (or add watches for) its subdirectories. Use it when the
	// files of interest sit directly in one directory whose subtree is large and
	// irrelevant (e.g. a browser profile dir). Leave IgnoreDir nil in that case:
	// IgnoreDir is also consulted against file-event leaf names (see
	// pathIgnored), so a predicate meant only to suppress descent would also drop
	// the very file events you want to receive.
	WatchRootOnly bool
}

// Watcher watches a directory tree and reports relevant file changes.
type Watcher struct {
	root    string
	w       *fsnotify.Watcher
	enqueue func(path string)
	opts    Options
}

// New creates a Watcher rooted at root. enqueue is called (from Run's goroutine)
// with the path of each changed file of interest. Call Run to start, Close to
// release resources.
func New(root string, enqueue func(path string), opts Options) (*Watcher, error) {
	w, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}
	watcher := &Watcher{root: root, w: w, enqueue: enqueue, opts: opts}
	if err := watcher.addTree(root); err != nil {
		_ = w.Close()
		return nil, err
	}
	return watcher, nil
}

// Watched returns the number of directories currently watched (test hook).
func (w *Watcher) Watched() int { return len(w.w.WatchList()) }

// Run processes events until ctx is cancelled, then returns ctx.Err().
func (w *Watcher) Run(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err, ok := <-w.w.Errors:
			if !ok {
				return nil
			}
			if w.opts.OnError != nil && err != nil {
				w.opts.OnError(err)
			}
		case ev, ok := <-w.w.Events:
			if !ok {
				return nil
			}
			w.handle(ev)
		}
	}
}

// Close releases the underlying fsnotify watcher.
func (w *Watcher) Close() error { return w.w.Close() }

func (w *Watcher) handle(ev fsnotify.Event) {
	path := ev.Name

	// A newly created directory must be watched, and any files already inside
	// it (created before we added the watch) re-synced — unless we are watching
	// the root directory only, in which case new subdirectories are ignored.
	if ev.Op&fsnotify.Create != 0 {
		if fi, err := os.Stat(path); err == nil && fi.IsDir() {
			if !w.opts.WatchRootOnly && !w.ignoredDir(path) {
				_ = w.addTree(path)
				w.enqueueExisting(path)
			}
			return
		}
	}

	// A removed/renamed directory yields no per-child events. We detect the
	// directory case via our watch list (os.Stat fails once it is gone), drop
	// its subtree watches, and let the consumer prune what it indexed under it.
	if ev.Op&(fsnotify.Remove|fsnotify.Rename) != 0 && w.isWatchedDir(path) {
		w.removeSubtreeWatches(path)
		if w.opts.OnDirRemove != nil {
			w.opts.OnDirRemove(path)
		}
		return
	}

	// Ignore events inside ignored directories.
	if w.pathIgnored(path) {
		return
	}
	// Chmod-only events carry no content change.
	if ev.Op == fsnotify.Chmod {
		return
	}
	if w.opts.Accept != nil && !w.opts.Accept(path) {
		return
	}
	w.enqueue(path)
}

// addTree walks dir and adds a watch for every non-ignored directory. In
// WatchRootOnly mode it watches just dir itself and does not descend.
func (w *Watcher) addTree(dir string) error {
	if w.opts.WatchRootOnly {
		if err := w.w.Add(dir); err != nil && w.opts.OnError != nil {
			w.opts.OnError(err)
		}
		return nil
	}
	return filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			// A directory may vanish between listing and walking; skip it.
			return nil //nolint:nilerr
		}
		if !d.IsDir() {
			return nil
		}
		if path != dir && w.opts.IgnoreDir != nil && w.opts.IgnoreDir(d.Name()) {
			return filepath.SkipDir
		}
		if dir != path && w.ignoredDir(path) {
			return filepath.SkipDir
		}
		if err := w.w.Add(path); err != nil && w.opts.OnError != nil {
			w.opts.OnError(err)
		}
		return nil
	})
}

// enqueueExisting walks a freshly created directory and enqueues every accepted
// file already present, covering events that fired before the watch was added.
func (w *Watcher) enqueueExisting(dir string) {
	_ = filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil //nolint:nilerr
		}
		if d.IsDir() {
			if path != dir && w.opts.IgnoreDir != nil && w.opts.IgnoreDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if w.opts.Accept == nil || w.opts.Accept(path) {
			w.enqueue(path)
		}
		return nil
	})
}

// isWatchedDir reports whether path is currently one of our watched directories
// (only directories are ever watched, so membership in the watch list means it
// was a directory).
func (w *Watcher) isWatchedDir(path string) bool {
	return slices.Contains(w.w.WatchList(), path)
}

// removeSubtreeWatches drops the watch on dir and on every descendant directory,
// best-effort. fsnotify does not reliably auto-remove watches for a directory
// that was renamed/deleted out from under it.
func (w *Watcher) removeSubtreeWatches(dir string) {
	prefix := dir + string(filepath.Separator)
	for _, watched := range w.w.WatchList() {
		if watched == dir || strings.HasPrefix(watched, prefix) {
			_ = w.w.Remove(watched)
		}
	}
}

// ignoredDir reports whether dirPath's base name is ignored.
func (w *Watcher) ignoredDir(dirPath string) bool {
	if w.opts.IgnoreDir == nil {
		return false
	}
	return w.opts.IgnoreDir(filepath.Base(dirPath))
}

// pathIgnored reports whether any component of path (relative to root) is an
// ignored directory name, so events under e.g. chiton-out/ never trigger a
// self-feeding re-extraction loop.
func (w *Watcher) pathIgnored(path string) bool {
	if w.opts.IgnoreDir == nil {
		return false
	}
	rel, err := filepath.Rel(w.root, path)
	if err != nil {
		rel = path
	}
	return slices.ContainsFunc(splitPath(rel), w.opts.IgnoreDir)
}

func splitPath(p string) []string {
	var parts []string
	for {
		dir, file := filepath.Split(p)
		if file != "" {
			parts = append(parts, file)
		}
		if dir == "" {
			break
		}
		p = filepath.Clean(dir)
		if p == "." || p == string(filepath.Separator) {
			break
		}
	}
	return parts
}
