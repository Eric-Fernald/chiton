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
	"sync"

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

	// mu guards dirs, which is our own record of the directories we asked
	// fsnotify to watch.
	//
	// We deliberately do NOT use fsnotify's WatchList() to decide whether a
	// removed path was a directory. On Linux, fsnotify drops a watch from that
	// list while it drains the kernel's event buffer — IN_DELETE_SELF and
	// IN_IGNORED for the removed directory are processed synchronously, before
	// the parent's Remove event is handed to us on the Events channel. By the
	// time we see "root/pkg was removed", fsnotify has already forgotten it was
	// a watched directory, so a WatchList() check reports false and the removal
	// is silently treated as a file event: OnDirRemove never fires and the
	// consumer never prunes the subtree.
	//
	// Our own set is only mutated by us, so the answer does not depend on how
	// far fsnotify has got through its buffer.
	mu   sync.Mutex
	dirs map[string]bool
}

// New creates a Watcher rooted at root. enqueue is called (from Run's goroutine)
// with the path of each changed file of interest. Call Run to start, Close to
// release resources.
func New(root string, enqueue func(path string), opts Options) (*Watcher, error) {
	w, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}
	watcher := &Watcher{root: root, w: w, enqueue: enqueue, opts: opts, dirs: map[string]bool{}}
	if err := watcher.addTree(root); err != nil {
		_ = w.Close()
		return nil, err
	}
	return watcher, nil
}

// Watched returns the number of directories currently watched.
func (w *Watcher) Watched() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.dirs)
}

// markWatched records that we added a watch for dir.
func (w *Watcher) markWatched(dir string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.dirs[dir] = true
}

// watchedUnder returns dir plus every watched descendant directory.
func (w *Watcher) watchedUnder(dir string) []string {
	prefix := dir + string(filepath.Separator)
	w.mu.Lock()
	defer w.mu.Unlock()
	var out []string
	for watched := range w.dirs {
		if watched == dir || strings.HasPrefix(watched, prefix) {
			out = append(out, watched)
		}
	}
	return out
}

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

	// A removed/renamed directory yields no per-child events. os.Stat cannot
	// tell us it was a directory — it is already gone — so we ask our own
	// watched-directory set, which is authoritative here precisely because
	// fsnotify's may already have been pruned (see Watcher.dirs). Then drop the
	// subtree watches and let the consumer prune what it indexed underneath.
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
		if err := w.w.Add(dir); err != nil {
			if w.opts.OnError != nil {
				w.opts.OnError(err)
			}
			return nil
		}
		w.markWatched(dir)
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
		if err := w.w.Add(path); err != nil {
			if w.opts.OnError != nil {
				w.opts.OnError(err)
			}
			return nil
		}
		w.markWatched(path)
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

// isWatchedDir reports whether path is one of the directories we are watching
// (only directories are ever watched, so membership means it was a directory).
//
// This consults our own set rather than fsnotify's — see the note on
// Watcher.dirs for why that distinction is load-bearing.
func (w *Watcher) isWatchedDir(path string) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.dirs[path]
}

// removeSubtreeWatches drops the watch on dir and on every descendant directory,
// best-effort. On Linux the kernel has usually already invalidated these watches
// by the time we get here, so fsnotify's Remove returns ErrNonExistentWatch —
// that is expected and ignored. Forgetting them on our side is the part that
// matters.
func (w *Watcher) removeSubtreeWatches(dir string) {
	stale := w.watchedUnder(dir)
	for _, watched := range stale {
		_ = w.w.Remove(watched)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, watched := range stale {
		delete(w.dirs, watched)
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
