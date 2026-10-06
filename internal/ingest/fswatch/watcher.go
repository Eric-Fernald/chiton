// Package fswatch wraps fsnotify, which only watches single directories, to watch every directory in a tree
// (including ones created later) and pass file events to a callback.
package fswatch

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/fsnotify/fsnotify"
)

// Options customise which paths the watcher cares about.
type Options struct {
	// Skip reports whether a path under root is ignored: a directory not to watch or descend into, or a file
	// whose events to drop. isDir says whether the path is known to be a directory.
	Skip func(path string, isDir bool) bool
	// Accept reports whether a file path is of interest (e.g. has a supported
	// extractor). Events for files Accept rejects are dropped.
	Accept func(path string) bool
	// OnError, if set, receives non-fatal watcher errors.
	OnError func(error)
	// OnDirRemove, if set, is called when a watched directory is removed or renamed away. fsnotify
	// sends no per-child events for this, so the consumer must prune what it indexed underneath.
	OnDirRemove func(dir string)
	// WatchRootOnly watches only root, never its subdirectories (e.g. a huge browser profile dir). Leave
	// Skip nil with it, since Skip also sees file-event paths and would drop wanted events.
	WatchRootOnly bool
}

// Watcher watches a directory tree and reports relevant file changes.
type Watcher struct {
	root    string
	w       *fsnotify.Watcher
	enqueue func(path string)
	opts    Options

	// mu guards dirs, our own record of watched directories. fsnotify's WatchList can't classify a
	// removal, because on Linux it forgets a deleted dir before the parent's Remove event reaches us.
	mu   sync.Mutex
	dirs map[string]bool
}

// New creates a Watcher rooted at root. Run's goroutine calls enqueue with each changed file of interest.
// Call Run to start and Close to release resources.
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

	// Watch a newly created directory and re-sync files already inside it (created before the watch
	// was added), unless watching the root only.
	if ev.Op&fsnotify.Create != 0 {
		if fi, err := os.Stat(path); err == nil && fi.IsDir() {
			if !w.opts.WatchRootOnly && !w.ignoredDir(path) {
				_ = w.addTree(path)
				w.enqueueExisting(path)
			}
			return
		}
	}

	// A removed or renamed directory yields no per-child events and os.Stat can't classify a gone path, so
	// ask our own watched set (fsnotify's may already be pruned), then drop the subtree and notify.
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
		if path != dir && w.ignoredDir(path) {
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
			if path != dir && w.ignoredDir(path) {
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

// isWatchedDir reports whether path is a directory we watch, consulting our own set rather than
// fsnotify's (see Watcher.dirs).
func (w *Watcher) isWatchedDir(path string) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.dirs[path]
}

// removeSubtreeWatches drops and forgets the watches on dir and its descendants. On Linux the kernel has
// usually invalidated them already, so it expects and ignores fsnotify's ErrNonExistentWatch.
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

// ignoredDir reports whether the directory at dirPath is skipped.
func (w *Watcher) ignoredDir(dirPath string) bool {
	return w.opts.Skip != nil && w.opts.Skip(dirPath, true)
}

// pathIgnored reports whether path is skipped or lies under a skipped directory, so events under e.g.
// chiton-out/ never trigger a self-feeding re-extraction loop.
func (w *Watcher) pathIgnored(path string) bool {
	return w.opts.Skip != nil && w.opts.Skip(path, false)
}
