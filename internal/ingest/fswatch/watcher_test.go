package fswatch

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/fsnotify/fsnotify"
)

type collector struct {
	mu   sync.Mutex
	seen map[string]int
}

func newCollector() *collector { return &collector{seen: map[string]int{}} }

func (c *collector) enqueue(p string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.seen[p]++
}

func (c *collector) has(p string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.seen[p] > 0
}

func goOpts() Options {
	return Options{
		Skip: func(p string, _ bool) bool {
			for _, part := range strings.Split(filepath.ToSlash(p), "/") {
				if part == ".git" || part == "chiton-out" {
					return true
				}
			}
			return false
		},
		Accept: func(p string) bool { return strings.HasSuffix(p, ".go") },
	}
}

// waitFor polls cond until true or the deadline elapses.
func waitFor(cond func() bool, d time.Duration) bool {
	deadline := time.After(d)
	for {
		if cond() {
			return true
		}
		select {
		case <-deadline:
			return cond()
		case <-time.After(10 * time.Millisecond):
		}
	}
}

func startWatcher(t *testing.T, root string, c *collector) {
	t.Helper()
	w, err := New(root, c.enqueue, goOpts())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		_ = w.Run(ctx)
		close(done)
	}()
	t.Cleanup(func() {
		cancel()
		<-done
		_ = w.Close()
	})
}

func TestWatchDetectsWrite(t *testing.T) {
	root := t.TempDir()
	c := newCollector()
	startWatcher(t, root, c)

	p := filepath.Join(root, "a.go")
	if err := os.WriteFile(p, []byte("package a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !waitFor(func() bool { return c.has(p) }, 5*time.Second) {
		t.Fatalf("watcher did not report write to %s", p)
	}
}

func TestWatchNewSubdir(t *testing.T) {
	root := t.TempDir()
	c := newCollector()
	startWatcher(t, root, c)

	sub := filepath.Join(root, "pkg")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(sub, "b.go")
	// Small settle so the Create-dir event registers the new watch first; the
	// watcher also re-scans created dirs, so this is belt-and-suspenders.
	time.Sleep(150 * time.Millisecond)
	if err := os.WriteFile(p, []byte("package pkg\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !waitFor(func() bool { return c.has(p) }, 5*time.Second) {
		t.Fatalf("watcher did not report file in newly created subdir %s", p)
	}
}

func TestWatchIgnoresIgnoredDir(t *testing.T) {
	root := t.TempDir()
	c := newCollector()
	startWatcher(t, root, c)

	git := filepath.Join(root, ".git")
	if err := os.MkdirAll(git, 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(git, "hook.go")
	if err := os.WriteFile(p, []byte("package x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Also touch a real file to give the watcher something to deliver, so the
	// negative assertion isn't just racing an empty event stream.
	real := filepath.Join(root, "real.go")
	if err := os.WriteFile(real, []byte("package r\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !waitFor(func() bool { return c.has(real) }, 5*time.Second) {
		t.Fatal("watcher missed the real file")
	}
	if c.has(p) {
		t.Errorf("watcher should ignore files under .git, but saw %s", p)
	}
}

// TestWatchDirRemove verifies removing a watched directory fires OnDirRemove with its path, since
// fsnotify sends no per-child events for a directory removal.
func TestWatchDirRemove(t *testing.T) {
	root := t.TempDir()
	sub := filepath.Join(root, "pkg")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sub, "x.go"), []byte("package pkg\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	removed := map[string]bool{}
	opts := goOpts()
	opts.OnDirRemove = func(dir string) {
		mu.Lock()
		removed[dir] = true
		mu.Unlock()
	}
	w, err := New(root, func(string) {}, opts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _ = w.Run(ctx); close(done) }()
	t.Cleanup(func() { cancel(); <-done; _ = w.Close() })

	// Confirm the subdir is actually watched before removing it.
	if !waitFor(func() bool {
		return w.isWatchedDir(sub)
	}, 3*time.Second) {
		t.Fatal("subdir was never watched")
	}

	if err := os.RemoveAll(sub); err != nil {
		t.Fatal(err)
	}
	if !waitFor(func() bool {
		mu.Lock()
		defer mu.Unlock()
		return removed[sub]
	}, 5*time.Second) {
		t.Fatalf("OnDirRemove was not called for %s", sub)
	}
}

func TestWatchRejectsNonAccepted(t *testing.T) {
	root := t.TempDir()
	c := newCollector()
	startWatcher(t, root, c)

	txt := filepath.Join(root, "notes.txt")
	if err := os.WriteFile(txt, []byte("hi\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gof := filepath.Join(root, "c.go")
	if err := os.WriteFile(gof, []byte("package c\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !waitFor(func() bool { return c.has(gof) }, 5*time.Second) {
		t.Fatal("watcher missed the .go file")
	}
	if c.has(txt) {
		t.Errorf("watcher should reject non-.go file %s", txt)
	}
}

// TestDirRemoveWhenWatchAlreadyDropped reproduces, on every platform, the Linux ordering in which fsnotify
// has forgotten a deleted dir when its Remove event arrives, and checks that OnDirRemove still fires.
func TestDirRemoveWhenWatchAlreadyDropped(t *testing.T) {
	root := t.TempDir()
	sub := filepath.Join(root, "pkg")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	removed := map[string]bool{}
	opts := goOpts()
	opts.OnDirRemove = func(dir string) {
		mu.Lock()
		removed[dir] = true
		mu.Unlock()
	}

	w, err := New(root, func(string) {}, opts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = w.Close() })

	if !w.isWatchedDir(sub) {
		t.Fatalf("precondition: %s should be watched after New", sub)
	}

	// Exactly the state Linux leaves us in: gone from fsnotify, gone from disk,
	// but the Remove event for it has not been processed yet.
	_ = w.w.Remove(sub)
	if err := os.RemoveAll(sub); err != nil {
		t.Fatal(err)
	}
	if slices.Contains(w.w.WatchList(), sub) {
		t.Fatalf("precondition: %s should already be gone from fsnotify's watch list", sub)
	}

	w.handle(fsnotify.Event{Name: sub, Op: fsnotify.Remove})

	mu.Lock()
	defer mu.Unlock()
	if !removed[sub] {
		t.Errorf("OnDirRemove was not called for %s; the directory removal was misread as a file event", sub)
	}
}

// TestDirRemoveForgetsSubtree checks a removed directory and its descendants no longer count as watched,
// or a later event on a recreated path would be misclassified.
func TestDirRemoveForgetsSubtree(t *testing.T) {
	root := t.TempDir()
	sub := filepath.Join(root, "pkg")
	nested := filepath.Join(sub, "inner")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}

	w, err := New(root, func(string) {}, goOpts())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = w.Close() })

	for _, d := range []string{root, sub, nested} {
		if !w.isWatchedDir(d) {
			t.Fatalf("precondition: %s should be watched after New", d)
		}
	}

	if err := os.RemoveAll(sub); err != nil {
		t.Fatal(err)
	}
	w.handle(fsnotify.Event{Name: sub, Op: fsnotify.Remove})

	if w.isWatchedDir(sub) {
		t.Errorf("%s is still recorded as watched after its removal", sub)
	}
	if w.isWatchedDir(nested) {
		t.Errorf("descendant %s is still recorded as watched after its parent was removed", nested)
	}
	if !w.isWatchedDir(root) {
		t.Errorf("root %s should still be watched", root)
	}
}
