package ingest

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/cache"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
	"github.com/Eric-Fernald/chiton/internal/walk"
)

// logSink collects an engine's log lines.
type logSink struct {
	mu    sync.Mutex
	lines []string
}

func (l *logSink) logf(format string, args ...any) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.lines = append(l.lines, fmt.Sprintf(format, args...))
}

// count returns how many lines contain sub.
func (l *logSink) count(sub string) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	n := 0
	for _, line := range l.lines {
		if strings.Contains(line, sub) {
			n++
		}
	}
	return n
}

// loggingEngine is the harness engine with its log captured and watch debounces short enough to poll for.
func loggingEngine(h *harness) (*Engine, *logSink) {
	logs := &logSink{}
	return New(h.gs, h.cache, Config{
		GraphPath:       h.eng.cfg.GraphPath,
		Debounce:        20 * time.Millisecond,
		ClusterDebounce: 50 * time.Millisecond,
	}, nil, nil, logs.logf), logs
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// startWatch runs eng.Watch over h.dir and returns once its watches are registered. The watch is stopped
// before the harness closes the store.
func startWatch(t *testing.T, h *harness, eng *Engine, logs *logSink) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- eng.Watch(ctx, h.dir) }()
	t.Cleanup(func() {
		cancel()
		if err := <-done; err != nil {
			t.Errorf("Watch: %v", err)
		}
	})
	waitFor(t, "the watcher to start", func() bool { return logs.count("watch: watching") > 0 })
}

// TestWatchSavesCacheWhileRunning checks a settled batch reaches cache.json while watch is still up, so a
// hard kill does not leave the cache behind the store.
func TestWatchSavesCacheWhileRunning(t *testing.T) {
	h := newHarness(t, false, false)
	eng, logs := loggingEngine(h)
	startWatch(t, h, eng, logs)

	const content = "package src\nfunc W() {}\n"
	uri := model.FileURI(h.write(t, "w.go", content))
	cachePath := filepath.Join(h.graphDir, "cache.json")
	waitFor(t, "cache.json to record w.go before shutdown", func() bool {
		onDisk, err := cache.Load(cachePath, h.gs.Epoch())
		return err == nil && !onDisk.Changed(uri, model.ContentSHA([]byte(content)))
	})
}

// syncHook is a store whose Sync first calls during, standing in for watch work that overlaps the flush.
type syncHook struct {
	*badgerstore.Store
	during func()
}

func (s syncHook) Sync() error {
	s.during()
	return s.Store.Sync()
}

// TestPersistCacheHoldsWritesDuringStoreSync checks a file synced while the store flushes is not in the cache
// that flush saves, so cache.json never vouches for a write the flush did not cover.
func TestPersistCacheHoldsWritesDuringStoreSync(t *testing.T) {
	h := newHarness(t, false, false)
	const content = "package src\nfunc Late() {}\n"
	p := h.write(t, "late.go", content)
	uri, sha := model.FileURI(p), model.ContentSHA([]byte(content))

	var eng *Engine
	synced := make(chan error, 1)
	eng = New(syncHook{h.gs, func() {
		go func() {
			_, err := eng.SyncFile(context.Background(), p)
			synced <- err
		}()
		time.Sleep(200 * time.Millisecond) // long enough for an unblocked SyncFile to finish
	}}, h.cache, Config{GraphPath: h.eng.cfg.GraphPath}, nil, nil, nil)

	if err := eng.persistCache(); err != nil {
		t.Fatal(err)
	}
	onDisk, err := cache.Load(filepath.Join(h.graphDir, "cache.json"), h.gs.Epoch())
	if err != nil {
		t.Fatal(err)
	}
	if !onDisk.Changed(uri, sha) {
		t.Error("cache.json records a file written after the store sync began")
	}
	if err := <-synced; err != nil || h.cache.Changed(uri, sha) {
		t.Errorf("SyncFile err = %v; want the file indexed once the save finished", err)
	}
}

// TestWatchLogsReclusterFailure checks a failed re-cluster in watch mode is reported, not dropped.
func TestWatchLogsReclusterFailure(t *testing.T) {
	h := newHarness(t, false, false)
	eng, logs := loggingEngine(h)
	if err := os.WriteFile(cluster.SidecarPath(eng.cfg.GraphPath), []byte("{truncated"), 0o644); err != nil {
		t.Fatal(err)
	}
	startWatch(t, h, eng, logs)

	h.write(t, "w.go", "package src\nfunc W() {}\n")
	waitFor(t, "the failed re-cluster to be logged", func() bool { return logs.count("watch: re-cluster failed") > 0 })
}

// TestRescanFailureIsLogged checks an overflow rescan that fails (here the root is gone) is reported.
func TestRescanFailureIsLogged(t *testing.T) {
	h := newHarness(t, false, false)
	eng, logs := loggingEngine(h)
	gone := filepath.Join(h.dir, "gone")
	policy, err := walk.LoadPolicy(gone)
	if err != nil {
		t.Fatal(err)
	}
	q := eng.rescanQueue(gone, policy, func(string) {})
	q.Start(context.Background())
	q.Enqueue("rescan")
	q.Drain()
	if logs.count("watch: rescan failed") != 1 {
		t.Errorf("log = %q; want the rescan failure reported once", logs.lines)
	}
}

// TestReclusterRetriesAfterFailure checks a failed pass keeps its seeds: the next one adds the new file's
// symbols and drops the deleted file's.
func TestReclusterRetriesAfterFailure(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	pa := h.write(t, "a.go", "package src\nfunc A() {}\n")
	if _, err := h.eng.SyncFile(ctx, pa); err != nil {
		t.Fatal(err)
	}
	seedSidecar(t, h)
	sidecar := cluster.SidecarPath(h.eng.cfg.GraphPath)
	good, err := os.ReadFile(sidecar)
	if err != nil {
		t.Fatal(err)
	}

	pb := h.write(t, "b.go", "package src\nfunc B() {}\n")
	if _, err := h.eng.SyncFile(ctx, pb); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(pa); err != nil {
		t.Fatal(err)
	}
	if _, err := h.eng.DeleteFile(ctx, pa); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(sidecar, []byte("{truncated"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := h.eng.recluster(ctx); err == nil {
		t.Fatal("recluster over an unreadable sidecar returned no error")
	}

	if err := os.WriteFile(sidecar, good, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := h.eng.recluster(ctx); err != nil {
		t.Fatal(err)
	}
	res, err := cluster.LoadSidecar(sidecar, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := res.Nodes[model.NewNodeID(model.FileURI(pb), "B")]; !ok {
		t.Error("B, synced before the failed pass, never reached the sidecar")
	}
	if _, ok := res.Nodes[model.NewNodeID(model.FileURI(pa), "A")]; ok {
		t.Error("A, deleted before the failed pass, is still in the sidecar")
	}
}

// TestSyncFileLogsFailedSkips checks the single-file path (watch, sync git) reports a file it could not read,
// and stays quiet about unchanged and unsupported ones.
func TestSyncFileLogsFailedSkips(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	eng, logs := loggingEngine(h)
	p := h.write(t, "a.go", "package src\nfunc A() {}\n")

	eng.lstat = func(string) (fs.FileInfo, error) { return nil, fs.ErrPermission }
	if out, err := eng.SyncFile(ctx, p); err != nil || !out.Skipped {
		t.Fatalf("SyncFile = %+v, %v; want a skip", out, err)
	}
	if logs.count("skip "+p) != 1 {
		t.Fatalf("log = %q; want the unreadable file reported", logs.lines)
	}

	eng.lstat = os.Lstat
	for _, quiet := range []string{p, p, h.write(t, "blob.bin", "x")} {
		if _, err := eng.SyncFile(ctx, quiet); err != nil {
			t.Fatal(err)
		}
	}
	if logs.count("skip ") != 1 {
		t.Errorf("log = %q; want no line for an indexed, unchanged, or unsupported file", logs.lines)
	}
}
