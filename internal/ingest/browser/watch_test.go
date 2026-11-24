package browser

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func countURLNodes(t *testing.T, gs interface {
	AllNodes(context.Context) ([]model.Node, error)
}) int {
	t.Helper()
	nodes, _ := gs.AllNodes(context.Background())
	n := 0
	for _, nd := range nodes {
		if nd.Kind == model.KindURL {
			n++
		}
	}
	return n
}

// TestWatchReingestsOnChange guards that a History file change reaches the ingest handler (a past IgnoreDir
// misuse dropped every event) and that shutdown drains cleanly.
func TestWatchReingestsOnChange(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "History")
	writeHistoryDB(t, dbPath, []Visit{
		{URL: "https://example.com/watch", Title: "Watch", VisitCount: 1, LastVisit: time.Now().UTC()},
	})

	gs := newStore(t)
	ing := New(gs)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- ing.Watch(ctx, []DetectedDB{{Browser: "chrome", Path: dbPath}}, Options{}, 50*time.Millisecond, nil)
	}()

	// Watch does not ingest existing state, only changes — so it should be empty
	// until we mutate the file.
	time.Sleep(250 * time.Millisecond)
	if n := countURLNodes(t, gs); n != 0 {
		t.Fatalf("expected 0 url nodes before any change, got %d", n)
	}

	// Mutate the History file (rewrite it) to trigger a change event.
	if err := os.Remove(dbPath); err != nil {
		t.Fatal(err)
	}
	writeHistoryDB(t, dbPath, []Visit{
		{URL: "https://example.com/watch", Title: "Watch", VisitCount: 2, LastVisit: time.Now().UTC()},
	})

	// Poll for the handler to run.
	deadline := time.Now().Add(8 * time.Second)
	got := 0
	for time.Now().Before(deadline) {
		if got = countURLNodes(t, gs); got > 0 {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if got == 0 {
		t.Fatal("watcher never re-ingested the changed History file (event dropped?)")
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Watch returned error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Watch did not shut down within 5s of cancel")
	}
}
