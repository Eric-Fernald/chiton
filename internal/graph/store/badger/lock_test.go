package badger

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	badgerdb "github.com/dgraph-io/badger/v4"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// seededStore creates a store holding node "a", closes it, and returns its directory.
func seededStore(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "graph.badger")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertNodes(context.Background(), []model.Node{{ID: "a", Label: "a"}}); err != nil {
		t.Fatal(err)
	}
	if err := gs.Close(); err != nil {
		t.Fatal(err)
	}
	return dir
}

// memLogs lists dir's Badger memtable logs. A cleanly closed store has none.
func memLogs(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var logs []string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".mem") {
			logs = append(logs, e.Name())
		}
	}
	return logs
}

func TestReaderRefusedWhileWriterOpen(t *testing.T) {
	dir := seededStore(t)
	rw, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer rw.Close()
	ro, err := OpenReadOnly(dir)
	if !errors.Is(err, ErrStoreBusy) {
		if ro != nil {
			ro.Close()
		}
		t.Fatalf("OpenReadOnly beside a writer = %v, want ErrStoreBusy", err)
	}
}

func TestWriterRefusedWhileReaderOpenOnWindows(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("only Windows refuses a writer beside a reader")
	}
	dir := seededStore(t)
	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	rw, err := Open(dir)
	if !errors.Is(err, ErrStoreBusy) {
		if rw != nil {
			rw.Close()
		}
		ro.Close()
		t.Fatalf("Open beside a reader = %v, want ErrStoreBusy", err)
	}
	if logs := memLogs(t, dir); len(logs) != 0 {
		t.Errorf("refused writer left memtable logs %v", logs)
	}
	if err := ro.Close(); err != nil {
		t.Fatal(err)
	}
	rw, err = Open(dir)
	if err != nil {
		t.Fatalf("Open after the reader closed: %v", err)
	}
	rw.Close()
}

func TestWriterOpensBesideReaderOffWindows(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows refuses a writer beside a reader")
	}
	dir := seededStore(t)
	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ro.Close()
	rw, err := Open(dir)
	if err != nil {
		t.Fatalf("Open beside a reader = %v; a Unix reader keeps its snapshot, so a writer may open", err)
	}
	rw.Close()
}

func TestReadOnlyOpenReplaysCrashedWriterLog(t *testing.T) {
	dir := seededStore(t)
	// A crashed writer leaves a preallocated, mostly zero memtable log that a read-only Badger open rejects.
	if err := os.WriteFile(filepath.Join(dir, "00099.mem"), make([]byte, 1<<20), 0o600); err != nil {
		t.Fatal(err)
	}
	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatalf("OpenReadOnly after a crashed writer: %v", err)
	}
	defer ro.Close()
	if n, err := ro.GetNode(context.Background(), "a"); err != nil || n == nil {
		t.Fatalf("GetNode(a) = %v, %v; want the node", n, err)
	}
	if logs := memLogs(t, dir); len(logs) != 0 {
		t.Errorf("memtable logs %v survived the repair", logs)
	}
}

func TestIsTruncateNeeded(t *testing.T) {
	wrapped := fmt.Errorf("while opening memtables err: %+v", badgerdb.ErrTruncateNeeded)
	if !isTruncateNeeded(wrapped) {
		t.Errorf("isTruncateNeeded(%v) = false", wrapped)
	}
	if isTruncateNeeded(errors.New("other")) || isTruncateNeeded(nil) {
		t.Error("isTruncateNeeded matched an unrelated error")
	}
}
