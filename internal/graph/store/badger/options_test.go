package badger

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func TestValueLogPreallocationBounded(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "graph.badger")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	logs, err := filepath.Glob(filepath.Join(dir, "*.vlog"))
	if err != nil || len(logs) == 0 {
		t.Fatalf("value logs = %v, %v", logs, err)
	}
	// Badger preallocates twice the file cap while a writer is open, and a crash leaves the file at that size.
	const maxPrealloc = 128 << 20
	for _, p := range logs {
		st, err := os.Stat(p)
		if err != nil {
			t.Fatal(err)
		}
		if st.Size() > maxPrealloc {
			t.Errorf("%s is %d bytes while open, want at most %d", filepath.Base(p), st.Size(), maxPrealloc)
		}
	}
}

// TestCloseWithValueLogGCStaysSealed guards the order in Close: GC commits before the seal records its
// version, so a store with writes and a collected value log still reopens sealed, with no reindex.
func TestCloseWithValueLogGCStaysSealed(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "graph.badger")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertNodes(context.Background(), []model.Node{{ID: "a", Kind: model.KindFile}, {ID: "b", Kind: model.KindFile}}); err != nil {
		t.Fatal(err)
	}
	if err := gs.Close(); err != nil {
		t.Fatal(err)
	}
	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	ok, err := sealed(s.db)
	if err != nil || !ok {
		t.Fatalf("sealed after Close with GC = %v, %v; want true", ok, err)
	}
}
