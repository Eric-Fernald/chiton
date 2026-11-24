package badger

import (
	"os"
	"path/filepath"
	"testing"
)

func TestValueLogPreallocationBounded(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "graph.badger")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := gs.Close(); err != nil {
			t.Error(err)
		}
	})
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
