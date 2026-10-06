package badger

import (
	"path/filepath"
	"testing"
)

func TestEpochIdentifiesStore(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "a")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	epoch := gs.Epoch()
	if epoch == "" {
		t.Fatal("new store has no epoch")
	}
	if err := gs.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if err := gs.Close(); err != nil {
		t.Fatal(err)
	}

	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	if ro.Epoch() != epoch {
		t.Errorf("reopened epoch = %q, want %q", ro.Epoch(), epoch)
	}
	if err := ro.Sync(); err != nil {
		t.Errorf("read-only Sync: %v", err)
	}
	ro.Close()

	other, err := Open(filepath.Join(t.TempDir(), "b"))
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	if other.Epoch() == epoch {
		t.Fatal("two stores share an epoch")
	}
}
