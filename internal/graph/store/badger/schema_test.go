package badger

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	badgerdb "github.com/dgraph-io/badger/v4"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// schemaOf returns db's m:schema stamp.
func schemaOf(t *testing.T, db *badgerdb.DB) string {
	t.Helper()
	var v string
	if err := db.View(func(txn *badgerdb.Txn) (err error) {
		v, err = getMeta(txn, metaSchema)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	return v
}

func TestFreshStoreStampedWithSchema(t *testing.T) {
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	if got := schemaOf(t, gs.db); got != strconv.Itoa(schemaVersion) {
		t.Fatalf("fresh store schema = %q, want %d", got, schemaVersion)
	}
}

func TestStoredSchemaOfUnstampedStores(t *testing.T) {
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	if err := gs.db.Update(func(txn *badgerdb.Txn) error { return txn.Delete([]byte(metaSchema)) }); err != nil {
		t.Fatal(err)
	}
	if v, err := storedSchema(gs.db); err != nil || v != schemaFresh {
		t.Fatalf("unstamped store without nodes = v%d, %v; want fresh", v, err)
	}
	if err := gs.UpsertNodes(context.Background(), []model.Node{{ID: "a"}}); err != nil {
		t.Fatal(err)
	}
	if v, err := storedSchema(gs.db); err != nil || v != 1 {
		t.Fatalf("unstamped store with nodes = v%d, %v; want v1 (written before versioning)", v, err)
	}
}

func TestNewerSchemaRefused(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "graph.badger")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := setMeta(gs.db, metaSchema, strconv.Itoa(schemaVersion+1)); err != nil {
		t.Fatal(err)
	}
	if err := gs.Close(); err != nil {
		t.Fatal(err)
	}
	if s, err := Open(dir); !errors.Is(err, ErrSchemaTooNew) {
		if s != nil {
			s.Close()
		}
		t.Fatalf("Open of a newer store = %v, want ErrSchemaTooNew", err)
	}
	if s, err := OpenReadOnly(dir); !errors.Is(err, ErrSchemaTooNew) {
		if s != nil {
			s.Close()
		}
		t.Fatalf("OpenReadOnly of a newer store = %v, want ErrSchemaTooNew", err)
	}
}

func TestMigrateStampsEachStep(t *testing.T) {
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	var ran []int
	steps := map[int]func(*badgerdb.DB) error{
		1: func(*badgerdb.DB) error { ran = append(ran, 1); return nil },
		2: func(*badgerdb.DB) error { ran = append(ran, 2); return errors.New("boom") },
	}
	if err := migrate(gs.db, 1, 3, steps); err == nil {
		t.Fatal("migrate succeeded despite a failing step")
	}
	if !slices.Equal(ran, []int{1, 2}) {
		t.Fatalf("steps ran %v, want [1 2]", ran)
	}
	if got := schemaOf(t, gs.db); got != "2" {
		t.Fatalf("schema after the v2 step failed = %q, want 2 (only the v1 step stamped)", got)
	}
	if err := migrate(gs.db, 3, 4, steps); err == nil {
		t.Fatal("migrate without a v3 step succeeded")
	}
}
