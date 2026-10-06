package badger

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	badgerdb "github.com/dgraph-io/badger/v4"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// indexKeys returns every kind and source-URI index key in gs, in key order.
func indexKeys(t *testing.T, gs *Store) []string {
	t.Helper()
	var keys []string
	if err := gs.db.View(func(txn *badgerdb.Txn) error {
		for _, pfx := range []string{pfxKind, pfxURI} {
			if err := eachKey(txn, pfx, func(k []byte) error {
				keys = append(keys, pfx+string(k))
				return nil
			}); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return keys
}

func sortedKeys(keys ...[]byte) []string {
	out := make([]string, len(keys))
	for i, k := range keys {
		out[i] = string(k)
	}
	slices.Sort(out)
	return out
}

// rawDB opens dir with Badger directly, bypassing chiton's lock, layout, and seal handling.
func rawDB(t *testing.T, dir string) *badgerdb.DB {
	t.Helper()
	db, err := badgerdb.Open(badgerOptions(dir, Options{}, false))
	if err != nil {
		t.Fatal(err)
	}
	return db
}

func TestIndexesTrackNodeWrites(t *testing.T) {
	ctx := context.Background()
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	check := func(stage string, want []string) {
		t.Helper()
		if got := indexKeys(t, gs); !slices.Equal(got, want) {
			t.Fatalf("%s: index keys %q, want %q", stage, got, want)
		}
	}

	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "a", Kind: model.KindSymbol, SourceURI: "file:///x.go"},
		{ID: "b", Kind: model.KindFile, SourceURI: "file:///x.go"},
		{ID: "c", Kind: model.KindFile, SourceURI: "bad\xffuri"},
	}); err != nil {
		t.Fatal(err)
	}
	check("upsert", sortedKeys(
		kindKey(model.KindSymbol, "a"), uriKey("file:///x.go", "a"),
		kindKey(model.KindFile, "b"), uriKey("file:///x.go", "b"),
		kindKey(model.KindFile, "c"), uriKey("bad�uri", "c"), // keyed as the JSON record stores it
	))

	// Re-kinding and moving a, twice in one batch, leaves only its final keys.
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "a", Kind: model.KindRule, SourceURI: "file:///z.go"},
		{ID: "a", Kind: model.KindConcept, SourceURI: "file:///y.go"},
	}); err != nil {
		t.Fatal(err)
	}
	check("re-kind", sortedKeys(
		kindKey(model.KindConcept, "a"), uriKey("file:///y.go", "a"),
		kindKey(model.KindFile, "b"), uriKey("file:///x.go", "b"),
		kindKey(model.KindFile, "c"), uriKey("bad�uri", "c"),
	))

	if n, err := gs.DeleteBySourceURI(ctx, "file:///x.go"); err != nil || n != 1 {
		t.Fatalf("DeleteBySourceURI = %d, %v; want 1", n, err)
	}
	if n, err := gs.DeleteByIDs(ctx, []string{"a"}); err != nil || n != 1 {
		t.Fatalf("DeleteByIDs = %d, %v; want 1", n, err)
	}
	check("delete", sortedKeys(kindKey(model.KindFile, "c"), uriKey("bad�uri", "c")))
}

// TestReplaceSourcesKeepsIndexesExact moves a node between sources in one ReplaceSources batch: the old source
// deletes it while the new one re-adds it with the same kind, so its kind key is deleted and set in one batch.
func TestReplaceSourcesKeepsIndexesExact(t *testing.T) {
	ctx := context.Background()
	gs := openTestStore(t)
	const x, y = "file:///x.go", "file:///y.go"
	if _, err := gs.ReplaceSources(ctx, []store.SourceUpdate{{URI: x, Nodes: []model.Node{
		{ID: "a", Kind: model.KindSymbol, SourceURI: x},
		{ID: "b", Kind: model.KindFile, SourceURI: x},
	}}}); err != nil {
		t.Fatal(err)
	}
	deleted, err := gs.ReplaceSources(ctx, []store.SourceUpdate{
		{URI: x, Nodes: []model.Node{{ID: "b", Kind: model.KindSymbol, SourceURI: x}}},
		{URI: y, Nodes: []model.Node{{ID: "a", Kind: model.KindSymbol, SourceURI: y}}, Fresh: true},
	})
	if err != nil || !slices.Equal(deleted, []int{1, 0}) {
		t.Fatalf("ReplaceSources = %v, %v; want [1 0]", deleted, err)
	}
	want := sortedKeys(
		kindKey(model.KindSymbol, "a"), uriKey(y, "a"),
		kindKey(model.KindSymbol, "b"), uriKey(x, "b"),
	)
	if got := indexKeys(t, gs); !slices.Equal(got, want) {
		t.Fatalf("index keys %q, want %q", got, want)
	}
}

func TestCountByKind(t *testing.T) {
	ctx := context.Background()
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "a", Kind: model.KindSymbol}, {ID: "b", Kind: model.KindSymbol}, {ID: "c", Kind: model.KindFile}, {ID: "d"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{
		{ID: "ab", Src: "a", Dst: "b", Kind: model.EdgeCalls},
		{ID: "ca", Src: "c", Dst: "a", Kind: model.EdgeCalls},
		{ID: "bc", Src: "b", Dst: "c", Kind: "x:y"}, // an imported kind may contain ':'
	}); err != nil {
		t.Fatal(err)
	}
	check := func(wantNodes map[model.NodeKind]int, wantEdges map[model.EdgeKind]int) {
		t.Helper()
		nodes, edges, err := gs.CountByKind(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !maps.Equal(nodes, wantNodes) || !maps.Equal(edges, wantEdges) {
			t.Fatalf("CountByKind = %v %v, want %v %v", nodes, edges, wantNodes, wantEdges)
		}
	}
	check(map[model.NodeKind]int{model.KindSymbol: 2, model.KindFile: 1, "": 1},
		map[model.EdgeKind]int{model.EdgeCalls: 2, "x:y": 1})
	if _, err := gs.DeleteByIDs(ctx, []string{"a"}); err != nil {
		t.Fatal(err)
	}
	check(map[model.NodeKind]int{model.KindSymbol: 1, model.KindFile: 1, "": 1},
		map[model.EdgeKind]int{"x:y": 1})
}

// TestV1StoreMigratesOnReadOnlyOpen rewinds a store to the pre-index layout, unstamped (written before
// versioning) or stamped v1, and checks that a read-only open upgrades it and comes back read-only and indexed.
func TestV1StoreMigratesOnReadOnlyOpen(t *testing.T) {
	for _, stamp := range []string{"", "1"} {
		t.Run("stamp="+stamp, func(t *testing.T) {
			ctx := context.Background()
			dir := filepath.Join(t.TempDir(), "graph.badger")
			gs, err := Open(dir)
			if err != nil {
				t.Fatal(err)
			}
			if err := gs.UpsertNodes(ctx, []model.Node{
				{ID: "a", Kind: model.KindFile, SourceURI: "file:///a.go"},
				{ID: "b", Kind: model.KindSymbol, SourceURI: "file:///a.go"},
			}); err != nil {
				t.Fatal(err)
			}
			if err := gs.Close(); err != nil {
				t.Fatal(err)
			}
			// Rewind to a v1 store: node records only, no index or metadata keys beyond the optional stamp.
			db := rawDB(t, dir)
			if err := db.DropPrefix([]byte(pfxURI), []byte(pfxKind), []byte(pfxMeta)); err != nil {
				t.Fatal(err)
			}
			if stamp != "" {
				if err := setMeta(db, metaSchema, stamp); err != nil {
					t.Fatal(err)
				}
			}
			if err := db.Close(); err != nil {
				t.Fatal(err)
			}

			ro, err := OpenReadOnly(dir)
			if err != nil {
				t.Fatalf("OpenReadOnly of a v1 store: %v", err)
			}
			defer ro.Close()
			if !ro.ReadOnly() {
				t.Fatal("OpenReadOnly after the upgrade returned a writable store")
			}
			if got := schemaOf(t, ro.db); got != strconv.Itoa(schemaVersion) {
				t.Fatalf("schema after upgrade = %q, want %d", got, schemaVersion)
			}
			if got, err := ro.NodesMatching(ctx, store.NodeFilter{SourceURI: "file:///a.go"}); err != nil || len(got) != 2 {
				t.Fatalf("NodesMatching by source_uri = %d nodes, %v; want 2", len(got), err)
			}
			nodes, _, err := ro.CountByKind(ctx)
			if err != nil || !maps.Equal(nodes, map[model.NodeKind]int{model.KindFile: 1, model.KindSymbol: 1}) {
				t.Fatalf("CountByKind nodes = %v, %v", nodes, err)
			}
		})
	}
}

// TestUnsealedStoreIsReindexed guards the seal: a write that bypassed index upkeep (a crash, or an older
// chiton) must not hide a node from DeleteBySourceURI.
func TestUnsealedStoreIsReindexed(t *testing.T) {
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "graph.badger")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertNodes(ctx, []model.Node{{ID: "a", Kind: model.KindFile, SourceURI: "file:///x.go"}}); err != nil {
		t.Fatal(err)
	}
	if err := gs.Close(); err != nil {
		t.Fatal(err)
	}
	rec, err := json.Marshal(model.Node{ID: "b", Kind: model.KindSymbol, SourceURI: "file:///x.go"})
	if err != nil {
		t.Fatal(err)
	}
	db := rawDB(t, dir)
	if err := db.Update(func(txn *badgerdb.Txn) error { return txn.Set(nodeKey("b"), rec) }); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	gs, err = Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	if n, err := gs.DeleteBySourceURI(ctx, "file:///x.go"); err != nil || n != 2 {
		t.Fatalf("DeleteBySourceURI = %d, %v; want both nodes, including the unindexed one", n, err)
	}
}

// TestCleanCloseIsSealed checks that a reopen after a clean close, writable or read-only, skips the reindex.
func TestCleanCloseIsSealed(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "graph.badger")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertNodes(context.Background(), []model.Node{{ID: "a", Kind: model.KindFile}}); err != nil {
		t.Fatal(err)
	}
	if err := gs.Close(); err != nil {
		t.Fatal(err)
	}
	for _, open := range []func(string) (*Store, error){Open, OpenReadOnly, Open} {
		s, err := open(dir)
		if err != nil {
			t.Fatal(err)
		}
		ok, err := sealed(s.db)
		if cerr := s.Close(); err == nil {
			err = cerr
		}
		if err != nil || !ok {
			t.Fatalf("sealed after a clean close (read-only %v) = %v, %v; want true", s.readOnly, ok, err)
		}
	}
}

func TestStoredStringMatchesJSON(t *testing.T) {
	for _, s := range []string{"plain", "é", "bad\xffutf8", "two\xff\xfebad", "trunc\xc3", "mix\xe2\x82"} {
		b, err := json.Marshal(s)
		if err != nil {
			t.Fatal(err)
		}
		var back string
		if err := json.Unmarshal(b, &back); err != nil {
			t.Fatal(err)
		}
		if got := storedString(s); got != back {
			t.Errorf("storedString(%q) = %q, want the JSON round trip %q", s, got, back)
		}
	}
}

// BenchmarkDeleteBySourceURI deletes (and, untimed, restores) one 4-node file's nodes among n; with the index
// its cost no longer grows with n.
func BenchmarkDeleteBySourceURI(b *testing.B) {
	for _, n := range []int{2_000, 20_000} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			ctx := context.Background()
			gs, err := Open(filepath.Join(b.TempDir(), "graph.badger"))
			if err != nil {
				b.Fatal(err)
			}
			defer gs.Close()
			nodes := make([]model.Node, n)
			var target []model.Node
			files := n / 4
			for i := range nodes {
				nodes[i] = model.Node{ID: fmt.Sprintf("%032d", i), Kind: model.KindSymbol, SourceURI: fmt.Sprintf("file:///f%d.go", i%files)}
				if i%files == 7 {
					target = append(target, nodes[i])
				}
			}
			if err := gs.UpsertNodes(ctx, nodes); err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for range b.N {
				if _, err := gs.DeleteBySourceURI(ctx, "file:///f7.go"); err != nil {
					b.Fatal(err)
				}
				b.StopTimer()
				if err := gs.UpsertNodes(ctx, target); err != nil {
					b.Fatal(err)
				}
				b.StartTimer()
			}
		})
	}
}
