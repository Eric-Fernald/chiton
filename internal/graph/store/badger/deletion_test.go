package badger

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

func TestDeleteByIDsRemovesNodesEdgesAndEmbeddings(t *testing.T) {
	ctx := context.Background()
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	nodes := []model.Node{
		{ID: "a", Kind: model.KindFinding, Label: "a", SourceURI: "finding://a"},
		{ID: "b", Kind: model.KindFile, Label: "b", SourceURI: "file://b"},
		{ID: "c", Kind: model.KindFile, Label: "c", SourceURI: "file://c"},
	}
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{
		{ID: "ab", Src: "a", Dst: "b", Kind: model.EdgeAffects},
		{ID: "cb", Src: "c", Dst: "b", Kind: model.EdgeReferences},
	}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "a", Vec: []float32{1, 2, 3}}}); err != nil {
		t.Fatal(err)
	}

	// Delete "a" (and a non-existent id, which must be skipped silently).
	n, err := gs.DeleteByIDs(ctx, []string{"a", "missing"})
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("DeleteByIDs deleted %d, want 1", n)
	}

	if got, _ := gs.GetNode(ctx, "a"); got != nil {
		t.Fatal("node a should be gone")
	}
	if got, _ := gs.GetNode(ctx, "b"); got == nil {
		t.Fatal("node b should remain")
	}
	// The a→b edge must be gone; the c→b edge must remain.
	edges, _ := gs.AllEdges(ctx)
	if len(edges) != 1 || edges[0].ID != "cb" {
		t.Fatalf("edges after delete = %+v, want only cb", edges)
	}
	embs, _ := gs.AllEmbeddings(ctx)
	if len(embs) != 0 {
		t.Fatalf("embedding for a should be gone, got %d", len(embs))
	}
}

func TestReadOnlyReflectsOpenMode(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "graph.badger")
	rw, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if rw.ReadOnly() {
		t.Fatal("Open should not be read-only")
	}
	rw.Close()

	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ro.Close()
	if !ro.ReadOnly() {
		t.Fatal("OpenReadOnly should report read-only")
	}
}
