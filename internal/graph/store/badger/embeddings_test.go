package badger

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

func TestEmbeddingRoundTrip(t *testing.T) {
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	ctx := context.Background()
	want := []store.Embedding{
		{NodeID: "a", Vec: []float32{0.1, -0.2, 0.3}},
		{NodeID: "b", Vec: []float32{1, 2, 3, 4}},
	}
	if err := gs.UpsertEmbeddings(ctx, want); err != nil {
		t.Fatal(err)
	}

	got, err := gs.AllEmbeddings(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d embeddings, want 2", len(got))
	}
	byID := map[string][]float32{}
	for _, e := range got {
		byID[e.NodeID] = e.Vec
	}
	for _, w := range want {
		gv := byID[w.NodeID]
		if len(gv) != len(w.Vec) {
			t.Fatalf("node %s dim = %d, want %d", w.NodeID, len(gv), len(w.Vec))
		}
		for i := range w.Vec {
			if gv[i] != w.Vec[i] {
				t.Fatalf("node %s [%d] = %v, want %v", w.NodeID, i, gv[i], w.Vec[i])
			}
		}
	}
}

func TestUpsertEmbeddingsReplaces(t *testing.T) {
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	ctx := context.Background()
	_ = gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "a", Vec: []float32{1, 0}}})
	_ = gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "a", Vec: []float32{0, 1}}})

	got, _ := gs.AllEmbeddings(ctx)
	if len(got) != 1 {
		t.Fatalf("got %d, want 1 after replace", len(got))
	}
	if got[0].Vec[0] != 0 || got[0].Vec[1] != 1 {
		t.Fatalf("embedding not replaced: %v", got[0].Vec)
	}
}

func TestDeleteBySourceURIClearsEmbedding(t *testing.T) {
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	ctx := context.Background()
	n := model.Node{ID: "n1", Kind: model.KindSymbol, Label: "Foo", SourceURI: "file://x.go"}
	if err := gs.UpsertNodes(ctx, []model.Node{n}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "n1", Vec: []float32{1, 2}}}); err != nil {
		t.Fatal(err)
	}

	if _, err := gs.DeleteBySourceURI(ctx, "file://x.go"); err != nil {
		t.Fatal(err)
	}
	got, _ := gs.AllEmbeddings(ctx)
	if len(got) != 0 {
		t.Fatalf("expected embedding cleared after node delete, got %d", len(got))
	}
}

func TestSnapshotIncludesEmbedding(t *testing.T) {
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	ctx := context.Background()
	n := model.Node{ID: "n1", Kind: model.KindSymbol, Label: "Foo", SourceURI: "file://x.go"}
	_ = gs.UpsertNodes(ctx, []model.Node{n})
	_ = gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "n1", Vec: []float32{0.5, 0.5}}})

	var buf bytes.Buffer
	if err := gs.Snapshot(ctx, &buf); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), `"embedding"`) {
		t.Fatalf("snapshot missing embedding field:\n%s", buf.String())
	}
}
