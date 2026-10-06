package badger

import (
	"bytes"
	"context"
	"encoding/json"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	badgerdb "github.com/dgraph-io/badger/v4"

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

func openEmbStore(t *testing.T) *Store {
	t.Helper()
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	return gs
}

// TestEmbeddingMetadataRoundTrip checks model and text hash survive storage, and EmbeddingMetas reports them with
// the dimension without decoding the vector.
func TestEmbeddingMetadataRoundTrip(t *testing.T) {
	gs, ctx := openEmbStore(t), context.Background()
	sha := model.ContentSHA([]byte("Retry"))
	want := store.Embedding{NodeID: "a", Vec: []float32{0.25, -1, 3}, Model: "nomic-embed-text", TextSHA: sha}
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{want}); err != nil {
		t.Fatal(err)
	}
	got, err := gs.AllEmbeddings(ctx)
	if err != nil || len(got) != 1 || got[0].Model != want.Model || got[0].TextSHA != sha || !slices.Equal(got[0].Vec, want.Vec) {
		t.Fatalf("AllEmbeddings = %+v, %v; want %+v", got, err, want)
	}
	metas, err := gs.EmbeddingMetas(ctx)
	if err != nil || !slices.Equal(metas, []store.EmbeddingMeta{{NodeID: "a", Model: "nomic-embed-text", Dim: 3, TextSHA: sha}}) {
		t.Fatalf("EmbeddingMetas = %+v, %v", metas, err)
	}
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "b", Vec: []float32{1}, Model: strings.Repeat("x", 256)}}); err == nil {
		t.Error("want an error for a model id over 255 bytes")
	}
}

// TestLegacyEmbeddingDecodesAsUnknownModel checks a bare vector written before metadata existed still reads.
func TestLegacyEmbeddingDecodesAsUnknownModel(t *testing.T) {
	gs, ctx := openEmbStore(t), context.Background()
	raw := make([]byte, 12)
	putVec(raw, []float32{1, 2, 3})
	if err := gs.db.Update(func(txn *badgerdb.Txn) error { return txn.Set(embKey("old"), raw) }); err != nil {
		t.Fatal(err)
	}
	got, err := gs.AllEmbeddings(ctx)
	if err != nil || len(got) != 1 || got[0].Model != "" || got[0].TextSHA != "" || !slices.Equal(got[0].Vec, []float32{1, 2, 3}) {
		t.Fatalf("legacy AllEmbeddings = %+v, %v", got, err)
	}
	metas, err := gs.EmbeddingMetas(ctx)
	if err != nil || len(metas) != 1 || metas[0].Model != "" || metas[0].Dim != 3 {
		t.Fatalf("legacy EmbeddingMetas = %+v, %v", metas, err)
	}
}

func TestDeleteEmbeddingsKeepsNodes(t *testing.T) {
	gs, ctx := openEmbStore(t), context.Background()
	if err := gs.UpsertNodes(ctx, []model.Node{{ID: "n1", Kind: model.KindSymbol, Label: "Foo", SourceURI: "file://x.go"}}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "n1", Vec: []float32{1}}, {NodeID: "n2", Vec: []float32{2}}}); err != nil {
		t.Fatal(err)
	}
	if err := gs.DeleteEmbeddings(ctx, []string{"n1", "missing"}); err != nil {
		t.Fatal(err)
	}
	if got, _ := gs.AllEmbeddings(ctx); len(got) != 1 || got[0].NodeID != "n2" {
		t.Errorf("embeddings after delete = %+v, want only n2", got)
	}
	if n, _ := gs.GetNode(ctx, "n1"); n == nil {
		t.Error("DeleteEmbeddings removed the node itself")
	}
}

func TestSnapshotDecodesTaggedEmbedding(t *testing.T) {
	gs, ctx := openEmbStore(t), context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{{ID: "n1", Kind: model.KindSymbol, Label: "Foo", SourceURI: "file://x.go"}})
	_ = gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "n1", Vec: []float32{0.5, -0.5}, Model: "m", TextSHA: model.ContentSHA([]byte("Foo"))}})
	var buf bytes.Buffer
	if err := gs.Snapshot(ctx, &buf); err != nil {
		t.Fatal(err)
	}
	var snap struct {
		Nodes []model.Node `json:"nodes"`
	}
	if err := json.Unmarshal(buf.Bytes(), &snap); err != nil {
		t.Fatal(err)
	}
	if len(snap.Nodes) != 1 || !slices.Equal(snap.Nodes[0].Embedding, []float32{0.5, -0.5}) {
		t.Fatalf("snapshot embedding = %+v, want [0.5 -0.5]", snap.Nodes)
	}
}
