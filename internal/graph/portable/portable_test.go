package portable

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

func newStore(t *testing.T) *badgerstore.Store {
	t.Helper()
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { gs.Close() })
	return gs
}

// seed populates src with two nodes (one carrying an embedding) and one edge,
// then returns it.
func seed(t *testing.T, ctx context.Context) *badgerstore.Store {
	t.Helper()
	src := newStore(t)
	a := model.Node{ID: model.NewNodeID("file://a.go", "A"), Kind: model.KindSymbol, Label: "A", SourceURI: "file://a.go", Provenance: model.ProvExtracted, Confidence: 1}
	b := model.Node{ID: model.NewNodeID("pkg://fmt", "fmt"), Kind: model.KindSymbol, Label: "fmt", SourceURI: "pkg://fmt", Attrs: map[string]any{"external": true}, Provenance: model.ProvExtracted, Confidence: 1}
	if err := src.UpsertNodes(ctx, []model.Node{a, b}); err != nil {
		t.Fatalf("upsert nodes: %v", err)
	}
	e := model.Edge{ID: model.NewEdgeID(a.ID, b.ID, model.EdgeImports), Src: a.ID, Dst: b.ID, Kind: model.EdgeImports, Provenance: model.ProvExtracted, Confidence: 1}
	if err := src.UpsertEdges(ctx, []model.Edge{e}); err != nil {
		t.Fatalf("upsert edges: %v", err)
	}
	if err := src.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: a.ID, Vec: []float32{0.1, 0.2, 0.3}}}); err != nil {
		t.Fatalf("upsert embeddings: %v", err)
	}
	return src
}

func TestSnapshotRoundTripMerge(t *testing.T) {
	ctx := context.Background()
	src := seed(t, ctx)

	var buf bytes.Buffer
	if err := src.Snapshot(ctx, &buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}

	snap, err := Read(&buf)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(snap.Nodes) != 2 || len(snap.Edges) != 1 {
		t.Fatalf("snapshot shape: got %d nodes, %d edges; want 2, 1", len(snap.Nodes), len(snap.Edges))
	}

	dst := newStore(t)
	stats, err := Merge(ctx, dst, snap, 0)
	if err != nil {
		t.Fatalf("merge: %v", err)
	}
	if stats.Nodes != 2 || stats.Edges != 1 || stats.Embeddings != 1 {
		t.Fatalf("merge stats: %+v; want 2 nodes, 1 edge, 1 embedding", stats)
	}

	// The embedding must land in the embedding keyspace, not inlined on the node.
	gotNodes, err := dst.AllNodes(ctx)
	if err != nil {
		t.Fatalf("all nodes: %v", err)
	}
	if len(gotNodes) != 2 {
		t.Fatalf("dst node count: got %d, want 2", len(gotNodes))
	}
	for _, n := range gotNodes {
		if len(n.Embedding) != 0 {
			t.Errorf("node %s carries an inlined embedding after merge (should live in the v: keyspace)", n.Label)
		}
	}
	embs, err := dst.AllEmbeddings(ctx)
	if err != nil {
		t.Fatalf("all embeddings: %v", err)
	}
	if len(embs) != 1 {
		t.Fatalf("dst embedding count: got %d, want 1", len(embs))
	}
}

func TestMergeIdempotent(t *testing.T) {
	ctx := context.Background()
	src := seed(t, ctx)
	var buf bytes.Buffer
	if err := src.Snapshot(ctx, &buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	snap, err := Read(&buf)
	if err != nil {
		t.Fatalf("read: %v", err)
	}

	dst := newStore(t)
	if _, err := Merge(ctx, dst, snap, 0); err != nil {
		t.Fatalf("merge 1: %v", err)
	}
	if _, err := Merge(ctx, dst, snap, 0); err != nil {
		t.Fatalf("merge 2: %v", err)
	}
	nodes, _ := dst.AllNodes(ctx)
	edges, _ := dst.AllEdges(ctx)
	if len(nodes) != 2 || len(edges) != 1 {
		t.Fatalf("after two merges: got %d nodes, %d edges; want 2, 1 (merge must be idempotent)", len(nodes), len(edges))
	}
}

func TestReadBOM(t *testing.T) {
	data := append([]byte{0xEF, 0xBB, 0xBF}, []byte(`{"nodes":[],"edges":[]}`)...)
	if _, err := Read(bytes.NewReader(data)); err != nil {
		t.Fatalf("read with BOM: %v", err)
	}
}
