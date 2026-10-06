package badger

import (
	"context"
	"path/filepath"
	"slices"
	"testing"

	badgerdb "github.com/dgraph-io/badger/v4"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// pathIDs returns p's node and edge IDs in path order.
func pathIDs(p store.Path) (nodes, edges []string) {
	for _, n := range p.Nodes {
		nodes = append(nodes, n.ID)
	}
	for _, e := range p.Edges {
		edges = append(edges, e.ID)
	}
	return nodes, edges
}

// pathStore holds a-b-c-d (c→b points backwards), a side edge a→s, and an isolated z.
func pathStore(t *testing.T) *Store {
	t.Helper()
	ctx := context.Background()
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	var nodes []model.Node
	for _, id := range []string{"a", "b", "c", "d", "s", "z"} {
		nodes = append(nodes, model.Node{ID: id, Label: id})
	}
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{
		{ID: "ab", Src: "a", Dst: "b", Kind: model.EdgeCalls},
		{ID: "cb", Src: "c", Dst: "b", Kind: model.EdgeReferences},
		{ID: "cd", Src: "c", Dst: "d", Kind: model.EdgeCalls},
		{ID: "as", Src: "a", Dst: "s", Kind: model.EdgeContains},
	}); err != nil {
		t.Fatal(err)
	}
	return gs
}

func TestShortestPath(t *testing.T) {
	ctx := context.Background()
	gs := pathStore(t)
	for _, tc := range []struct {
		src, dst             string
		wantNodes, wantEdges []string
	}{
		{"a", "d", []string{"a", "b", "c", "d"}, []string{"ab", "cb", "cd"}},
		{"d", "a", []string{"d", "c", "b", "a"}, []string{"cd", "cb", "ab"}},
		{"a", "a", []string{"a"}, nil},
		{"a", "z", nil, nil},       // unreachable
		{"a", "missing", nil, nil}, // unknown node
	} {
		p, err := gs.ShortestPath(ctx, tc.src, tc.dst)
		if err != nil {
			t.Fatalf("ShortestPath(%s, %s): %v", tc.src, tc.dst, err)
		}
		nodes, edges := pathIDs(p)
		if !slices.Equal(nodes, tc.wantNodes) || !slices.Equal(edges, tc.wantEdges) {
			t.Errorf("ShortestPath(%s, %s) = %v %v, want %v %v", tc.src, tc.dst, nodes, edges, tc.wantNodes, tc.wantEdges)
		}
	}
}

func TestShortestPathDecodesOnlyPathEdges(t *testing.T) {
	ctx := context.Background()
	gs := pathStore(t)
	// Corrupt the side edge a→s, which the a→d search passes over.
	if err := gs.db.Update(func(txn *badgerdb.Txn) error { return txn.Set(edgeKey("as"), []byte("{")) }); err != nil {
		t.Fatal(err)
	}
	p, err := gs.ShortestPath(ctx, "a", "d")
	if err != nil {
		t.Fatalf("ShortestPath decoded an edge off the path: %v", err)
	}
	if nodes, _ := pathIDs(p); !slices.Equal(nodes, []string{"a", "b", "c", "d"}) {
		t.Fatalf("path = %v, want a b c d", nodes)
	}
}
