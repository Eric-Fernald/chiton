package store_test

import (
	"context"
	"errors"
	"maps"
	"path/filepath"
	"slices"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

// plain hides every optional capability of the wrapped store, forcing the helpers' fallbacks.
type plain struct{ store.GraphStore }

type plainEmb struct{ store.EmbeddingStore }

func openStore(t *testing.T) *badger.Store {
	t.Helper()
	gs, err := badger.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	return gs
}

func TestPickByFile(t *testing.T) {
	nodes := []model.Node{
		{ID: "0", Kind: model.KindFile, SourceURI: "file:///a/fetch.go"},
		{ID: "1", Kind: model.KindFile, SourceURI: "file:///b/FetchAll.go"},
		{ID: "2", Kind: model.KindSymbol, SourceURI: "file:///b/FetchAll.go"},
		{ID: "3", Kind: model.KindSymbol, SourceURI: "file:///c/other.go"},
	}
	for _, tc := range []struct{ file, want string }{
		{"", "0"},            // no hint picks the first node
		{"fetchall.go", "2"}, // case-insensitive, symbol preferred over the file node
		{"fetch.go", "0"},    // only a file node matches
		{"nomatch", "0"},     // no match falls back to the first node
		{"other", "3"},
	} {
		if got := store.PickByFile(nodes, tc.file).ID; got != tc.want {
			t.Errorf("PickByFile(%q) = %s, want %s", tc.file, got, tc.want)
		}
	}
}

func TestFindNode(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "f", Kind: model.KindFile, Label: "fetchall.go", SourceURI: "file:///x/fetchall.go"},
		{ID: "s", Kind: model.KindSymbol, Label: "fetch", SourceURI: "file:///x/fetchall.go"},
	}); err != nil {
		t.Fatal(err)
	}
	if n, found, err := store.FindNode(ctx, gs, "FETCH", ""); err != nil || !found || n.ID != "f" {
		t.Errorf("FindNode no hint = %s, %v, %v; want f (first match)", n.ID, found, err)
	}
	if n, found, err := store.FindNode(ctx, gs, "fetch", "fetchall"); err != nil || !found || n.ID != "s" {
		t.Errorf("FindNode with hint = %s, %v, %v; want s", n.ID, found, err)
	}
	if n, found, err := store.FindNode(ctx, gs, "zzz", ""); err != nil || found || n.ID != "" {
		t.Errorf("FindNode no match = %+v, %v, %v", n, found, err)
	}
}

type failingSearch struct{ store.GraphStore }

func (failingSearch) SearchNodes(context.Context, string, int) ([]model.Node, error) {
	return nil, errors.New("boom")
}

func TestFindNodeSearchError(t *testing.T) {
	if _, found, err := store.FindNode(context.Background(), failingSearch{}, "x", ""); found || err == nil {
		t.Fatalf("FindNode = found %v, err %v; want the search error", found, err)
	}
}

func TestIsReadOnly(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "graph.badger")
	rw, err := badger.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if store.IsReadOnly(rw) {
		t.Error("read-write store reported read-only")
	}
	rw.Close()
	ro, err := badger.OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ro.Close()
	if !store.IsReadOnly(ro) {
		t.Error("read-only store not reported read-only")
	}
	if store.IsReadOnly(plain{ro}) {
		t.Error("store without ReadOnlyAware reported read-only")
	}
}

func TestNodesMatchingFallbackAgrees(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "1", Kind: model.KindFinding, SourceURI: "finding://1", Attrs: map[string]any{"target_uri": "file:///a.go"}},
		{ID: "2", Kind: model.KindFinding, SourceURI: "finding://2"},
		{ID: "3", Kind: model.KindMeeting, SourceURI: "meeting://x", Attrs: map[string]any{"target_uri": "file:///a.go"}},
		{ID: "4", Kind: model.KindDecision, SourceURI: "meeting://x"},
	}); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		f    store.NodeFilter
		want []string
	}{
		{store.NodeFilter{}, []string{"1", "2", "3", "4"}},
		{store.NodeFilter{Kinds: []model.NodeKind{model.KindFinding}}, []string{"1", "2"}},
		{store.NodeFilter{Kinds: []model.NodeKind{model.KindDecision, model.KindMeeting}, SourceURI: "meeting://x"}, []string{"3", "4"}},
		{store.NodeFilter{Attrs: map[string]string{"target_uri": "file:///a.go"}}, []string{"1", "3"}},
		{store.NodeFilter{Kinds: []model.NodeKind{model.KindFinding}, Attrs: map[string]string{"target_uri": ""}}, []string{"2"}},
	} {
		for name, s := range map[string]store.GraphStore{"scanner": gs, "fallback": plain{gs}} {
			got, err := store.NodesMatching(ctx, s, tc.f)
			if err != nil {
				t.Fatal(err)
			}
			var ids []string
			for _, n := range got {
				ids = append(ids, n.ID)
			}
			if !slices.Equal(ids, tc.want) {
				t.Errorf("%s NodesMatching(%+v) = %v, want %v", name, tc.f, ids, tc.want)
			}
		}
	}
}

func TestEmbeddingIDsFallbackAgrees(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "b", Vec: []float32{1}}, {NodeID: "a", Vec: []float32{2}}}); err != nil {
		t.Fatal(err)
	}
	for name, es := range map[string]store.EmbeddingStore{"lister": gs, "fallback": plainEmb{gs}} {
		ids, err := store.EmbeddingIDs(ctx, es)
		if err != nil || !slices.Equal(ids, []string{"a", "b"}) {
			t.Errorf("%s EmbeddingIDs = %v, %v; want [a b]", name, ids, err)
		}
	}
}

func TestCountByKindFallbackAgrees(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "a", Kind: model.KindSymbol}, {ID: "b", Kind: model.KindFile}, {ID: "c", Kind: model.KindSymbol},
	}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{
		{ID: "ab", Src: "a", Dst: "b", Kind: model.EdgeCalls},
		{ID: "cb", Src: "c", Dst: "b", Kind: model.EdgeReferences},
	}); err != nil {
		t.Fatal(err)
	}
	wantNodes := map[model.NodeKind]int{model.KindSymbol: 2, model.KindFile: 1}
	wantEdges := map[model.EdgeKind]int{model.EdgeCalls: 1, model.EdgeReferences: 1}
	for name, s := range map[string]store.GraphStore{"counter": gs, "fallback": plain{gs}} {
		nodes, edges, err := store.CountByKind(ctx, s)
		if err != nil || !maps.Equal(nodes, wantNodes) || !maps.Equal(edges, wantEdges) {
			t.Errorf("%s CountByKind = %v %v %v, want %v %v", name, nodes, edges, err, wantNodes, wantEdges)
		}
	}
}

// TestReplaceSourcesBatch replaces two sources in one call, one of them Fresh, through SourceReplacer and through
// the DeleteBySourceURI fallback, which counts every prior node of the source as deleted, kept ones included.
func TestReplaceSourcesBatch(t *testing.T) {
	ctx := context.Background()
	const x, y = "file:///x.go", "file:///y.go"
	for name, tc := range map[string]struct {
		wrap        func(store.GraphStore) store.GraphStore
		wantDeleted []int
	}{
		"replacer": {func(gs store.GraphStore) store.GraphStore { return gs }, []int{1, 0}},
		"fallback": {func(gs store.GraphStore) store.GraphStore { return plain{gs} }, []int{2, 0}},
	} {
		t.Run(name, func(t *testing.T) {
			gs := openStore(t)
			s := tc.wrap(gs)
			if _, err := store.ReplaceSources(ctx, s, store.SourceUpdate{URI: x, Nodes: []model.Node{
				{ID: "xf", Kind: model.KindFile, SourceURI: x}, {ID: "old", Kind: model.KindSymbol, SourceURI: x},
			}}); err != nil {
				t.Fatal(err)
			}
			deleted, err := store.ReplaceSources(ctx, s,
				store.SourceUpdate{URI: x, Nodes: []model.Node{
					{ID: "xf", Kind: model.KindFile, SourceURI: x}, {ID: "new", Kind: model.KindSymbol, SourceURI: x},
				}},
				store.SourceUpdate{URI: y, Fresh: true,
					Nodes: []model.Node{{ID: "yf", Kind: model.KindFile, SourceURI: y}},
					Edges: []model.Edge{{ID: "yx", Src: "yf", Dst: "new", Kind: model.EdgeReferences}}},
			)
			if err != nil || !slices.Equal(deleted, tc.wantDeleted) {
				t.Fatalf("ReplaceSources = %v, %v; want %v", deleted, err, tc.wantDeleted)
			}
			nodes, err := gs.AllNodes(ctx)
			if err != nil {
				t.Fatal(err)
			}
			var ids []string
			for _, n := range nodes {
				ids = append(ids, n.ID)
			}
			slices.Sort(ids)
			if want := []string{"new", "xf", "yf"}; !slices.Equal(ids, want) {
				t.Errorf("nodes %v, want %v", ids, want)
			}
			if edges, err := gs.AllEdges(ctx); err != nil || len(edges) != 1 || edges[0].ID != "yx" {
				t.Errorf("edges %v, %v; want only yx", edges, err)
			}
		})
	}
}

func TestDeleteNodesFallbackAgrees(t *testing.T) {
	ctx := context.Background()
	for name, wrap := range map[string]func(*badger.Store) store.GraphStore{
		"deleter":  func(s *badger.Store) store.GraphStore { return s },
		"fallback": func(s *badger.Store) store.GraphStore { return plain{s} },
	} {
		gs := openStore(t)
		nodes := []model.Node{{ID: "1", SourceURI: "u://1"}, {ID: "2", SourceURI: "u://2"}, {ID: "3", SourceURI: "u://3"}}
		if err := gs.UpsertNodes(ctx, nodes); err != nil {
			t.Fatal(err)
		}
		if n, err := store.DeleteNodes(ctx, wrap(gs), nodes[:2]); err != nil || n != 2 {
			t.Errorf("%s DeleteNodes = %d, %v; want 2, nil", name, n, err)
		}
		if left, _ := gs.AllNodes(ctx); len(left) != 1 || left[0].ID != "3" {
			t.Errorf("%s left %v, want only node 3", name, left)
		}
	}
}

func TestEmbeddingMetasFallbackAgrees(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "b", Vec: []float32{1}, Model: "m"}, {NodeID: "a", Vec: []float32{2, 3}}}); err != nil {
		t.Fatal(err)
	}
	want := []store.EmbeddingMeta{{NodeID: "a", Dim: 2}, {NodeID: "b", Model: "m", Dim: 1}}
	for name, es := range map[string]store.EmbeddingStore{"lister": gs, "fallback": plainEmb{gs}} {
		if got, err := store.EmbeddingMetas(ctx, es); err != nil || !slices.Equal(got, want) {
			t.Errorf("%s EmbeddingMetas = %+v, %v; want %+v", name, got, err, want)
		}
	}
}
