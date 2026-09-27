package store_test

import (
	"context"
	"errors"
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
