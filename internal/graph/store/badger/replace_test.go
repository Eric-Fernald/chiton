package badger

import (
	"context"
	"path/filepath"
	"slices"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

func openTestStore(t *testing.T) *Store {
	t.Helper()
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	return gs
}

func edgeIDs(t *testing.T, gs *Store) []string {
	t.Helper()
	edges, err := gs.AllEdges(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, e := range edges {
		ids = append(ids, e.ID)
	}
	slices.Sort(ids)
	return ids
}

const fileURI = "file:///a.go"

func sym(id, label, doc string) model.Node {
	return model.Node{ID: id, Kind: model.KindSymbol, Label: label, SourceURI: fileURI,
		Provenance: model.ProvExtracted, Attrs: map[string]any{"docstring": doc}}
}

func owned(id, src, dst string, kind model.EdgeKind) model.Edge {
	return model.Edge{ID: id, Src: src, Dst: dst, Kind: kind, Provenance: model.ProvExtracted}
}

var fileNode = model.Node{ID: "f", Kind: model.KindFile, Label: "a.go", SourceURI: fileURI, Provenance: model.ProvExtracted}

// TestReplaceSourcesKeepsForeignEdges replays edit-then-rename: an edit keeps edges other sources attached
// to a surviving symbol and drops the owned edge it no longer emits; a rename drops the symbol and its edges.
func TestReplaceSourcesKeepsForeignEdges(t *testing.T) {
	ctx := context.Background()
	gs := openTestStore(t)
	first := store.SourceUpdate{URI: fileURI,
		Nodes: []model.Node{fileNode, sym("foo", "Foo", "v1"), sym("bar", "Bar", "v1")},
		Edges: []model.Edge{owned("f-foo", "f", "foo", model.EdgeContains), owned("f-bar", "f", "bar", model.EdgeContains),
			owned("foo-bar", "foo", "bar", model.EdgeCalls)},
	}
	if _, err := gs.ReplaceSources(ctx, []store.SourceUpdate{first}); err != nil {
		t.Fatal(err)
	}
	// Meeting linkage attaches a decision and a meeting to Foo; both symbols have vectors.
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "dec", Kind: model.KindDecision, Label: "keep Foo", SourceURI: "decision://meeting://m"},
		{ID: "mtg", Kind: model.KindMeeting, Label: "standup", SourceURI: "meeting://m"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{
		{ID: "dec-foo", Src: "dec", Dst: "foo", Kind: model.EdgeDecidedIn, Provenance: model.ProvInferred},
		{ID: "foo-mtg", Src: "foo", Dst: "mtg", Kind: model.EdgeDiscussedIn, Provenance: model.ProvInferred},
	}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "foo", Vec: []float32{1}}, {NodeID: "bar", Vec: []float32{1}}}); err != nil {
		t.Fatal(err)
	}

	// Edit: Foo is unchanged but no longer calls Bar, and Bar's docstring changes.
	edit := store.SourceUpdate{URI: fileURI,
		Nodes: []model.Node{fileNode, sym("foo", "Foo", "v1"), sym("bar", "Bar", "v2")},
		Edges: first.Edges[:2],
	}
	if _, err := gs.ReplaceSources(ctx, []store.SourceUpdate{edit}); err != nil {
		t.Fatal(err)
	}
	if got, want := edgeIDs(t, gs), []string{"dec-foo", "f-bar", "f-foo", "foo-mtg"}; !slices.Equal(got, want) {
		t.Fatalf("edges after edit = %v, want %v", got, want)
	}
	if got, _ := gs.EmbeddingIDs(ctx); !slices.Equal(got, []string{"foo"}) {
		t.Fatalf("embeddings after edit = %v, want only unchanged foo's", got)
	}

	// Rename Foo away: it goes, taking every edge attached to it.
	rename := store.SourceUpdate{URI: fileURI, Nodes: []model.Node{fileNode, sym("bar", "Bar", "v2")}, Edges: first.Edges[1:2]}
	deleted, err := gs.ReplaceSources(ctx, []store.SourceUpdate{rename})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(deleted, []int{1}) {
		t.Errorf("deleted = %v, want [1]", deleted)
	}
	if got := edgeIDs(t, gs); !slices.Equal(got, []string{"f-bar"}) {
		t.Fatalf("edges after rename = %v, want [f-bar]", got)
	}
	for _, id := range []string{"dec", "mtg"} {
		if n, _ := gs.GetNode(ctx, id); n == nil {
			t.Errorf("other source's node %s was deleted", id)
		}
	}
}

// TestReplaceSourcesCarriesStatusOfUnchangedNodes checks an unchanged node keeps extraction_status and a
// changed one loses it.
func TestReplaceSourcesCarriesStatusOfUnchangedNodes(t *testing.T) {
	ctx := context.Background()
	gs := openTestStore(t)
	sec := func(text string) model.Node {
		return model.Node{ID: "s", Kind: model.KindDocSection, Label: "Intro", SourceURI: fileURI,
			Attrs: map[string]any{"text": text, "level": 1}}
	}
	replace := func(n model.Node) {
		t.Helper()
		if _, err := gs.ReplaceSources(ctx, []store.SourceUpdate{{URI: fileURI, Nodes: []model.Node{fileNode, n}}}); err != nil {
			t.Fatal(err)
		}
	}
	status := func() string {
		t.Helper()
		n, err := gs.GetNode(ctx, "s")
		if err != nil || n == nil {
			t.Fatalf("GetNode(s) = %v, %v", n, err)
		}
		return model.AttrString(n.Attrs, "extraction_status")
	}
	replace(sec("hello"))
	done := sec("hello")
	done.Attrs["extraction_status"] = "done"
	if err := gs.UpsertNodes(ctx, []model.Node{done}); err != nil {
		t.Fatal(err)
	}
	replace(sec("hello"))
	if got := status(); got != "done" {
		t.Errorf("unchanged section status = %q, want done", got)
	}
	replace(sec("rewritten"))
	if got := status(); got != "" {
		t.Errorf("changed section status = %q, want it cleared", got)
	}
}
