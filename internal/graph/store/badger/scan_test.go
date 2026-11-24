package badger

import (
	"context"
	"fmt"
	"path/filepath"
	"slices"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// TestNodesMatchingEqualsFilteredAllNodes checks the raw-bytes probe never drops a match, including values
// whose JSON escaping is non-trivial.
func TestNodesMatchingEqualsFilteredAllNodes(t *testing.T) {
	ctx := context.Background()
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	vals := []string{"file:///a.go", "file:///a.go.bak", "a&b<c>", `C:\x`, "é", `q"uote`, "bad\xffutf8", "bad\ufffdutf8",
		"line\nbreak", "tab\there", "\b\f", "sep\u2028x", "", "finding", "meeting"}
	kinds := []model.NodeKind{model.KindFinding, model.KindMeeting, model.KindSymbol, "odd<kind>", "ctl\x01kind", ""}
	var nodes []model.Node
	for i, v := range vals {
		for j, k := range kinds {
			attrs := map[string]any{"target_uri": v, "n": i}
			if j%2 == 0 {
				attrs = map[string]any{"other": v} // leaves target_uri absent
			}
			nodes = append(nodes, model.Node{ID: fmt.Sprint("n", i, "_", j), Kind: k, Label: v, SourceURI: v, Attrs: attrs})
		}
	}
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	all, err := gs.AllNodes(ctx)
	if err != nil {
		t.Fatal(err)
	}

	var filters []store.NodeFilter
	filters = append(filters, store.NodeFilter{}, store.NodeFilter{Kinds: kinds}, store.NodeFilter{Kinds: []model.NodeKind{"none"}})
	for _, k := range kinds {
		filters = append(filters, store.NodeFilter{Kinds: []model.NodeKind{k}}, store.NodeFilter{Kinds: []model.NodeKind{k, model.KindMeeting}})
	}
	for _, v := range append(vals, "bad\xffutf8", "missing") {
		filters = append(filters,
			store.NodeFilter{SourceURI: v},
			store.NodeFilter{Attrs: map[string]string{"target_uri": v}},
			store.NodeFilter{Kinds: []model.NodeKind{model.KindFinding}, Attrs: map[string]string{"target_uri": v, "other": ""}},
			store.NodeFilter{Kinds: []model.NodeKind{model.KindMeeting, "ctl\x01kind"}, SourceURI: v, Attrs: map[string]string{"other": v}},
		)
	}
	for _, f := range filters {
		var want []model.Node
		for _, n := range all {
			if f.Match(n) {
				want = append(want, n)
			}
		}
		got, err := gs.NodesMatching(ctx, f)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.EqualFunc(got, want, func(a, b model.Node) bool { return a.ID == b.ID }) {
			t.Errorf("NodesMatching(%+v) = %d nodes, want %d", f, len(got), len(want))
		}
	}
}

func TestStringProbeRejectsUnstableEscapes(t *testing.T) {
	for _, s := range []string{"bad\xffutf8", "bad\ufffdutf8", "a\nb", "\x01", "\b"} {
		if p := stringProbe(s); p != nil {
			t.Errorf("stringProbe(%q) = %q, want nil", s, p)
		}
	}
	if got := string(stringProbe(`a<b>&"c`)); got != `"a\u003cb\u003e\u0026\"c"` {
		t.Errorf("stringProbe escaping = %s", got)
	}
}

func TestEmbeddingIDs(t *testing.T) {
	ctx := context.Background()
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	if ids, err := gs.EmbeddingIDs(ctx); err != nil || len(ids) != 0 {
		t.Fatalf("empty store EmbeddingIDs = %v, %v", ids, err)
	}
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "b", Vec: []float32{1}}, {NodeID: "a", Vec: []float32{2, 3}}}); err != nil {
		t.Fatal(err)
	}
	ids, err := gs.EmbeddingIDs(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ids, []string{"a", "b"}) {
		t.Fatalf("EmbeddingIDs = %v, want [a b]", ids)
	}
}
