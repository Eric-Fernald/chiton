package retrieve

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/embed"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
	"github.com/Eric-Fernald/chiton/internal/vector"
)

// buildEngine wires a temp Badger store, a hash embedder, and a flat index
// over the embeddable nodes, returning a ready Engine.
func buildEngine(t *testing.T, nodes []model.Node, edges []model.Edge) *Engine {
	t.Helper()
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })

	ctx := context.Background()
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	if len(edges) > 0 {
		if err := gs.UpsertEdges(ctx, edges); err != nil {
			t.Fatal(err)
		}
	}

	e := embed.NewHashEmbedder(512)
	idx := vector.NewFlat()
	for _, n := range nodes {
		text := embed.TextFor(n)
		if text == "" {
			continue
		}
		vecs, err := e.Embed(ctx, []string{text})
		if err != nil {
			t.Fatal(err)
		}
		idx.Add([]vector.Item{{ID: n.ID, Vec: vecs[0]}})
	}
	return New(gs, idx, e, nil)
}

func sym(id, label, doc string) model.Node {
	return model.Node{
		ID: id, Kind: model.KindSymbol, Label: label,
		SourceURI:  "file://" + id + ".go",
		Attrs:      map[string]any{"docstring": doc},
		Provenance: model.ProvExtracted, Confidence: 1,
	}
}

func TestQueryRanksSemanticMatchFirst(t *testing.T) {
	nodes := []model.Node{
		sym("login", "Login", "authenticate users with jwt tokens and bcrypt password hashing"),
		sym("bake", "Bake", "bake a chocolate cake in the oven"),
		sym("ship", "Ship", "ship a package via the postal service"),
	}
	e := buildEngine(t, nodes, nil)

	res, err := e.Query(context.Background(), "how do we authenticate a user login", Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Nodes) == 0 {
		t.Fatal("expected results")
	}
	if res.Nodes[0].Node.Label != "Login" {
		t.Fatalf("top result = %q, want Login\nfull: %+v", res.Nodes[0].Node.Label, res.Nodes)
	}
}

func TestQueryExpandsGraphNeighbors(t *testing.T) {
	// Login (semantic seed) CALLS hashPassword (no semantic overlap with query).
	nodes := []model.Node{
		sym("login", "Login", "authenticate users at sign in"),
		sym("hash", "hashPassword", "xyzzy plugh frobnicate"),
	}
	edges := []model.Edge{{
		ID:  model.NewEdgeID("login", "hash", model.EdgeCalls),
		Src: "login", Dst: "hash", Kind: model.EdgeCalls,
		Provenance: model.ProvExtracted, Confidence: 1,
	}}
	e := buildEngine(t, nodes, edges)

	// SeedK=1 so only Login seeds; hashPassword must arrive via graph expansion.
	res, err := e.Query(context.Background(), "authenticate users at sign in", Options{SeedK: 1, Hops: 1})
	if err != nil {
		t.Fatal(err)
	}

	var seedHop, neighborHop = -1, -1
	for _, s := range res.Nodes {
		switch s.Node.Label {
		case "Login":
			seedHop = s.Hop
		case "hashPassword":
			neighborHop = s.Hop
		}
	}
	if seedHop != 0 {
		t.Fatalf("seed hop = %d, want 0", seedHop)
	}
	if neighborHop != 1 {
		t.Fatalf("neighbor hop = %d, want 1 (graph expansion failed)", neighborHop)
	}
}

func TestQueryExcludesAmbiguousByDefault(t *testing.T) {
	concept := model.Node{
		ID: "c1", Kind: model.KindConcept, Label: "authentication strategy",
		SourceURI:  "concept://c1",
		Attrs:      map[string]any{"description": "authenticate users at login with tokens"},
		Provenance: model.ProvAmbiguous, Confidence: 0.3,
	}
	nodes := []model.Node{
		sym("login", "Login", "authenticate users at login with tokens"),
		concept,
	}
	e := buildEngine(t, nodes, nil)

	def, err := e.Query(context.Background(), "authenticate users at login with tokens", Options{})
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range def.Nodes {
		if s.Node.Provenance == model.ProvAmbiguous {
			t.Fatal("AMBIGUOUS node returned without IncludeAmbiguous")
		}
	}

	inc, err := e.Query(context.Background(), "authenticate users at login with tokens", Options{IncludeAmbiguous: true})
	if err != nil {
		t.Fatal(err)
	}
	var found bool
	for _, s := range inc.Nodes {
		if s.Node.Provenance == model.ProvAmbiguous {
			found = true
		}
	}
	if !found {
		t.Fatal("AMBIGUOUS node missing even with IncludeAmbiguous")
	}
}

func TestQueryEmptyIndex(t *testing.T) {
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	e := New(gs, vector.NewFlat(), embed.NewHashEmbedder(64), nil)

	res, err := e.Query(context.Background(), "anything", Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Nodes) != 0 {
		t.Fatalf("expected no nodes from empty index, got %d", len(res.Nodes))
	}
}

func TestQueryTokenBudgetTrims(t *testing.T) {
	var nodes []model.Node
	for i := 0; i < 40; i++ {
		id := "n" + string(rune('a'+i%26)) + string(rune('a'+i/26))
		nodes = append(nodes, sym(id, "auth handler", "authenticate users at login with tokens and sessions"))
	}
	e := buildEngine(t, nodes, nil)

	res, err := e.Query(context.Background(), "authenticate users at login", Options{SeedK: 40, MaxNodes: 40, MaxTokens: 200})
	if err != nil {
		t.Fatal(err)
	}
	// 200 tokens ≈ 800 chars; each node JSON is well over 100 chars, so the
	// budget must cut well below 40.
	if len(res.Nodes) >= 40 {
		t.Fatalf("token budget did not trim: got %d nodes", len(res.Nodes))
	}
	if len(res.Nodes) == 0 {
		t.Fatal("budget trimmed everything; should keep at least one")
	}
}
