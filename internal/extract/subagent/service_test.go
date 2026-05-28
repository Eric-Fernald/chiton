package subagent

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/embed"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

func newStore(t *testing.T) *badgerstore.Store {
	t.Helper()
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	return gs
}

func docSection(id, label, text string) model.Node {
	return model.Node{
		ID: id, Kind: model.KindDocSection, Label: label,
		SourceURI:  "file://doc.md",
		Attrs:      map[string]any{"text": text},
		Provenance: model.ProvExtracted, Confidence: 1,
	}
}

func TestPendingFindsUndoneDocAndImage(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	nodes := []model.Node{
		docSection("d1", "Auth", "we authenticate with JWTs"),
		docSection("d2", "Empty", ""), // no text → not a task
		{ID: "img1", Kind: model.KindImage, Label: "arch.png",
			SourceURI: "file://arch.png", Attrs: map[string]any{"path": "/tmp/arch.png"}},
		{ID: "done1", Kind: model.KindDocSection, Label: "Old",
			SourceURI: "file://doc.md",
			Attrs:     map[string]any{"text": "x", "extraction_status": "done"}},
	}
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}

	svc := New(gs, nil, 0)
	specs, err := svc.Pending(ctx, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, s := range specs {
		got[s.ID] = s.Kind
	}
	if len(got) != 2 {
		t.Fatalf("want 2 pending specs, got %d: %+v", len(got), specs)
	}
	if got["d1"] != KindDocConcepts {
		t.Fatalf("d1 kind = %q", got["d1"])
	}
	if got["img1"] != KindImageDescription {
		t.Fatalf("img1 kind = %q", got["img1"])
	}
	if _, ok := got["d2"]; ok {
		t.Fatal("empty section should not be a task")
	}
	if _, ok := got["done1"]; ok {
		t.Fatal("done section should not be a task")
	}

	// Kind filter narrows to one type.
	only, _ := svc.Pending(ctx, KindImageDescription, 0)
	if len(only) != 1 || only[0].ID != "img1" {
		t.Fatalf("kind filter failed: %+v", only)
	}
}

func TestSubmitWritesConceptsAndMarksDone(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "we authenticate with JWTs")})

	svc := New(gs, nil, 0.5)
	rep, err := svc.Submit(ctx, SubmitResult{
		TaskID: "d1",
		Concepts: []ConceptResult{
			{Label: "JWT", Description: "JSON Web Token", Confidence: 0.9},
			{Label: "Maybe", Description: "uncertain", Confidence: 0.2}, // below threshold → AMBIGUOUS
		},
		Edges: []EdgeResult{
			{From: "", To: "JWT", Kind: "DESCRIBES", Confidence: 0.9},
			{From: "JWT", To: "Maybe", Kind: "ELABORATES", Confidence: 0.8},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.ConceptsAdded != 2 || rep.EdgesAdded != 2 || rep.Ambiguous != 1 {
		t.Fatalf("unexpected report: %+v", rep)
	}

	all, _ := gs.AllNodes(ctx)
	var concepts, inferred, ambiguous int
	var doneMarked bool
	for _, n := range all {
		if n.Kind == model.KindConcept {
			concepts++
			switch n.Provenance {
			case model.ProvInferred:
				inferred++
			case model.ProvAmbiguous:
				ambiguous++
			}
		}
		if n.ID == "d1" && n.Attrs["extraction_status"] == "done" {
			doneMarked = true
		}
	}
	if concepts != 2 {
		t.Fatalf("want 2 concept nodes, got %d", concepts)
	}
	if inferred != 1 || ambiguous != 1 {
		t.Fatalf("provenance split wrong: inferred=%d ambiguous=%d", inferred, ambiguous)
	}
	if !doneMarked {
		t.Fatal("source node not marked extraction_status=done")
	}

	// After submit, the source is no longer pending.
	specs, _ := svc.Pending(ctx, "", 0)
	if len(specs) != 0 {
		t.Fatalf("expected no pending after submit, got %d", len(specs))
	}

	// Edges present with correct kinds.
	edges, _ := gs.AllEdges(ctx)
	kinds := map[model.EdgeKind]int{}
	for _, e := range edges {
		kinds[e.Kind]++
	}
	if kinds[model.EdgeDescribes] != 1 || kinds[model.EdgeElaborates] != 1 {
		t.Fatalf("edge kinds wrong: %+v", kinds)
	}
}

func TestSubmitIdempotent(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "jwt auth")})

	svc := New(gs, nil, 0.5)
	res := SubmitResult{
		TaskID:   "d1",
		Concepts: []ConceptResult{{Label: "JWT", Description: "token", Confidence: 0.9}},
	}
	if _, err := svc.Submit(ctx, res); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.Submit(ctx, res); err != nil {
		t.Fatal(err)
	}

	all, _ := gs.AllNodes(ctx)
	var concepts int
	for _, n := range all {
		if n.Kind == model.KindConcept {
			concepts++
		}
	}
	if concepts != 1 {
		t.Fatalf("idempotency broken: %d concept nodes after double submit", concepts)
	}
}

func TestSubmitDefaultsEdgesToDescribes(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "jwt auth")})

	svc := New(gs, nil, 0.5)
	rep, err := svc.Submit(ctx, SubmitResult{
		TaskID:   "d1",
		Concepts: []ConceptResult{{Label: "JWT", Confidence: 0.9}},
		// no edges → default DESCRIBES from source to each concept
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.EdgesAdded != 1 {
		t.Fatalf("want 1 default edge, got %d", rep.EdgesAdded)
	}
	edges, _ := gs.AllEdges(ctx)
	if len(edges) != 1 || edges[0].Kind != model.EdgeDescribes || edges[0].Src != "d1" {
		t.Fatalf("default edge wrong: %+v", edges)
	}
}

func TestSubmitEmbedsConceptsWhenEmbedderPresent(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "jwt auth")})

	svc := New(gs, embed.NewHashEmbedder(64), 0.5)
	if _, err := svc.Submit(ctx, SubmitResult{
		TaskID:   "d1",
		Concepts: []ConceptResult{{Label: "JWT", Description: "token", Confidence: 0.9}},
	}); err != nil {
		t.Fatal(err)
	}

	es := store.GraphStore(gs).(store.EmbeddingStore)
	embs, _ := es.AllEmbeddings(ctx)
	if len(embs) != 1 {
		t.Fatalf("expected 1 concept embedding, got %d", len(embs))
	}
}

func TestSubmitUnknownTaskErrors(t *testing.T) {
	gs := newStore(t)
	svc := New(gs, nil, 0)
	if _, err := svc.Submit(context.Background(), SubmitResult{TaskID: "missing"}); err == nil {
		t.Fatal("expected error for unknown task id")
	}
}

func TestResolveProvenance(t *testing.T) {
	cases := []struct {
		conf     float32
		wantProv model.Provenance
		wantConf float32
	}{
		{0.9, model.ProvInferred, 0.9},
		{0.5, model.ProvInferred, 0.5},
		{0.49, model.ProvAmbiguous, 0.49},
		{-1, model.ProvAmbiguous, 0},
		{2, model.ProvInferred, 1},
	}
	for _, c := range cases {
		p, conf := resolveProvenance(c.conf, 0.5)
		if p != c.wantProv || conf != c.wantConf {
			t.Fatalf("resolveProvenance(%v) = (%s,%v), want (%s,%v)", c.conf, p, conf, c.wantProv, c.wantConf)
		}
	}
}
