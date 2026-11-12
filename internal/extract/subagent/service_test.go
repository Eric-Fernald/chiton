package subagent

import (
	"context"
	"errors"
	"fmt"
	"net"
	"path/filepath"
	"slices"
	"strings"
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

// TestSubmitDescribesWhenAllEdgesUnresolved verifies that when every declared edge targets an uncreated
// concept, the source still gets a DESCRIBES edge to each concept rather than orphaning them.
func TestSubmitDescribesWhenAllEdgesUnresolved(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "jwt auth")})

	svc := New(gs, nil, 0.5)
	rep, err := svc.Submit(ctx, SubmitResult{
		TaskID:   "d1",
		Concepts: []ConceptResult{{Label: "JWT", Confidence: 0.9}},
		Edges: []EdgeResult{
			// Non-empty edges, but the target was never created → all skipped.
			{From: "", To: "Nonexistent", Kind: "DESCRIBES", Confidence: 0.9},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.EdgesAdded != 1 {
		t.Fatalf("want 1 fallback DESCRIBES edge, got %d", rep.EdgesAdded)
	}
	edges, _ := gs.AllEdges(ctx)
	if len(edges) != 1 || edges[0].Kind != model.EdgeDescribes || edges[0].Src != "d1" {
		t.Fatalf("expected one DESCRIBES edge from the source, got %+v", edges)
	}
	all, _ := gs.AllNodes(ctx)
	var conceptID string
	for _, n := range all {
		if n.Kind == model.KindConcept {
			conceptID = n.ID
		}
	}
	if conceptID == "" || edges[0].Dst != conceptID {
		t.Fatalf("concept left orphaned: edge dst %q, concept %q", edges[0].Dst, conceptID)
	}
}

// TestSubmitRejectsReadOnlyStore verifies submit fails fast with an actionable message on a read-only
// store, instead of a raw backend write error after the embed round-trip.
func TestSubmitRejectsReadOnlyStore(t *testing.T) {
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "graph.badger")
	rw, err := badgerstore.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := rw.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "jwt auth")}); err != nil {
		t.Fatal(err)
	}
	rw.Close()

	ro, err := badgerstore.OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ro.Close() })

	svc := New(ro, embed.NewHashEmbedder(64), 0.5)
	_, err = svc.Submit(ctx, SubmitResult{
		TaskID:   "d1",
		Concepts: []ConceptResult{{Label: "JWT", Confidence: 0.9}},
	})
	if err == nil {
		t.Fatal("expected an error submitting to a read-only store")
	}
	if !strings.Contains(err.Error(), "read-only") || !strings.Contains(err.Error(), "writable") {
		t.Fatalf("expected an actionable read-only message, got: %v", err)
	}
}

func TestSubmitEmbedsConceptsWhenEmbedderPresent(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "jwt auth")})

	emb := embed.NewHashEmbedder(64)
	svc := New(gs, emb, 0.5)
	if _, err := svc.Submit(ctx, SubmitResult{
		TaskID: "d1",
		Concepts: []ConceptResult{
			{Label: "JWT", Description: "token", Confidence: 0.9},
			{Label: "Session", Description: "server-side login state", Confidence: 0.8},
		},
	}); err != nil {
		t.Fatal(err)
	}

	es := store.GraphStore(gs).(store.EmbeddingStore)
	embs, _ := es.AllEmbeddings(ctx)
	if len(embs) != 2 {
		t.Fatalf("expected 2 concept embeddings, got %d", len(embs))
	}
	// Each stored vector must belong to its own concept's text.
	for _, e := range embs {
		n, err := gs.GetNode(ctx, e.NodeID)
		if err != nil || n == nil {
			t.Fatalf("embedding for unknown node %q: %v", e.NodeID, err)
		}
		want, _ := emb.Embed(ctx, []string{embed.TextFor(*n)})
		if !slices.Equal(e.Vec, want[0]) {
			t.Fatalf("embedding for %q does not match its concept text", n.Label)
		}
	}
}

// flakyEmbedder fails any call whose batch contains "poison", and with short set drops the last vector of any
// multi-text batch.
type flakyEmbedder struct {
	embed.Embedder
	short bool
}

func (f flakyEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	for _, t := range texts {
		if strings.Contains(t, "poison") {
			return nil, errors.New("embedder rejected batch")
		}
	}
	vecs, err := f.Embedder.Embed(ctx, texts)
	if f.short && len(vecs) > 1 {
		vecs = vecs[:len(vecs)-1]
	}
	return vecs, err
}

// TestSubmitEmbedFallsBackPerConcept verifies that when the batch embed fails or comes back short, only the
// concepts that fail on their own lose their inline vectors.
func TestSubmitEmbedFallsBackPerConcept(t *testing.T) {
	for _, tc := range []struct {
		name     string
		emb      flakyEmbedder
		concepts []string
		want     []string
	}{
		{"batch error", flakyEmbedder{}, []string{"JWT", "poison pill", "Session"}, []string{"JWT", "Session"}},
		{"short batch", flakyEmbedder{short: true}, []string{"JWT", "Session"}, []string{"JWT", "Session"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gs := newStore(t)
			ctx := context.Background()
			_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "jwt auth")})
			tc.emb.Embedder = embed.NewHashEmbedder(64)

			res := SubmitResult{TaskID: "d1"}
			for _, c := range tc.concepts {
				res.Concepts = append(res.Concepts, ConceptResult{Label: c, Confidence: 0.9})
			}
			if _, err := New(gs, tc.emb, 0.5).Submit(ctx, res); err != nil {
				t.Fatal(err)
			}

			embs, _ := store.GraphStore(gs).(store.EmbeddingStore).AllEmbeddings(ctx)
			var got []string
			for _, e := range embs {
				n, err := gs.GetNode(ctx, e.NodeID)
				if err != nil || n == nil {
					t.Fatalf("embedding for unknown node %q: %v", e.NodeID, err)
				}
				want, _ := tc.emb.Embedder.Embed(ctx, []string{embed.TextFor(*n)})
				if !slices.Equal(e.Vec, want[0]) {
					t.Fatalf("embedding for %q does not match its concept text", n.Label)
				}
				got = append(got, n.Label)
			}
			slices.Sort(got)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("embedded concepts = %v, want %v", got, tc.want)
			}
		})
	}
}

// downEmbedder fails every call the way an unreachable Ollama does (a transport error), counting calls.
type downEmbedder struct {
	embed.Embedder
	calls *int
}

func (d downEmbedder) Embed(context.Context, []string) ([][]float32, error) {
	*d.calls++
	return nil, fmt.Errorf("ollama: POST failed: %w", &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")})
}

// TestSubmitEmbedUnreachableSkipsFallback verifies an unreachable embedder costs one call, not one per concept.
func TestSubmitEmbedUnreachableSkipsFallback(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "jwt auth")})
	calls := 0
	emb := downEmbedder{Embedder: embed.NewHashEmbedder(64), calls: &calls}

	res := SubmitResult{TaskID: "d1", Concepts: []ConceptResult{
		{Label: "JWT", Confidence: 0.9}, {Label: "Session", Confidence: 0.9}, {Label: "Token", Confidence: 0.9},
	}}
	if _, err := New(gs, emb, 0.5).Submit(ctx, res); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("embedder called %d times, want 1 (no per-concept retries when unreachable)", calls)
	}
}

func TestSubmitUnknownTaskErrors(t *testing.T) {
	gs := newStore(t)
	svc := New(gs, nil, 0)
	if _, err := svc.Submit(context.Background(), SubmitResult{TaskID: "missing"}); err == nil {
		t.Fatal("expected error for unknown task id")
	}
}

// A task_id naming a symbol or meeting must not mark it extracted (which would also hide the meeting from
// pending_meeting_links, since both flows share extraction_status) or hang concepts off it.
func TestSubmitRejectsNodesThatAreNotExtractionTasks(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{
		{ID: "sym1", Kind: model.KindSymbol, Label: "fetchAll", SourceURI: "file://a.go"},
		{ID: "m1", Kind: model.KindMeeting, Label: "Sync", SourceURI: "meeting://sync", Attrs: map[string]any{"text": "x"}},
		docSection("empty", "Empty", ""),
	})
	svc := New(gs, nil, 0.5)
	for _, id := range []string{"sym1", "m1", "empty"} {
		if _, err := svc.Submit(ctx, SubmitResult{TaskID: id, Concepts: []ConceptResult{{Label: "X", Confidence: 0.9}}}); err == nil {
			t.Errorf("Submit(task_id=%s) succeeded; want it rejected", id)
		}
		if n, _ := gs.GetNode(ctx, id); model.AttrString(n.Attrs, "extraction_status") != "" {
			t.Errorf("%s was marked extracted", id)
		}
	}
}

// Concepts are global, so a later task must not rewrite one an earlier task created.
func TestSubmitKeepsExistingConcept(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "we use JWTs"), docSection("d2", "Notes", "tokens")})
	svc := New(gs, nil, 0.5)
	if _, err := svc.Submit(ctx, SubmitResult{TaskID: "d1",
		Concepts: []ConceptResult{{Label: "JWT", Description: "JSON Web Token", Confidence: 0.9}}}); err != nil {
		t.Fatal(err)
	}
	rep, err := svc.Submit(ctx, SubmitResult{TaskID: "d2",
		Concepts: []ConceptResult{{Label: "JWT", Description: "attacker text", Confidence: 0.1}}})
	if err != nil {
		t.Fatal(err)
	}
	if rep.ConceptsAdded != 0 {
		t.Errorf("ConceptsAdded = %d for an existing concept, want 0", rep.ConceptsAdded)
	}
	id := model.NewNodeID("concept://jwt", "JWT")
	n, _ := gs.GetNode(ctx, id)
	if n == nil || model.AttrString(n.Attrs, "description") != "JSON Web Token" ||
		model.AttrString(n.Attrs, "extracted_from") != "d1" || n.Provenance != model.ProvInferred {
		t.Fatalf("existing concept was rewritten: %+v", n)
	}
	edges, _ := gs.AllEdges(ctx)
	if !slices.ContainsFunc(edges, func(e model.Edge) bool { return e.Src == "d2" && e.Dst == id }) {
		t.Error("d2 was not linked to the existing concept")
	}
}

func conceptNodeID(label string) string {
	return model.NewNodeID("concept://"+model.Slug(label, "concept"), label)
}

func hasEdge(t *testing.T, gs store.GraphStore, src, dst string, kind model.EdgeKind) bool {
	t.Helper()
	edges, err := gs.AllEdges(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return slices.ContainsFunc(edges, func(e model.Edge) bool { return e.Src == src && e.Dst == dst && e.Kind == kind })
}

// A resubmission is the source's full current set, so a concept the new result omits loses its link to the source.
func TestResubmitDropsEdgesTheNewResultOmits(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "we use JWTs and sessions")})
	svc := New(gs, nil, 0.5)
	if _, err := svc.Submit(ctx, SubmitResult{TaskID: "d1", Concepts: []ConceptResult{
		{Label: "JWT", Confidence: 0.9}, {Label: "Session", Confidence: 0.9},
	}}); err != nil {
		t.Fatal(err)
	}

	rep, err := svc.Submit(ctx, SubmitResult{TaskID: "d1", Concepts: []ConceptResult{{Label: "JWT", Confidence: 0.9}}})
	if err != nil {
		t.Fatal(err)
	}
	if rep.EdgesRemoved != 1 {
		t.Errorf("EdgesRemoved = %d, want 1", rep.EdgesRemoved)
	}
	if hasEdge(t, gs, "d1", conceptNodeID("Session"), model.EdgeDescribes) {
		t.Error("d1 still describes Session, which the resubmission omitted")
	}
	if !hasEdge(t, gs, "d1", conceptNodeID("JWT"), model.EdgeDescribes) {
		t.Error("d1 lost its edge to JWT, which the resubmission kept")
	}
	if n, _ := gs.GetNode(ctx, conceptNodeID("Session")); n == nil {
		t.Error("Session concept was deleted; Submit should only unlink it")
	}
}

func TestResubmitDropsOmittedConceptToConceptEdge(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "JWTs carry the session")})
	svc := New(gs, nil, 0.5)
	concepts := []ConceptResult{{Label: "JWT", Confidence: 0.9}, {Label: "Session", Confidence: 0.9}}
	describes := []EdgeResult{{To: "JWT", Confidence: 0.9}, {To: "Session", Confidence: 0.9}}
	if _, err := svc.Submit(ctx, SubmitResult{TaskID: "d1", Concepts: concepts,
		Edges: append(slices.Clone(describes), EdgeResult{From: "JWT", To: "Session", Kind: "ELABORATES", Confidence: 0.8}),
	}); err != nil {
		t.Fatal(err)
	}
	jwt, session := conceptNodeID("JWT"), conceptNodeID("Session")
	if !hasEdge(t, gs, jwt, session, model.EdgeElaborates) {
		t.Fatal("setup: JWT→Session edge missing")
	}

	if _, err := svc.Submit(ctx, SubmitResult{TaskID: "d1", Concepts: concepts, Edges: describes}); err != nil {
		t.Fatal(err)
	}
	if hasEdge(t, gs, jwt, session, model.EdgeElaborates) {
		t.Error("JWT→Session survived a resubmission that omitted it")
	}
}

// Reconciling one source must not touch edges another source's task wrote.
func TestResubmitKeepsOtherTasksEdges(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "we use JWTs"), docSection("d2", "Notes", "JWTs and sessions")})
	svc := New(gs, nil, 0.5)
	if _, err := svc.Submit(ctx, SubmitResult{TaskID: "d1", Concepts: []ConceptResult{{Label: "JWT", Confidence: 0.9}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.Submit(ctx, SubmitResult{TaskID: "d2",
		Concepts: []ConceptResult{{Label: "JWT", Confidence: 0.9}, {Label: "Session", Confidence: 0.9}},
		Edges: []EdgeResult{
			{To: "JWT", Confidence: 0.9}, {To: "Session", Confidence: 0.9},
			{From: "JWT", To: "Session", Kind: "ELABORATES", Confidence: 0.8},
		},
	}); err != nil {
		t.Fatal(err)
	}

	rep, err := svc.Submit(ctx, SubmitResult{TaskID: "d1"})
	if err != nil {
		t.Fatal(err)
	}
	if rep.EdgesRemoved != 1 {
		t.Errorf("EdgesRemoved = %d, want 1 (only d1→JWT)", rep.EdgesRemoved)
	}
	jwt, session := conceptNodeID("JWT"), conceptNodeID("Session")
	if hasEdge(t, gs, "d1", jwt, model.EdgeDescribes) {
		t.Error("d1→JWT survived an empty resubmission")
	}
	if !hasEdge(t, gs, "d2", jwt, model.EdgeDescribes) || !hasEdge(t, gs, jwt, session, model.EdgeElaborates) {
		t.Error("d1's resubmission removed edges d2's task wrote")
	}
}

// The task that created a concept may correct it; TestSubmitKeepsExistingConcept covers every other task.
func TestOwnerResubmitRewritesItsConcept(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "we use JWTs")})
	svc := New(gs, nil, 0.5)
	first := SubmitResult{TaskID: "d1", Concepts: []ConceptResult{{Label: "JWT", Description: "a cookie", Confidence: 0.6}}}
	if _, err := svc.Submit(ctx, first); err != nil {
		t.Fatal(err)
	}
	before, _ := gs.GetNode(ctx, conceptNodeID("JWT"))

	rep, err := svc.Submit(ctx, first)
	if err != nil {
		t.Fatal(err)
	}
	if rep.ConceptsUpdated != 0 {
		t.Errorf("ConceptsUpdated = %d for an identical resubmission, want 0", rep.ConceptsUpdated)
	}

	rep, err = svc.Submit(ctx, SubmitResult{TaskID: "d1",
		Concepts: []ConceptResult{{Label: "JWT", Description: "JSON Web Token", Confidence: 0.9}}})
	if err != nil {
		t.Fatal(err)
	}
	if rep.ConceptsAdded != 0 || rep.ConceptsUpdated != 1 {
		t.Errorf("report = %+v, want 0 added and 1 updated", rep)
	}
	after, _ := gs.GetNode(ctx, conceptNodeID("JWT"))
	if model.AttrString(after.Attrs, "description") != "JSON Web Token" || after.Confidence != 0.9 {
		t.Errorf("owner's correction not applied: %+v", after)
	}
	if !after.CreatedAt.Equal(before.CreatedAt) {
		t.Errorf("CreatedAt changed on update: %v → %v", before.CreatedAt, after.CreatedAt)
	}
}

func TestSweepOrphansDemotesConceptsNoSourceLinksTo(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	concept := func(id string) model.Node {
		return model.Node{ID: id, Kind: model.KindConcept, Label: id, SourceURI: "concept://" + id,
			Provenance: model.ProvInferred, Confidence: 0.9}
	}
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "x"), concept("linked"), concept("viaconcept"), concept("alone")})
	_ = gs.UpsertEdges(ctx, []model.Edge{
		{ID: "e1", Src: "d1", Dst: "linked", Kind: model.EdgeDescribes, Provenance: model.ProvInferred},
		// A link from another concept is not evidence: nothing in a source supports "viaconcept".
		{ID: "e2", Src: "linked", Dst: "viaconcept", Kind: model.EdgeElaborates, Provenance: model.ProvInferred},
	})
	svc := New(gs, nil, 0.5)

	rep, err := svc.SweepOrphans(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Demoted != 2 {
		t.Errorf("Demoted = %d, want 2", rep.Demoted)
	}
	for id, want := range map[string]model.Provenance{
		"linked": model.ProvInferred, "viaconcept": model.ProvAmbiguous, "alone": model.ProvAmbiguous,
	} {
		n, _ := gs.GetNode(ctx, id)
		if n.Provenance != want {
			t.Errorf("%s provenance = %s, want %s", id, n.Provenance, want)
		}
		if orphaned := model.AttrString(n.Attrs, "orphaned_at") != ""; orphaned != (want == model.ProvAmbiguous) {
			t.Errorf("%s orphaned_at set = %v", id, orphaned)
		}
	}

	if rep, err = svc.SweepOrphans(ctx); err != nil || rep.Demoted != 0 {
		t.Errorf("second sweep = %+v, %v; want nothing left to demote", rep, err)
	}
}

// An orphan has no owner left, so the next task to extract it takes it over and it is visible again.
func TestSubmitAdoptsOrphanedConcept(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "sessions"), docSection("d2", "Notes", "sessions again")})
	svc := New(gs, nil, 0.5)
	if _, err := svc.Submit(ctx, SubmitResult{TaskID: "d1",
		Concepts: []ConceptResult{{Label: "Session", Description: "old", Confidence: 0.9}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.Submit(ctx, SubmitResult{TaskID: "d1"}); err != nil {
		t.Fatal(err)
	}
	if rep, err := svc.SweepOrphans(ctx); err != nil || rep.Demoted != 1 {
		t.Fatalf("setup sweep = %+v, %v; want Session demoted", rep, err)
	}

	if _, err := svc.Submit(ctx, SubmitResult{TaskID: "d2",
		Concepts: []ConceptResult{{Label: "Session", Description: "server-side login state", Confidence: 0.8}}}); err != nil {
		t.Fatal(err)
	}
	n, _ := gs.GetNode(ctx, conceptNodeID("Session"))
	if n.Provenance != model.ProvInferred || model.AttrString(n.Attrs, "orphaned_at") != "" ||
		model.AttrString(n.Attrs, "extracted_from") != "d2" || model.AttrString(n.Attrs, "description") != "server-side login state" {
		t.Fatalf("orphan was not adopted by d2: %+v", n)
	}
}
