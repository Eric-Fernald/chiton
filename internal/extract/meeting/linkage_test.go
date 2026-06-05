package meeting

import (
	"context"
	"path/filepath"
	"testing"
	"time"

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

func symbolNode(label, uri string) model.Node {
	return model.Node{
		ID: model.NewNodeID(uri, label), Kind: model.KindSymbol, Label: label,
		SourceURI: uri, Provenance: model.ProvExtracted, Confidence: 1,
	}
}

// seedMeetingAndCode upserts one meeting (mentioning fetchAll) plus a couple of
// code symbols, and returns the meeting node ID.
func seedMeetingAndCode(t *testing.T, gs store.GraphStore) string {
	t.Helper()
	ctx := context.Background()
	tr := &Transcript{
		Title: "Retry Policy Sync",
		Turns: []Turn{
			{Speaker: "Alice", Text: "We decided to cap retries at 3 in fetchAll."},
			{Speaker: "Bob", Text: "And cacheGet should stay as-is."},
		},
	}
	mNodes, mEdges := ToGraph(tr, time.Now())
	code := []model.Node{
		symbolNode("fetchAll", "file://client.go"),
		symbolNode("cacheGet", "file://cache.go"),
	}
	if err := gs.UpsertNodes(ctx, append(mNodes, code...)); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, mEdges); err != nil {
		t.Fatal(err)
	}
	return mNodes[0].ID
}

func TestPendingEmitsMeetingLinkage(t *testing.T) {
	gs := newStore(t)
	id := seedMeetingAndCode(t, gs)
	svc := NewLinkService(gs, 0)

	specs, err := svc.Pending(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 1 {
		t.Fatalf("got %d specs want 1: %+v", len(specs), specs)
	}
	s := specs[0]
	if s.ID != id || s.Kind != KindMeetingLinkage {
		t.Errorf("spec id/kind = %q/%q", s.ID, s.Kind)
	}
	// fetchAll is name-dropped in the transcript and must appear in the menu.
	found := false
	for _, c := range s.CandidateSymbols {
		if c == "fetchAll" {
			found = true
		}
	}
	if !found {
		t.Errorf("candidate_symbols %v missing fetchAll", s.CandidateSymbols)
	}
}

func TestSubmitCreatesDecisionAndEdges(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	id := seedMeetingAndCode(t, gs)
	svc := NewLinkService(gs, 0.5)

	rep, err := svc.Submit(ctx, LinkResult{
		TaskID: id,
		Decisions: []DecisionResult{{
			Summary:     "Cap retries at 3 with backoff",
			Rationale:   "Avoids hammering the upstream on outage",
			Confidence:  0.9,
			References:  []string{"fetchAll", "doesNotExist"},
			ActionItems: []ActionItem{{Text: "implement the cap", Assignee: "Alice"}},
		}},
		Discussed: []string{"cacheGet"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.DecisionsAdded != 1 {
		t.Errorf("decisions=%d want 1", rep.DecisionsAdded)
	}
	if rep.LinkedSymbols != 2 { // fetchAll (decision) + cacheGet (discussed)
		t.Errorf("linked symbols=%d want 2", rep.LinkedSymbols)
	}
	if len(rep.UnresolvedRefs) != 1 || rep.UnresolvedRefs[0] != "doesNotExist" {
		t.Errorf("unresolved=%v want [doesNotExist]", rep.UnresolvedRefs)
	}

	// Verify graph shape.
	nodes, _ := gs.AllNodes(ctx)
	kinds := map[model.NodeKind]int{}
	for _, n := range nodes {
		kinds[n.Kind]++
	}
	if kinds[model.KindDecision] != 1 {
		t.Errorf("decision nodes=%d want 1", kinds[model.KindDecision])
	}
	if kinds[model.KindRationale] != 1 {
		t.Errorf("rationale nodes=%d want 1", kinds[model.KindRationale])
	}

	edges, _ := gs.AllEdges(ctx)
	ek := map[model.EdgeKind]int{}
	for _, e := range edges {
		ek[e.Kind]++
	}
	// DECIDED_IN: decision→meeting + decision→fetchAll = 2
	if ek[model.EdgeDecidedIn] != 2 {
		t.Errorf("DECIDED_IN=%d want 2", ek[model.EdgeDecidedIn])
	}
	// DISCUSSED_IN: fetchAll→meeting + cacheGet→meeting = 2
	if ek[model.EdgeDiscussedIn] != 2 {
		t.Errorf("DISCUSSED_IN=%d want 2", ek[model.EdgeDiscussedIn])
	}
	if ek[model.EdgeRationaleFor] != 1 {
		t.Errorf("RATIONALE_FOR=%d want 1", ek[model.EdgeRationaleFor])
	}
	if ek[model.EdgeActionFor] != 1 {
		t.Errorf("ACTION_FOR=%d want 1", ek[model.EdgeActionFor])
	}

	// Meeting is marked linked → Pending no longer emits it.
	specs, _ := svc.Pending(ctx, 0)
	if len(specs) != 0 {
		t.Errorf("Pending still returns %d specs after submit", len(specs))
	}

	// Decision provenance is INFERRED (conf 0.9 ≥ 0.5 threshold).
	for _, n := range nodes {
		if n.Kind == model.KindDecision && n.Provenance != model.ProvInferred {
			t.Errorf("decision provenance=%q want INFERRED", n.Provenance)
		}
	}
}

func TestSubmitSupersedesPriorDecisions(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	id := seedMeetingAndCode(t, gs)
	svc := NewLinkService(gs, 0.5)

	first := LinkResult{TaskID: id, Decisions: []DecisionResult{
		{Summary: "Old decision to remove", Confidence: 0.8},
		{Summary: "Surviving decision", Confidence: 0.8},
	}}
	if _, err := svc.Submit(ctx, first); err != nil {
		t.Fatal(err)
	}
	// Re-submit must reset extraction_status; meetings only re-link if pending.
	// Simulate re-link by clearing the flag.
	m, _ := gs.GetNode(ctx, id)
	delete(m.Attrs, "extraction_status")
	if err := gs.UpsertNodes(ctx, []model.Node{*m}); err != nil {
		t.Fatal(err)
	}

	second := LinkResult{TaskID: id, Decisions: []DecisionResult{
		{Summary: "Surviving decision", Confidence: 0.8},
	}}
	if _, err := svc.Submit(ctx, second); err != nil {
		t.Fatal(err)
	}

	nodes, _ := gs.AllNodes(ctx)
	var decisions []string
	for _, n := range nodes {
		if n.Kind == model.KindDecision {
			decisions = append(decisions, n.Label)
		}
	}
	if len(decisions) != 1 || decisions[0] != "Surviving decision" {
		t.Errorf("decisions after re-submit=%v want [Surviving decision]", decisions)
	}
}

func TestSubmitDistinctIDsForDuplicateSummaries(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	id := seedMeetingAndCode(t, gs)
	svc := NewLinkService(gs, 0.5)

	// Two decisions with identical summaries (and rationales) in one meeting must
	// not collide onto a single node.
	if _, err := svc.Submit(ctx, LinkResult{
		TaskID: id,
		Decisions: []DecisionResult{
			{Summary: "Ship it", Rationale: "good enough", Confidence: 0.8},
			{Summary: "Ship it", Rationale: "good enough", Confidence: 0.8},
		},
	}); err != nil {
		t.Fatal(err)
	}
	nodes, _ := gs.AllNodes(ctx)
	dec, rat := 0, 0
	for _, n := range nodes {
		switch n.Kind {
		case model.KindDecision:
			dec++
		case model.KindRationale:
			rat++
		}
	}
	if dec != 2 {
		t.Errorf("decision nodes=%d want 2 (identical summaries must not collide)", dec)
	}
	if rat != 2 {
		t.Errorf("rationale nodes=%d want 2", rat)
	}
}

func TestSubmitDiscussedConfidenceNotThreshold(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	id := seedMeetingAndCode(t, gs)
	// A high threshold would make a threshold-as-confidence discussed link AMBIGUOUS;
	// the fixed default (0.7) keeps it INFERRED below 0.7.
	svc := NewLinkService(gs, 0.6)
	if _, err := svc.Submit(ctx, LinkResult{TaskID: id, Discussed: []string{"fetchAll"}}); err != nil {
		t.Fatal(err)
	}
	edges, _ := gs.AllEdges(ctx)
	for _, e := range edges {
		if e.Kind == model.EdgeDiscussedIn {
			if e.Provenance != model.ProvInferred {
				t.Errorf("discussed edge provenance=%q want INFERRED (conf 0.7 >= threshold 0.6)", e.Provenance)
			}
			if e.Confidence == 0.6 {
				t.Errorf("discussed edge confidence must not be the threshold value")
			}
		}
	}
}

func TestSubmitRejectsNonMeeting(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	if err := gs.UpsertNodes(ctx, []model.Node{symbolNode("foo", "file://x.go")}); err != nil {
		t.Fatal(err)
	}
	svc := NewLinkService(gs, 0)
	_, err := svc.Submit(ctx, LinkResult{TaskID: model.NewNodeID("file://x.go", "foo")})
	if err == nil {
		t.Fatal("expected error submitting against a non-meeting node")
	}
}

func TestIdentifierTokens(t *testing.T) {
	got := identifierTokens("fetchAll")
	if len(got) != 2 || got[0] != "fetch" || got[1] != "All" {
		t.Errorf("identifierTokens(fetchAll)=%v want [fetch All]", got)
	}
	got = identifierTokens("do_work")
	if len(got) != 2 || got[0] != "do" || got[1] != "work" {
		t.Errorf("identifierTokens(do_work)=%v want [do work]", got)
	}
}
