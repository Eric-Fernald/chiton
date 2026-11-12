package meeting

import (
	"context"
	"slices"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// submitRetryDecision links one decision governing fetchAll and returns the decision's node ID.
func submitRetryDecision(t *testing.T, gs store.GraphStore, svc *LinkService, refs ...string) string {
	t.Helper()
	ctx := context.Background()
	meetingID := seedMeetingAndCode(t, gs)
	if _, err := svc.Submit(ctx, LinkResult{TaskID: meetingID, Decisions: []DecisionResult{{
		Summary: "Cap retries at 3", Confidence: 0.9, References: refs,
	}}}); err != nil {
		t.Fatal(err)
	}
	decisions, err := store.NodesMatching(ctx, gs, store.NodeFilter{Kinds: []model.NodeKind{model.KindDecision}})
	if err != nil || len(decisions) != 1 {
		t.Fatalf("want 1 decision, got %d (%v)", len(decisions), err)
	}
	return decisions[0].ID
}

func governedBy(t *testing.T, gs store.GraphStore, decisionID string) []governRef {
	t.Helper()
	n, err := gs.GetNode(context.Background(), decisionID)
	if err != nil || n == nil {
		t.Fatalf("load decision: %v", err)
	}
	refs, _ := governed(*n)
	return refs
}

func decidedIn(t *testing.T, gs store.GraphStore, decisionID, symbolID string) bool {
	t.Helper()
	edges, err := gs.AllEdges(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return slices.ContainsFunc(edges, func(e model.Edge) bool {
		return e.Src == decisionID && e.Dst == symbolID && e.Kind == model.EdgeDecidedIn
	})
}

func deleteSymbol(t *testing.T, gs store.GraphStore, label, uri string) {
	t.Helper()
	if _, err := store.DeleteNodes(context.Background(), gs, []model.Node{symbolNode(label, uri)}); err != nil {
		t.Fatal(err)
	}
}

// A decision remembers every reference by label, so a later pass can tell what it governed after the node is gone.
func TestSubmitRecordsGovernedSymbols(t *testing.T) {
	gs := newStore(t)
	svc := NewLinkService(gs, 0.5)
	dec := submitRetryDecision(t, gs, svc, "fetchAll", "doesNotExist", "fetchAll")

	want := []governRef{
		{Label: "fetchAll", NodeID: symbolNode("fetchAll", "file://client.go").ID},
		{Label: "doesNotExist"},
	}
	if got := governedBy(t, gs, dec); !slices.Equal(got, want) {
		t.Fatalf("governs = %+v, want %+v", got, want)
	}
}

func TestRelinkFollowsSymbolToNewFile(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	svc := NewLinkService(gs, 0.5)
	dec := submitRetryDecision(t, gs, svc, "fetchAll")

	// Moving a symbol to another file changes its ID: the old node goes, taking the decision's edge with it.
	deleteSymbol(t, gs, "fetchAll", "file://client.go")
	moved := symbolNode("fetchAll", "file://http.go")
	if err := gs.UpsertNodes(ctx, []model.Node{moved}); err != nil {
		t.Fatal(err)
	}

	rep, err := svc.Relink(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Relinked != 1 || len(rep.Unresolved) != 0 {
		t.Errorf("report = %+v, want 1 relinked and nothing unresolved", rep)
	}
	if !decidedIn(t, gs, dec, moved.ID) {
		t.Error("decision is not linked to fetchAll at its new location")
	}
	if got := governedBy(t, gs, dec); len(got) != 1 || got[0].NodeID != moved.ID {
		t.Errorf("governs = %+v, want it to point at the moved symbol", got)
	}

	if rep, err = svc.Relink(ctx); err != nil || rep.Relinked != 0 {
		t.Errorf("second relink = %+v, %v; want nothing left to do", rep, err)
	}
}

func TestRelinkReportsSymbolThatIsGone(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	svc := NewLinkService(gs, 0.5)
	dec := submitRetryDecision(t, gs, svc, "fetchAll")
	deleteSymbol(t, gs, "fetchAll", "file://client.go")
	// A longer name containing the old one is a different symbol, not the old one renamed.
	if err := gs.UpsertNodes(ctx, []model.Node{symbolNode("fetchAllPages", "file://client.go")}); err != nil {
		t.Fatal(err)
	}

	rep, err := svc.Relink(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Relinked != 0 || !slices.Equal(rep.Unresolved, []string{"fetchAll"}) {
		t.Errorf("report = %+v, want fetchAll unresolved", rep)
	}
	if decidedIn(t, gs, dec, symbolNode("fetchAllPages", "file://client.go").ID) {
		t.Error("decision was linked to a symbol that only contains the old name")
	}
	if got := governedBy(t, gs, dec); len(got) != 1 || got[0] != (governRef{Label: "fetchAll"}) {
		t.Errorf("governs = %+v, want fetchAll kept with no node", got)
	}
}

func TestRelinkLeavesIntactLinksAlone(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	svc := NewLinkService(gs, 0.5)
	submitRetryDecision(t, gs, svc, "fetchAll")
	before, _ := gs.AllEdges(ctx)

	rep, err := svc.Relink(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Relinked != 0 || len(rep.Unresolved) != 0 {
		t.Errorf("report = %+v, want nothing to do", rep)
	}
	if after, _ := gs.AllEdges(ctx); len(after) != len(before) {
		t.Errorf("edge count changed from %d to %d", len(before), len(after))
	}
}

// Decisions linked before governs existed get it filled in from the edges they still have.
func TestRelinkBackfillsDecisionsWithoutGoverns(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	svc := NewLinkService(gs, 0.5)
	dec := submitRetryDecision(t, gs, svc, "fetchAll")
	old, _ := gs.GetNode(ctx, dec)
	delete(old.Attrs, "governs")
	if err := gs.UpsertNodes(ctx, []model.Node{*old}); err != nil {
		t.Fatal(err)
	}

	if _, err := svc.Relink(ctx); err != nil {
		t.Fatal(err)
	}
	want := []governRef{{Label: "fetchAll", NodeID: symbolNode("fetchAll", "file://client.go").ID}}
	if got := governedBy(t, gs, dec); !slices.Equal(got, want) {
		t.Fatalf("governs = %+v, want %+v", got, want)
	}
}
