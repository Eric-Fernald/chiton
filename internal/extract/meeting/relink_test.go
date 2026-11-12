package meeting

import (
	"context"
	"fmt"
	"slices"
	"strings"
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

// Names like Open and New exist in many packages. With two candidates there is no telling which one the decision
// was about, so Relink links neither.
func TestRelinkDoesNotGuessBetweenSameNamedSymbols(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	svc := NewLinkService(gs, 0.5)
	dec := submitRetryDecision(t, gs, svc, "fetchAll")
	deleteSymbol(t, gs, "fetchAll", "file://client.go")
	a, b := symbolNode("fetchAll", "file://http.go"), symbolNode("fetchAll", "file://legacy.go")
	if err := gs.UpsertNodes(ctx, []model.Node{a, b}); err != nil {
		t.Fatal(err)
	}

	rep, err := svc.Relink(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Relinked != 0 || !slices.Equal(rep.Ambiguous, []string{"fetchAll"}) || len(rep.Unresolved) != 0 {
		t.Errorf("report = %+v, want fetchAll reported ambiguous", rep)
	}
	if decidedIn(t, gs, dec, a.ID) || decidedIn(t, gs, dec, b.ID) {
		t.Error("Relink picked one of two same-named symbols")
	}
}

// The reference stub a caller in another package points at shares the function's label. It is not a second
// definition, and a crowd of longer names must not push the real match out of reach.
func TestRelinkFindsTheOneDefinitionAmongStubsAndLongerNames(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	svc := NewLinkService(gs, 0.5)
	dec := submitRetryDecision(t, gs, svc, "fetchAll")
	deleteSymbol(t, gs, "fetchAll", "file://client.go")
	moved := symbolNode("fetchAll", "file://http.go")
	stub := symbolNode("fetchAll", "pkg://example.com/client")
	stub.Attrs = map[string]any{"external": true}
	nodes := []model.Node{moved, stub}
	for i := range 400 {
		nodes = append(nodes, symbolNode(fmt.Sprintf("fetchAllPages%d", i), "file://pages.go"))
	}
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}

	rep, err := svc.Relink(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Relinked != 1 || !decidedIn(t, gs, dec, moved.ID) {
		t.Errorf("report = %+v, want the decision relinked to the moved definition", rep)
	}
}

// A resubmission is the meeting's full linkage, so a decision that no longer names a symbol loses its links to it.
func TestResubmitDropsLinksToReferencesNoLongerNamed(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	meetingID := seedMeetingAndCode(t, gs)
	svc := NewLinkService(gs, 0.5)
	submit := func(ref string, discussed ...string) LinkReport {
		t.Helper()
		rep, err := svc.Submit(ctx, LinkResult{TaskID: meetingID, Discussed: discussed,
			Decisions: []DecisionResult{{Summary: "Cap retries at 3", Confidence: 0.9, References: []string{ref}}}})
		if err != nil {
			t.Fatal(err)
		}
		return rep
	}
	has := func(src, dst string, kind model.EdgeKind) bool {
		t.Helper()
		edges, _ := gs.AllEdges(ctx)
		return slices.ContainsFunc(edges, func(e model.Edge) bool { return e.Src == src && e.Dst == dst && e.Kind == kind })
	}
	fetchAll, cacheGet := symbolNode("fetchAll", "file://client.go").ID, symbolNode("cacheGet", "file://cache.go").ID

	submit("fetchAll", "cacheGet")
	decisions, _ := store.NodesMatching(ctx, gs, store.NodeFilter{Kinds: []model.NodeKind{model.KindDecision}})
	dec := decisions[0].ID
	// Drift a verification verdict recorded against the old reference.
	if err := gs.UpsertEdges(ctx, []model.Edge{{ID: model.NewEdgeID(fetchAll, dec, model.EdgeContradicts), Src: fetchAll, Dst: dec,
		Kind: model.EdgeContradicts, Provenance: model.ProvInferred, Confidence: 0.9}}); err != nil {
		t.Fatal(err)
	}
	participants := 0
	edges, _ := gs.AllEdges(ctx)
	for _, e := range edges {
		if e.Src == meetingID && e.Kind == model.EdgeMentioned && e.Provenance == model.ProvExtracted {
			participants++
		}
	}

	rep := submit("cacheGet")
	if rep.EdgesRemoved == 0 {
		t.Errorf("EdgesRemoved = 0, want the old reference's edges counted")
	}
	for name, gone := range map[string]bool{
		"decision DECIDED_IN fetchAll":  has(dec, fetchAll, model.EdgeDecidedIn),
		"fetchAll DISCUSSED_IN meeting": has(fetchAll, meetingID, model.EdgeDiscussedIn),
		"meeting MENTIONED fetchAll":    has(meetingID, fetchAll, model.EdgeMentioned),
		"fetchAll CONTRADICTS decision": has(fetchAll, dec, model.EdgeContradicts),
	} {
		if gone {
			t.Errorf("%s survived a resubmission that no longer names fetchAll", name)
		}
	}
	if !has(dec, cacheGet, model.EdgeDecidedIn) || !has(cacheGet, meetingID, model.EdgeDiscussedIn) {
		t.Error("the new reference is not linked")
	}
	after := 0
	edges, _ = gs.AllEdges(ctx)
	for _, e := range edges {
		if e.Src == meetingID && e.Kind == model.EdgeMentioned && e.Provenance == model.ProvExtracted {
			after++
		}
	}
	if after != participants || participants == 0 {
		t.Errorf("participant edges went from %d to %d; ingest's own edges must stay", participants, after)
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

// A link the verifier rejected must stay gone: the symbol still exists under the same name, so Relink would
// otherwise put it straight back.
func TestRejectLinkRemovesEdgeAndRelinkLeavesItRemoved(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	svc := NewLinkService(gs, 0.5)
	dec := submitRetryDecision(t, gs, svc, "fetchAll", "cacheGet")
	fetchAll := symbolNode("fetchAll", "file://client.go").ID
	cacheGet := symbolNode("cacheGet", "file://cache.go").ID

	if err := svc.RejectLink(ctx, dec, fetchAll); err != nil {
		t.Fatal(err)
	}
	if decidedIn(t, gs, dec, fetchAll) {
		t.Error("rejected link still present")
	}
	if !decidedIn(t, gs, dec, cacheGet) {
		t.Error("rejecting one link removed another")
	}

	rep, err := svc.Relink(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Relinked != 0 || len(rep.Unresolved) != 0 {
		t.Errorf("relink after reject = %+v, want nothing restored or reported", rep)
	}
	if decidedIn(t, gs, dec, fetchAll) {
		t.Error("Relink restored a rejected link")
	}
}

func TestRejectLinkOnDecisionWithoutGovernsKeepsItsOtherLinks(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	svc := NewLinkService(gs, 0.5)
	dec := submitRetryDecision(t, gs, svc, "fetchAll", "cacheGet")
	old, _ := gs.GetNode(ctx, dec)
	delete(old.Attrs, "governs")
	if err := gs.UpsertNodes(ctx, []model.Node{*old}); err != nil {
		t.Fatal(err)
	}

	if err := svc.RejectLink(ctx, dec, symbolNode("fetchAll", "file://client.go").ID); err != nil {
		t.Fatal(err)
	}
	want := []governRef{
		{Label: "cacheGet", NodeID: symbolNode("cacheGet", "file://cache.go").ID},
		{Label: "fetchAll", Rejected: true},
	}
	got := governedBy(t, gs, dec)
	slices.SortFunc(got, func(a, b governRef) int { return strings.Compare(a.Label, b.Label) })
	if !slices.Equal(got, want) {
		t.Fatalf("governs = %+v, want %+v", got, want)
	}
}

func TestRejectLinkErrorsWhenNotLinked(t *testing.T) {
	gs := newStore(t)
	svc := NewLinkService(gs, 0.5)
	dec := submitRetryDecision(t, gs, svc, "fetchAll")
	if err := svc.RejectLink(context.Background(), dec, symbolNode("cacheGet", "file://cache.go").ID); err == nil {
		t.Fatal("rejecting a link that does not exist succeeded")
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
