package mcpsrv

import (
	"context"
	"strings"
	"testing"

	mcpsdk "github.com/mark3labs/mcp-go/server"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// call runs one tool handler and returns its text output.
func call(t *testing.T, h mcpsdk.ToolHandlerFunc, args map[string]any) string {
	t.Helper()
	res, err := h(context.Background(), toolReq(args))
	if err != nil {
		t.Fatal(err)
	}
	return resultText(t, res)
}

func TestGraphToolsRedactPeopleByDefault(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	seedLinkedMeeting(t, gs) // Alice and Bob speak in "Retry Policy Sync"
	alice := model.Node{
		ID: meeting.PersonID("alice@example.com"), Kind: model.KindPerson, Label: "Alice Smith",
		SourceURI:  meeting.PersonURI("alice@example.com"),
		Attrs:      map[string]any{"pii": true, "email": "alice@example.com"},
		Provenance: model.ProvExtracted, Confidence: 1,
	}
	mail := model.Node{
		ID: "mail1", Kind: model.KindEmail, Label: "Q3 plan", SourceURI: "email://q3",
		Attrs:      map[string]any{"subject": "Q3 plan", "from": "alice@example.com", "to": []any{"bob@example.com"}},
		Provenance: model.ProvExtracted, Confidence: 1,
	}
	if err := gs.UpsertNodes(ctx, []model.Node{alice, mail}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{{
		ID: model.NewEdgeID(mail.ID, alice.ID, model.EdgeMentioned), Src: mail.ID, Dst: alice.ID,
		Kind: model.EdgeMentioned, Provenance: model.ProvExtracted, Confidence: 1,
	}}); err != nil {
		t.Fatal(err)
	}
	overlay := &cluster.Results{Nodes: map[string]cluster.NodeResult{
		alice.ID: {Label: alice.Label, SourceURI: alice.SourceURI, Betweenness: 1},
	}}

	r := redactor{}
	outputs := map[string]string{
		"get_node meeting":     call(t, getNodeHandler(gs, r), map[string]any{"label": "Retry Policy Sync"}),
		"neighborhood meeting": call(t, neighborhoodHandler(gs, r), map[string]any{"label": "Retry Policy Sync", "depth": 1}),
		"neighborhood email":   call(t, neighborhoodHandler(gs, r), map[string]any{"label": "Q3 plan", "depth": 1}),
		"god_nodes":            call(t, godNodesHandler(overlay, r), map[string]any{}),
	}
	for tool, out := range outputs {
		for _, pii := range []string{"Alice", "Bob", "alice@example.com", "bob@example.com"} {
			if strings.Contains(out, pii) {
				t.Errorf("%s leaked %q: %s", tool, pii, out)
			}
		}
	}
	// A label search must not confirm that a person by that name exists.
	if out := call(t, queryHandler(gs, r), map[string]any{"query": "alice"}); !strings.Contains(out, "no nodes found") {
		t.Errorf("query_graph matched a person by name: %s", out)
	}

	// serve --include-pii restores the ungated view.
	out := call(t, getNodeHandler(gs, redactor{includePII: true}), map[string]any{"label": "Alice Smith"})
	if !strings.Contains(out, "alice@example.com") {
		t.Errorf("include-pii get_node = %s, want the address", out)
	}
}
