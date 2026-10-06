package mcpsrv

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	mcpsdk "github.com/mark3labs/mcp-go/server"

	"github.com/Eric-Fernald/chiton/internal/embed"
	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/retrieve"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/vector"
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
		"god_nodes":            call(t, godNodesHandler(gs, overlay, r), map[string]any{}),
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

// assertNoPII fails when out names any of the given people.
func assertNoPII(t *testing.T, tool, out string, people ...string) {
	t.Helper()
	for _, p := range people {
		if strings.Contains(out, p) {
			t.Errorf("%s leaked %q: %s", tool, p, out)
		}
	}
}

// overlayOf puts every stored node in one community, ranked by its position, as a cluster pass would.
func overlayOf(t *testing.T, gs store.GraphStore) *cluster.Results {
	t.Helper()
	nodes, err := gs.AllNodes(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	overlay := &cluster.Results{Nodes: map[string]cluster.NodeResult{}}
	for i, n := range nodes {
		overlay.Nodes[n.ID] = cluster.NodeResult{Label: n.Label, SourceURI: n.SourceURI, CommunityID: 1, Betweenness: float64(i)}
	}
	return overlay
}

func TestClusterToolsRedactSpeakerNamedTurns(t *testing.T) {
	gs := openStore(t)
	seedLinkedMeeting(t, gs) // turns are labelled "Alice" and "Bob"
	overlay := overlayOf(t, gs)

	r := redactor{}
	assertNoPII(t, "god_nodes", call(t, godNodesHandler(gs, overlay, r), map[string]any{"limit": 100}), "Alice", "Bob")
	assertNoPII(t, "graph_summary", call(t, graphSummaryHandler(gs, overlay, r), map[string]any{}), "Alice", "Bob")

	if out := call(t, godNodesHandler(gs, overlay, redactor{includePII: true}), map[string]any{"limit": 100}); !strings.Contains(out, "Alice") {
		t.Errorf("include-pii god_nodes = %s, want the speaker", out)
	}
}

func TestGraphSummaryScopeCannotProbeForPeople(t *testing.T) {
	gs := openStore(t)
	seedLinkedMeeting(t, gs) // Alice's identity URI is person://alice
	overlay := overlayOf(t, gs)

	for _, scope := range []string{"person://alice", "alice"} {
		out := call(t, graphSummaryHandler(gs, overlay, redactor{}), map[string]any{"scope": scope})
		var got struct {
			NodeCount int `json:"node_count"`
		}
		if err := json.Unmarshal([]byte(out), &got); err != nil {
			t.Fatalf("unmarshal: %v\n%s", err, out)
		}
		if got.NodeCount != 0 {
			t.Errorf("scope %q counted %d nodes, want 0: %s", scope, got.NodeCount, out)
		}
	}
}

func TestGraphToolsStripActionItemAssignees(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	mNodes, mEdges := meeting.ToGraph(&meeting.Transcript{
		Title: "Cache Planning",
		Turns: []meeting.Turn{{Speaker: "Dana", Text: "Someone should own cacheGet."}},
	}, time.Now())
	if err := gs.UpsertNodes(ctx, mNodes); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, mEdges); err != nil {
		t.Fatal(err)
	}
	if _, err := meeting.NewLinkService(gs, 0.5).Submit(ctx, meeting.LinkResult{
		TaskID: mNodes[0].ID,
		Decisions: []meeting.DecisionResult{{
			Summary: "Own the cache work", Confidence: 0.9,
			ActionItems: []meeting.ActionItem{{Text: "implement cacheGet", Assignee: "Carol"}},
		}},
	}); err != nil {
		t.Fatal(err)
	}

	out := call(t, getNodeHandler(gs, redactor{}), map[string]any{"label": "Own the cache work"})
	if !strings.Contains(out, "implement cacheGet") {
		t.Errorf("get_node dropped the action text: %s", out)
	}
	assertNoPII(t, "get_node decision", out, "Carol")
	assertNoPII(t, "neighborhood decision",
		call(t, neighborhoodHandler(gs, redactor{}), map[string]any{"label": "Own the cache work", "depth": 1}), "Carol")
}

func TestSemanticSearchRedactsPeople(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	seedLinkedMeeting(t, gs)
	nodes, err := gs.AllNodes(ctx)
	if err != nil {
		t.Fatal(err)
	}
	e := embed.NewHashEmbedder(64)
	idx := vector.NewFlat()
	for _, n := range nodes {
		if text := embed.TextFor(n); text != "" {
			vecs, err := e.Embed(ctx, []string{text})
			if err != nil {
				t.Fatal(err)
			}
			idx.Add([]vector.Item{{ID: n.ID, Vec: vecs[0]}})
		}
	}
	h := semanticSearchHandler(retrieve.New(gs, idx, e, nil), redactor{})
	out := call(t, h, map[string]any{"query": "fetchAll", "hops": 3})
	if !strings.Contains(out, "fetchAll") {
		t.Fatalf("semantic_search lost the symbol: %s", out)
	}
	assertNoPII(t, "semantic_search", out, "Alice", "Bob")

	// The summary counts what is returned, not what redaction removed.
	var got struct {
		Nodes   []any  `json:"nodes"`
		Edges   []any  `json:"edges"`
		Summary string `json:"summary"`
	}
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out)
	}
	if want := fmt.Sprintf("%d nodes, %d edges for %q", len(got.Nodes), len(got.Edges), "fetchAll"); got.Summary != want {
		t.Errorf("summary = %q, want %q", got.Summary, want)
	}
}
