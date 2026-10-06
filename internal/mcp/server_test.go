package mcpsrv

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

func TestPendingAndSubmitErrorText(t *testing.T) {
	res, _ := pendingResult[int](nil, errors.New("x"), "pending_meeting_links", "none")
	if got := resultText(t, res); got != "pending meeting links failed: x" {
		t.Errorf("pendingResult error text = %q", got)
	}
	submit := func(context.Context, struct{}) (int, error) { return 0, errors.New("y") }
	res, _ = bindAndSubmit(context.Background(), toolReq(map[string]any{}), "submit_analysis", submit)
	if got := resultText(t, res); got != "submit analysis failed: y" {
		t.Errorf("bindAndSubmit error text = %q", got)
	}
}

func TestGraphSummaryTopNodesByBetweenness(t *testing.T) {
	overlay := &cluster.Results{Nodes: map[string]cluster.NodeResult{}}
	for i := range 7 {
		label := fmt.Sprintf("n%d", i)
		overlay.Nodes[label] = cluster.NodeResult{Label: label, CommunityID: 1, Betweenness: float64(i)}
	}
	overlay.Nodes["solo"] = cluster.NodeResult{Label: "solo", CommunityID: 2}
	gs := openStore(t) // redaction labels overlay entries from their stored nodes
	var nodes []model.Node
	for id, nr := range overlay.Nodes {
		nodes = append(nodes, model.Node{ID: id, Kind: model.KindSymbol, Label: nr.Label, Provenance: model.ProvExtracted, Confidence: 1})
	}
	if err := gs.UpsertNodes(context.Background(), nodes); err != nil {
		t.Fatal(err)
	}

	res, err := graphSummaryHandler(gs, overlay, redactor{})(context.Background(), toolReq(map[string]any{}))
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Communities []struct {
			ID        int      `json:"id"`
			NodeCount int      `json:"node_count"`
			TopNodes  []string `json:"top_nodes"`
		} `json:"communities"`
		GodNodes []struct {
			Label string `json:"label"`
		} `json:"god_nodes"`
	}
	if err := json.Unmarshal([]byte(resultText(t, res)), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, resultText(t, res))
	}
	if len(got.Communities) != 2 || got.Communities[0].ID != 1 || got.Communities[0].NodeCount != 7 {
		t.Fatalf("communities = %+v, want community 1 (7 nodes) first", got.Communities)
	}
	if want := []string{"n6", "n5", "n4", "n3", "n2"}; !slices.Equal(got.Communities[0].TopNodes, want) {
		t.Errorf("top_nodes = %v, want %v", got.Communities[0].TopNodes, want)
	}
	if len(got.GodNodes) != 8 || got.GodNodes[0].Label != "n6" {
		t.Errorf("god_nodes = %+v, want 8 led by n6", got.GodNodes)
	}
}

func TestJSONResultIsCompactAndUnescaped(t *testing.T) {
	res, err := jsonResult(map[string]string{"sig": "func f(ch <-chan int) bool { return a && b }"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := resultText(t, res), `{"sig":"func f(ch <-chan int) bool { return a && b }"}`; got != want {
		t.Errorf("jsonResult = %s, want %s", got, want)
	}
}

// seedFanOut stores a hub symbol that CALLS n leaf symbols.
func seedFanOut(t *testing.T, gs store.GraphStore, n int) {
	t.Helper()
	ctx := context.Background()
	nodes := []model.Node{{ID: "hub", Kind: model.KindSymbol, Label: "Hub", SourceURI: "file://hub.go",
		Provenance: model.ProvExtracted, Confidence: 1}}
	var edges []model.Edge
	for i := range n {
		id := fmt.Sprintf("leaf%03d", i)
		nodes = append(nodes, model.Node{ID: id, Kind: model.KindSymbol, Label: id, SourceURI: "file://leaf.go",
			Provenance: model.ProvExtracted, Confidence: 1})
		edges = append(edges, model.Edge{ID: model.NewEdgeID("hub", id, model.EdgeCalls), Src: "hub", Dst: id,
			Kind: model.EdgeCalls, Provenance: model.ProvExtracted, Confidence: 1})
	}
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, edges); err != nil {
		t.Fatal(err)
	}
}

func TestNeighborhoodCapsLargeFanOut(t *testing.T) {
	gs := openStore(t)
	seedFanOut(t, gs, 300)
	h := neighborhoodHandler(gs, redactor{})

	var got struct {
		Nodes     []leanNode `json:"nodes"`
		Edges     []leanEdge `json:"edges"`
		Truncated bool       `json:"truncated"`
	}
	out := call(t, h, map[string]any{"label": "Hub", "depth": 1, "max_nodes": 50})
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out)
	}
	if len(got.Nodes) != 50 || len(got.Edges) != 49 || !got.Truncated {
		t.Errorf("max_nodes=50: %d nodes, %d edges, truncated=%v; want 50, 49, true", len(got.Nodes), len(got.Edges), got.Truncated)
	}

	out = call(t, h, map[string]any{"label": "Hub", "depth": 1, "max_nodes": 500, "max_tokens": 500})
	if len(out) > 3000 {
		t.Errorf("max_tokens=500 produced %d bytes, want roughly 2000 plus the seed and summary", len(out))
	}
	if strings.Contains(out, "\n") {
		t.Error("neighborhood output is indented; want compact JSON")
	}
}

func TestNeighborhoodSourcePrefixKeepsSeedAndUsesRedactedURI(t *testing.T) {
	gs := openStore(t)
	ctx := context.Background()
	nodes := []model.Node{
		{ID: "hub", Kind: model.KindSymbol, Label: "Hub", SourceURI: "file://hub.go", Provenance: model.ProvExtracted, Confidence: 1},
		{ID: "p", Kind: model.KindPerson, Label: "Alice", SourceURI: "person://alice@example.com", Provenance: model.ProvExtracted, Confidence: 1},
	}
	edges := []model.Edge{{ID: model.NewEdgeID("hub", "p", model.EdgeCalls), Src: "hub", Dst: "p", Kind: model.EdgeCalls, Provenance: model.ProvExtracted, Confidence: 1}}
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, edges); err != nil {
		t.Fatal(err)
	}
	h := neighborhoodHandler(gs, redactor{})
	var got struct {
		Nodes []leanNode `json:"nodes"`
	}
	out := call(t, h, map[string]any{"label": "Hub", "source_prefix": "alice@example"})
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Nodes) != 1 || got.Nodes[0].Label != "Hub" {
		t.Errorf("want only the seed when the prefix matches only a redacted URI, got %s", out)
	}
}
