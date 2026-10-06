package mcpsrv

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
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
