package mcpsrv

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// TestQueryGraphFiltersBeforeLimit checks source_prefix narrows the matches before limit caps them.
func TestQueryGraphFiltersBeforeLimit(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	var nodes []model.Node
	for i := range 15 {
		nodes = append(nodes, model.Node{
			ID: fmt.Sprintf("a%02d", i), Kind: model.KindSymbol, Label: fmt.Sprintf("NewThing%d", i),
			SourceURI: "file:///repo/internal/other/x.go",
		})
	}
	// Its ID sorts after the fifteen above, so a limit applied first never reaches it.
	nodes = append(nodes, model.Node{
		ID: "z", Kind: model.KindSymbol, Label: "NewServer", SourceURI: "file:///repo/internal/mcp/server.go",
	})
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	out := call(t, queryHandler(gs, redactor{}), map[string]any{"query": "New", "source_prefix": "internal/mcp"})
	if !strings.Contains(out, "NewServer") {
		t.Errorf("query_graph(New, source_prefix internal/mcp) = %s, want NewServer", out)
	}
	out = call(t, queryHandler(gs, redactor{}), map[string]any{"query": "New", "limit": 3})
	if got := strings.Count(out, `"label"`); got != 3 {
		t.Errorf("query_graph(New, limit 3) returned %d nodes, want 3: %s", got, out)
	}
}

// TestQueryGraphSkipsHiddenBeforeLimit checks redacted people don't use up the limit and hide a real match.
func TestQueryGraphSkipsHiddenBeforeLimit(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	var nodes []model.Node
	for i := range 12 {
		nodes = append(nodes, model.Node{
			ID: fmt.Sprintf("a%02d", i), Kind: model.KindPerson, Label: fmt.Sprintf("Alice %d", i),
			SourceURI: fmt.Sprintf("person://alice%d", i), Attrs: map[string]any{"pii": true},
		})
	}
	nodes = append(nodes, model.Node{ID: "z", Kind: model.KindSymbol, Label: "validate", SourceURI: "file:///repo/v.go"})
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	out := call(t, queryHandler(gs, redactor{}), map[string]any{"query": "ali"})
	if !strings.Contains(out, "validate") {
		t.Errorf("query_graph(ali) = %s, want the validate symbol", out)
	}
	if strings.Contains(out, "Alice") {
		t.Errorf("query_graph(ali) leaked a person: %s", out)
	}
}

// TestShortestPathRanksEndpoints checks endpoints resolve like get_node: a definition beats a same-named
// hidden person or a file that merely sorts first.
func TestShortestPathRanksEndpoints(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	nodes := []model.Node{
		{ID: "a", Kind: model.KindPerson, Label: "Retry", SourceURI: "person://retry", Attrs: map[string]any{"pii": true}},
		{ID: "b", Kind: model.KindFile, Label: "backoff", SourceURI: "file:///repo/backoff"},
		{ID: "y", Kind: model.KindSymbol, Label: "Retry", SourceURI: "file:///repo/retry.go"},
		{ID: "z", Kind: model.KindSymbol, Label: "backoff", SourceURI: "file:///repo/retry.go"},
	}
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{{
		ID: model.NewEdgeID("y", "z", model.EdgeCalls), Src: "y", Dst: "z", Kind: model.EdgeCalls,
	}}); err != nil {
		t.Fatal(err)
	}
	out := call(t, shortestPathHandler(gs, redactor{}), map[string]any{"from": "Retry", "to": "backoff"})
	if !strings.Contains(out, `"CALLS"`) {
		t.Errorf("shortest_path(Retry, backoff) = %s, want the CALLS edge between the two symbols", out)
	}
}
