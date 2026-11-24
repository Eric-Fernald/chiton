package meeting

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// TestSubmitPrefersSymbolAmongManySameLabels checks a result cap can't hide the symbol behind same-named nodes.
func TestSubmitPrefersSymbolAmongManySameLabels(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	id := seedMeetingAndCode(t, gs)
	var nodes []model.Node
	for i := range 30 {
		nodes = append(nodes, model.Node{
			ID: fmt.Sprintf("a%02d", i), Kind: model.KindDocSection, Label: "Retry",
			SourceURI: fmt.Sprintf("file://doc%d.md", i), Provenance: model.ProvExtracted, Confidence: 1,
		})
	}
	// Its ID sorts after the thirty headings, so only an uncapped read reaches it.
	sym := model.Node{
		ID: "zz", Kind: model.KindSymbol, Label: "Retry", SourceURI: "file://retry.go",
		Provenance: model.ProvExtracted, Confidence: 1,
	}
	if err := gs.UpsertNodes(ctx, append(nodes, sym)); err != nil {
		t.Fatal(err)
	}
	if _, err := NewLinkService(gs, 0.5).Submit(ctx, LinkResult{
		TaskID:    id,
		Decisions: []DecisionResult{{Summary: "Retry three times", Confidence: 0.9, References: []string{"Retry"}}},
	}); err != nil {
		t.Fatal(err)
	}
	edges, err := gs.AllEdges(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range edges {
		if e.Src == sym.ID || e.Dst == sym.ID {
			return
		}
	}
	t.Error("the decision was not linked to the Retry symbol")
}

type failingSearch struct{ store.GraphStore }

func (failingSearch) SearchNodes(context.Context, string, int) ([]model.Node, error) {
	return nil, errors.New("boom")
}

// TestSubmitReportsSearchError checks a store failure fails the submit, not records the reference as unresolved.
func TestSubmitReportsSearchError(t *testing.T) {
	gs := newStore(t)
	id := seedMeetingAndCode(t, gs)
	_, err := NewLinkService(failingSearch{gs}, 0.5).Submit(context.Background(), LinkResult{
		TaskID:    id,
		Decisions: []DecisionResult{{Summary: "Cap retries", Confidence: 0.9, References: []string{"fetchAll"}}},
	})
	if err == nil {
		t.Fatal("Submit succeeded although the symbol search failed")
	}
}
