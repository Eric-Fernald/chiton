package main

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

func TestVerifySweepDemotesOrphansAndReportsLostDecisionLinks(t *testing.T) {
	ctx := context.Background()
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "orphan", Kind: model.KindConcept, Label: "Session", SourceURI: "concept://session",
			Provenance: model.ProvInferred, Confidence: 0.9},
		{ID: "dec", Kind: model.KindDecision, Label: "Cap retries at 3", SourceURI: "decision://m",
			Provenance: model.ProvInferred, Confidence: 0.9,
			Attrs: map[string]any{"meeting_id": "m", "governs": []map[string]any{{"label": "fetchAll", "node_id": "gone"}}}},
	}); err != nil {
		t.Fatal(err)
	}

	var out bytes.Buffer
	if err := runVerifySweep(ctx, gs, 0.5, &out); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"1 concept(s) demoted", "0 decision link(s) restored", "fetchAll"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output missing %q:\n%s", want, out.String())
		}
	}
	if n, _ := gs.GetNode(ctx, "orphan"); n.Provenance != model.ProvAmbiguous {
		t.Errorf("orphan provenance = %s, want AMBIGUOUS", n.Provenance)
	}
}
