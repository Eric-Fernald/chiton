package main

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
	"github.com/Eric-Fernald/chiton/internal/verify"
)

func TestVerifySubmitAppliesEachResultAndSummarises(t *testing.T) {
	ctx := context.Background()
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "d1", Kind: model.KindDocSection, Label: "Auth", SourceURI: "file://doc.md",
			Attrs: map[string]any{"text": "We authenticate every request with a JWT."}, Provenance: model.ProvExtracted, Confidence: 1},
		{ID: "jwt", Kind: model.KindConcept, Label: "JWT", SourceURI: "concept://jwt", Provenance: model.ProvInferred, Confidence: 0.9},
		{ID: "oauth", Kind: model.KindConcept, Label: "OAuth", SourceURI: "concept://oauth", Provenance: model.ProvInferred, Confidence: 0.9},
	}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{
		{ID: "e1", Src: "d1", Dst: "jwt", Kind: model.EdgeDescribes, Provenance: model.ProvInferred},
		{ID: "e2", Src: "d1", Dst: "oauth", Kind: model.EdgeDescribes, Provenance: model.ProvInferred},
	}); err != nil {
		t.Fatal(err)
	}
	svc := verify.New(gs, t.TempDir(), 0.5)
	specs, err := svc.Pending(ctx, "", 0)
	if err != nil || len(specs) != 2 {
		t.Fatalf("want 2 pending tasks, got %d (%v)", len(specs), err)
	}
	var results []verify.Result
	for _, s := range specs {
		r := verify.Result{TaskID: s.ID, Fingerprint: s.Fingerprint, Verdict: verify.Unsupported, Confidence: 0.9}
		if s.ID == "jwt" {
			r.Verdict, r.Quote = verify.Supported, "every request with a JWT"
		}
		results = append(results, r)
	}

	var out bytes.Buffer
	if err := runVerifySubmit(ctx, svc, results, &out); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"jwt: supported", "oauth: unsupported", "demoted to AMBIGUOUS", "2 task(s), 2 applied"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output missing %q:\n%s", want, out.String())
		}
	}
	if n, _ := gs.GetNode(ctx, "oauth"); n.Provenance != model.ProvAmbiguous {
		t.Errorf("oauth provenance = %s, want AMBIGUOUS", n.Provenance)
	}
}

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
