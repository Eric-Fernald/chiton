package main

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/extract/subagent"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

func openSubmitStore(t *testing.T) *badgerstore.Store {
	t.Helper()
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	return gs
}

// TestExtractSubmitContinuesPastFailedResult checks one result naming a vanished section doesn't stop the
// results after it, and the command still fails so the caller notices.
func TestExtractSubmitContinuesPastFailedResult(t *testing.T) {
	ctx := context.Background()
	gs := openSubmitStore(t)
	if err := gs.UpsertNodes(ctx, []model.Node{{
		ID: "d1", Kind: model.KindDocSection, Label: "Auth", SourceURI: "file://doc.md",
		Attrs: map[string]any{"text": "We authenticate every request with a JWT."}, Provenance: model.ProvExtracted, Confidence: 1,
	}}); err != nil {
		t.Fatal(err)
	}
	results := []subagent.SubmitResult{
		{TaskID: "gone", Concepts: []subagent.ConceptResult{{Label: "Lost", Confidence: 0.9}}},
		{TaskID: "d1", Concepts: []subagent.ConceptResult{{Label: "JWT", Confidence: 0.9}}},
	}
	var out bytes.Buffer
	err := runExtractSubmit(ctx, subagent.New(gs, nil, 0.5), results, &out)
	if err == nil || !strings.Contains(err.Error(), "1 of 2") {
		t.Errorf("err = %v, want it to report 1 of 2 results failed", err)
	}
	for _, want := range []string{"gone: failed", "d1 — 1 concept(s)", "2 task(s)", "1 failed"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output missing %q:\n%s", want, out.String())
		}
	}
	if nodes, _ := gs.SearchNodes(ctx, "JWT", 0); len(nodes) != 1 {
		t.Errorf("the result after the failed one was not applied: %d JWT node(s)", len(nodes))
	}
}

// TestMeetingSubmitContinuesPastFailedResult is the same for meeting-linkage results.
func TestMeetingSubmitContinuesPastFailedResult(t *testing.T) {
	ctx := context.Background()
	gs := openSubmitStore(t)
	nodes, edges := meeting.ToGraph(&meeting.Transcript{
		Title: "Retry Policy Sync",
		Turns: []meeting.Turn{{Speaker: "Alice", Text: "We decided to cap retries at 3."}},
	}, time.Now())
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, edges); err != nil {
		t.Fatal(err)
	}
	results := []meeting.LinkResult{
		{TaskID: "gone", Decisions: []meeting.DecisionResult{{Summary: "Lost", Confidence: 0.9}}},
		{TaskID: nodes[0].ID, Decisions: []meeting.DecisionResult{{Summary: "Cap retries at 3", Confidence: 0.9}}},
	}
	var out bytes.Buffer
	err := runMeetingSubmit(ctx, meeting.NewLinkService(gs, 0.5), results, &out)
	if err == nil || !strings.Contains(err.Error(), "1 of 2") {
		t.Errorf("err = %v, want it to report 1 of 2 results failed", err)
	}
	for _, want := range []string{"gone: failed", "1 decision(s)", "2 task(s)", "1 failed"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output missing %q:\n%s", want, out.String())
		}
	}
}
