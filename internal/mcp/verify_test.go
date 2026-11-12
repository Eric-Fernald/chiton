package mcpsrv

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/verify"
)

func TestVerificationToolsNilService(t *testing.T) {
	ctx := context.Background()
	res, err := pendingVerificationsHandler(nil)(ctx, toolReq(nil))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(resultText(t, res), "unavailable") {
		t.Errorf("pending with no service = %s", resultText(t, res))
	}
	res, err = submitVerificationHandler(nil)(ctx, toolReq(map[string]any{"task_id": "x"}))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(resultText(t, res), "unavailable") {
		t.Errorf("submit with no service = %s", resultText(t, res))
	}
}

func TestPendingAndSubmitVerificationRoundTrip(t *testing.T) {
	gs := openStore(t)
	ctx := context.Background()
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "d1", Kind: model.KindDocSection, Label: "Auth", SourceURI: "file://doc.md",
			Attrs: map[string]any{"text": "We authenticate every request with a JWT."}, Provenance: model.ProvExtracted, Confidence: 1},
		{ID: "c1", Kind: model.KindConcept, Label: "JWT", SourceURI: "concept://jwt",
			Attrs: map[string]any{"description": "a token"}, Provenance: model.ProvInferred, Confidence: 0.9},
	}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{{ID: "e1", Src: "d1", Dst: "c1", Kind: model.EdgeDescribes, Provenance: model.ProvInferred}}); err != nil {
		t.Fatal(err)
	}
	svc := verify.New(gs, t.TempDir(), 0.5)

	var specs []verify.TaskSpec
	out := call(t, pendingVerificationsHandler(svc), map[string]any{"kind": verify.KindConceptSupport})
	if err := json.Unmarshal([]byte(out), &specs); err != nil || len(specs) != 1 {
		t.Fatalf("pending_verifications = %s (%v)", out, err)
	}

	out = call(t, submitVerificationHandler(svc), map[string]any{
		"task_id": specs[0].ID, "fingerprint": specs[0].Fingerprint, "verdict": "supported",
		"quote": "every request with a JWT", "confidence": 0.9,
	})
	var rep verify.Report
	if err := json.Unmarshal([]byte(out), &rep); err != nil || !rep.Applied {
		t.Fatalf("submit_verification = %s (%v)", out, err)
	}
	if out = call(t, pendingVerificationsHandler(svc), nil); !strings.Contains(out, "no pending verifications") {
		t.Errorf("after submit, pending_verifications = %s", out)
	}
}

// decisions_about is where "why is this code like this?" gets answered, so recorded drift has to show there.
func TestDecisionsAboutShowsRecordedDrift(t *testing.T) {
	gs := openStore(t)
	ctx := context.Background()
	seedLinkedMeeting(t, gs)
	decisions, err := store.NodesMatching(ctx, gs, store.NodeFilter{Kinds: []model.NodeKind{model.KindDecision}})
	if err != nil || len(decisions) != 1 {
		t.Fatalf("want 1 decision, got %d (%v)", len(decisions), err)
	}
	sym := model.NewNodeID("file://client.go", "fetchAll")
	drift := func(prov model.Provenance) {
		t.Helper()
		if err := gs.UpsertEdges(ctx, []model.Edge{{
			ID: model.NewEdgeID(sym, decisions[0].ID, model.EdgeContradicts), Src: sym, Dst: decisions[0].ID,
			Kind: model.EdgeContradicts, Provenance: prov, Confidence: 0.9,
			Attrs: map[string]any{"note": "retries are capped at 5, not 3"},
		}}); err != nil {
			t.Fatal(err)
		}
	}
	args := map[string]any{"topic": "fetchAll"}

	if out := call(t, decisionsAboutHandler(gs), args); strings.Contains(out, "drift") {
		t.Fatalf("no drift recorded yet, got: %s", out)
	}

	drift(model.ProvAmbiguous)
	if out := call(t, decisionsAboutHandler(gs), args); strings.Contains(out, "capped at 5") {
		t.Errorf("AMBIGUOUS drift should stay hidden, got: %s", out)
	}

	drift(model.ProvInferred)
	var hits []struct {
		Governs []string `json:"governs"`
		Drift   []struct {
			Symbol string `json:"symbol"`
			Note   string `json:"note"`
		} `json:"drift"`
	}
	out := call(t, decisionsAboutHandler(gs), args)
	if err := json.Unmarshal([]byte(out), &hits); err != nil || len(hits) != 1 {
		t.Fatalf("decisions_about = %s (%v)", out, err)
	}
	if len(hits[0].Drift) != 1 || hits[0].Drift[0].Symbol != "fetchAll" || hits[0].Drift[0].Note != "retries are capped at 5, not 3" {
		t.Errorf("drift = %+v", hits[0].Drift)
	}
	if len(hits[0].Governs) != 1 || hits[0].Governs[0] != "fetchAll" {
		t.Errorf("governs = %v, want it unchanged by the drift edge", hits[0].Governs)
	}
}
