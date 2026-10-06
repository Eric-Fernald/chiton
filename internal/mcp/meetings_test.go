package mcpsrv

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	urlingest "github.com/Eric-Fernald/chiton/internal/ingest/url"
)

// seedLinkedMeeting ingests a meeting + code symbol and links a decision to it,
// returning the meeting node ID.
func seedLinkedMeeting(t *testing.T, gs store.GraphStore) string {
	t.Helper()
	ctx := context.Background()
	tr := &meeting.Transcript{
		Title: "Retry Policy Sync",
		Date:  time.Date(2026, 6, 1, 9, 0, 0, 0, time.UTC),
		Turns: []meeting.Turn{
			{Speaker: "Alice", Text: "We should cap retries in fetchAll."},
			{Speaker: "Bob", Text: "Sounds right."},
		},
	}
	mNodes, mEdges := meeting.ToGraph(tr, time.Now())
	sym := model.Node{
		ID: model.NewNodeID("file://client.go", "fetchAll"), Kind: model.KindSymbol,
		Label: "fetchAll", SourceURI: "file://client.go",
		Provenance: model.ProvExtracted, Confidence: 1,
	}
	if err := gs.UpsertNodes(ctx, append(mNodes, sym)); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, mEdges); err != nil {
		t.Fatal(err)
	}
	svc := meeting.NewLinkService(gs, 0.5)
	if _, err := svc.Submit(ctx, meeting.LinkResult{
		TaskID: mNodes[0].ID,
		Decisions: []meeting.DecisionResult{{
			Summary: "Cap retries at 3 with backoff", Rationale: "Avoid hammering upstream",
			Confidence: 0.9, References: []string{"fetchAll"},
		}},
	}); err != nil {
		t.Fatal(err)
	}
	return mNodes[0].ID
}

func TestSearchMeetingsHandler(t *testing.T) {
	gs := openStore(t)
	seedLinkedMeeting(t, gs)
	ctx := context.Background()

	res, err := searchMeetingsHandler(gs)(ctx, toolReq(map[string]any{"query": "retries"}))
	if err != nil {
		t.Fatal(err)
	}
	var views []map[string]any
	if err := json.Unmarshal([]byte(resultText(t, res)), &views); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, resultText(t, res))
	}
	if len(views) != 1 {
		t.Fatalf("got %d meetings want 1", len(views))
	}
	// PII hidden by default: participants absent, count present.
	if _, ok := views[0]["participants"]; ok {
		t.Errorf("participants leaked without include_pii")
	}
	if pc, _ := views[0]["participant_count"].(float64); pc != 2 {
		t.Errorf("participant_count=%v want 2", views[0]["participant_count"])
	}
	decisions, _ := views[0]["decisions"].([]any)
	if len(decisions) != 1 {
		t.Errorf("decisions=%d want 1", len(decisions))
	}

	// include_pii surfaces names.
	res, _ = searchMeetingsHandler(gs)(ctx, toolReq(map[string]any{"query": "retries", "include_pii": true}))
	_ = json.Unmarshal([]byte(resultText(t, res)), &views)
	parts, _ := views[0]["participants"].([]any)
	if len(parts) != 2 {
		t.Errorf("participants=%v want 2 names with include_pii", parts)
	}
}

func TestSearchMeetingsSinceFilter(t *testing.T) {
	gs := openStore(t)
	seedLinkedMeeting(t, gs) // dated 2026-06-01
	ctx := context.Background()

	res, _ := searchMeetingsHandler(gs)(ctx, toolReq(map[string]any{"since": "2026-07-01"}))
	if got := resultText(t, res); !strings.Contains(got, "no meetings match") {
		t.Errorf("since filter should exclude older meeting, got: %s", got)
	}
}

func TestDecisionsAboutHandler(t *testing.T) {
	gs := openStore(t)
	seedLinkedMeeting(t, gs)
	ctx := context.Background()

	// By governed symbol.
	res, err := decisionsAboutHandler(gs)(ctx, toolReq(map[string]any{"topic": "fetchAll"}))
	if err != nil {
		t.Fatal(err)
	}
	var hits []map[string]any
	if err := json.Unmarshal([]byte(resultText(t, res)), &hits); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, resultText(t, res))
	}
	if len(hits) != 1 {
		t.Fatalf("got %d decisions want 1: %s", len(hits), resultText(t, res))
	}
	if hits[0]["meeting"] != "Retry Policy Sync" {
		t.Errorf("meeting=%v", hits[0]["meeting"])
	}
	governs, _ := hits[0]["governs"].([]any)
	if len(governs) != 1 || governs[0] != "fetchAll" {
		t.Errorf("governs=%v want [fetchAll]", governs)
	}

	// By free-text on the summary.
	res, _ = decisionsAboutHandler(gs)(ctx, toolReq(map[string]any{"topic": "backoff"}))
	if got := resultText(t, res); !strings.Contains(got, "Cap retries") {
		t.Errorf("free-text topic miss: %s", got)
	}

	// No match.
	res, _ = decisionsAboutHandler(gs)(ctx, toolReq(map[string]any{"topic": "nonexistent-thing"}))
	if got := resultText(t, res); !strings.Contains(got, "no decisions found") {
		t.Errorf("expected no-match message, got: %s", got)
	}
}

func TestDecisionsAboutPIIGating(t *testing.T) {
	gs := openStore(t)
	ctx := context.Background()
	tr := &meeting.Transcript{
		Title: "Cache Planning",
		Turns: []meeting.Turn{{Speaker: "Alice", Text: "I'll own the cache work in cacheGet."}},
	}
	mNodes, mEdges := meeting.ToGraph(tr, time.Now())
	sym := model.Node{
		ID: model.NewNodeID("file://cache.go", "cacheGet"), Kind: model.KindSymbol,
		Label: "cacheGet", SourceURI: "file://cache.go",
		Provenance: model.ProvExtracted, Confidence: 1,
	}
	if err := gs.UpsertNodes(ctx, append(mNodes, sym)); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, mEdges); err != nil {
		t.Fatal(err)
	}
	if _, err := meeting.NewLinkService(gs, 0.5).Submit(ctx, meeting.LinkResult{
		TaskID: mNodes[0].ID,
		Decisions: []meeting.DecisionResult{{
			Summary: "Own the cache work", Confidence: 0.9, References: []string{"cacheGet"},
			ActionItems: []meeting.ActionItem{{Text: "implement cacheGet", Assignee: "Alice"}},
		}},
	}); err != nil {
		t.Fatal(err)
	}

	// Default: action text present, assignee (PII) hidden.
	res, _ := decisionsAboutHandler(gs)(ctx, toolReq(map[string]any{"topic": "cacheGet"}))
	txt := resultText(t, res)
	if !strings.Contains(txt, "implement cacheGet") {
		t.Errorf("action text missing: %s", txt)
	}
	if strings.Contains(txt, "Alice") {
		t.Errorf("assignee PII leaked without include_pii: %s", txt)
	}

	// include_pii: assignee surfaced.
	res, _ = decisionsAboutHandler(gs)(ctx, toolReq(map[string]any{"topic": "cacheGet", "include_pii": true}))
	if !strings.Contains(resultText(t, res), "Alice") {
		t.Errorf("assignee should be present with include_pii: %s", resultText(t, res))
	}
}

func TestAddURLHandlerNilIngester(t *testing.T) {
	res, err := addURLHandler(nil)(context.Background(), toolReq(map[string]any{"url": "https://x"}))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(resultText(t, res), "unavailable") {
		t.Errorf("nil ingester should report unavailable, got: %s", resultText(t, res))
	}
}

// A prompt-injected add_url("~/.aws/credentials") must not put the file into the graph.
func TestAddURLHandlerRefusesLocalFiles(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	secret := filepath.Join(t.TempDir(), "credentials")
	if err := os.WriteFile(secret, []byte("aws_secret_access_key = hunter2"), 0o600); err != nil {
		t.Fatal(err)
	}
	h := addURLHandler(urlingest.New(gs))
	for _, raw := range []string{secret, model.FileURI(secret)} {
		res, err := h(ctx, toolReq(map[string]any{"url": raw}))
		if err != nil {
			t.Fatal(err)
		}
		if !res.IsError {
			t.Errorf("add_url(%q) = %s; want a refusal", raw, resultText(t, res))
		}
	}
	if nodes, _ := gs.AllNodes(ctx); len(nodes) != 0 {
		t.Fatalf("add_url ingested a local file: %d nodes", len(nodes))
	}
}
