package teams

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

// fakeSource serves canned transcripts.
type fakeSource struct {
	metas   []TranscriptMeta
	content map[string][]byte
}

func (f *fakeSource) ListTranscripts(_ context.Context, _ string) ([]TranscriptMeta, error) {
	return f.metas, nil
}
func (f *fakeSource) FetchContent(_ context.Context, _, tid string) ([]byte, error) {
	return f.content[tid], nil
}

func TestIngestMeeting(t *testing.T) {
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	vtt := "WEBVTT\n\n00:00:00.000 --> 00:00:03.000\n<v Alice>We ship Friday.</v>\n\n" +
		"00:00:03.000 --> 00:00:05.000\n<v Bob>Agreed.</v>\n"
	src := &fakeSource{
		metas:   []TranscriptMeta{{ID: "t1", MeetingID: "m1", CreatedAt: time.Date(2026, 6, 1, 9, 0, 0, 0, time.UTC)}},
		content: map[string][]byte{"t1": []byte(vtt)},
	}
	ing := NewIngester(gs, src)

	rep, err := ing.IngestMeeting(context.Background(), "m1")
	if err != nil {
		t.Fatal(err)
	}
	if rep.Transcripts != 1 || rep.Turns != 2 {
		t.Errorf("report=%+v want 1 transcript, 2 turns", rep)
	}

	nodes, _ := gs.AllNodes(context.Background())
	var meetingURI string
	kinds := map[model.NodeKind]int{}
	for _, n := range nodes {
		kinds[n.Kind]++
		if n.Kind == model.KindMeeting {
			meetingURI = n.SourceURI
		}
	}
	want := MeetingTranscriptURI("m1", "t1")
	if meetingURI != want {
		t.Errorf("meeting uri=%q want %q", meetingURI, want)
	}
	if kinds[model.KindPerson] != 2 || kinds[model.KindMeetingTurn] != 2 {
		t.Errorf("kinds=%v want 2 persons, 2 turns", kinds)
	}

	// Re-ingest must replace, not duplicate.
	if _, err := ing.IngestMeeting(context.Background(), "m1"); err != nil {
		t.Fatal(err)
	}
	nodes, _ = gs.AllNodes(context.Background())
	meetings := 0
	for _, n := range nodes {
		if n.Kind == model.KindMeeting {
			meetings++
		}
	}
	if meetings != 1 {
		t.Errorf("meetings after re-ingest=%d want 1", meetings)
	}
}

func TestWebhookValidationHandshake(t *testing.T) {
	h := NewWebhookHandler("secret", nil, nil)
	req := httptest.NewRequest(http.MethodPost, "/hook?validationToken=abc123", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || rec.Body.String() != "abc123" {
		t.Errorf("validation handshake: code=%d body=%q want 200/abc123", rec.Code, rec.Body.String())
	}
}

func TestWebhookDispatchesValidNotifications(t *testing.T) {
	var got []ChangeItem
	h := NewWebhookHandler("s3cr3t", func(_ context.Context, c ChangeItem) error {
		got = append(got, c)
		return nil
	}, nil)

	body := `{"value":[
	  {"subscriptionId":"sub1","changeType":"created","clientState":"s3cr3t",
	   "resource":"communications/onlineMeetings('MEET1')/transcripts('TID1')"},
	  {"subscriptionId":"sub2","changeType":"created","clientState":"WRONG",
	   "resource":"communications/onlineMeetings('MEET2')/transcripts('TID2')"}
	]}`
	req := httptest.NewRequest(http.MethodPost, "/hook", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusAccepted {
		t.Errorf("code=%d want 202", rec.Code)
	}
	if len(got) != 1 {
		t.Fatalf("dispatched %d changes want 1 (bad clientState must be dropped)", len(got))
	}
	mid, tid, ok := ParseTranscriptResource(got[0].Resource)
	if !ok || mid != "MEET1" || tid != "TID1" {
		t.Errorf("parsed resource=(%q,%q,%v) want MEET1/TID1/true", mid, tid, ok)
	}
}

func TestParseTranscriptResource(t *testing.T) {
	mid, tid, ok := ParseTranscriptResource("users('U1')/onlineMeetings('M1')/transcripts('T1')")
	if !ok || mid != "M1" || tid != "T1" {
		t.Errorf("got (%q,%q,%v)", mid, tid, ok)
	}
	if _, _, ok := ParseTranscriptResource("users('U1')/chats('C1')"); ok {
		t.Error("non-transcript resource should not parse")
	}
}
