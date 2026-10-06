package slack

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

// fakeSource serves a canned conversation.
type fakeSource struct {
	conv Conversation
	err  error
}

func (f *fakeSource) Name() string { return "slack" }
func (f *fakeSource) FetchConversation(_ context.Context, channelID, threadTS string) (Conversation, error) {
	if f.err != nil {
		return Conversation{}, f.err
	}
	c := f.conv
	if c.ChannelID == "" {
		c.ChannelID = channelID
	}
	c.ThreadTS = threadTS
	return c, nil
}

func TestIngestConversation(t *testing.T) {
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	conv := Conversation{
		ChannelID:   "C1",
		ChannelName: "eng",
		Messages: []Message{
			{User: "U1", Name: "Alice", Text: "Let's switch the retry policy to exponential backoff.", TS: "1735689600.000100"},
			{User: "U2", Name: "Bob", Text: "Agreed, I'll update fetchAll.", TS: "1735689660.000200"},
			{User: "U1", Name: "Alice", Text: "   ", TS: "1735689700.000300"}, // blank → skipped
		},
	}
	ing := NewIngester(gs, &fakeSource{conv: conv})

	rep, err := ing.IngestConversation(context.Background(), "C1", "")
	if err != nil {
		t.Fatal(err)
	}
	if rep.Messages != 3 || rep.Turns != 2 {
		t.Errorf("report=%+v want 3 messages, 2 turns", rep)
	}

	nodes, _ := gs.AllNodes(context.Background())
	kinds := map[model.NodeKind]int{}
	var meetingURI string
	for _, n := range nodes {
		kinds[n.Kind]++
		if n.Kind == model.KindMeeting {
			meetingURI = n.SourceURI
		}
	}
	if want := ConversationURI("C1", ""); meetingURI != want {
		t.Errorf("meeting uri=%q want %q", meetingURI, want)
	}
	if kinds[model.KindMeeting] != 1 || kinds[model.KindMeetingTurn] != 2 || kinds[model.KindPerson] != 2 {
		t.Errorf("kinds=%v want 1 meeting, 2 turns, 2 persons", kinds)
	}

	// Re-ingest must replace, not duplicate.
	if _, err := ing.IngestConversation(context.Background(), "C1", ""); err != nil {
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

func TestConversationURI(t *testing.T) {
	cases := []struct{ ch, th, want string }{
		{"C1", "", "slack://C1"},
		{"C1", "1735689600.000100", "slack://C1/1735689600.000100"},
	}
	for _, c := range cases {
		if got := ConversationURI(c.ch, c.th); got != c.want {
			t.Errorf("ConversationURI(%q,%q)=%q want %q", c.ch, c.th, got, c.want)
		}
	}
}

func TestConversationTitleThreadDiscriminator(t *testing.T) {
	a := conversationTitle(Conversation{ChannelName: "eng", ThreadTS: "1735689600.000100"})
	b := conversationTitle(Conversation{ChannelName: "eng", ThreadTS: "1735700000.000100"})
	if a == b {
		t.Errorf("distinct threads share a title: %q", a)
	}
	if !strings.Contains(a, "#eng") {
		t.Errorf("title %q missing channel", a)
	}
}

func TestParseSlackTS(t *testing.T) {
	got, err := parseSlackTS("1735689600.000600")
	if err != nil {
		t.Fatal(err)
	}
	if want := time.Unix(1735689600, 600*1000).UTC(); !got.Equal(want) {
		t.Errorf("parseSlackTS=%v want %v", got, want)
	}
	if _, err := parseSlackTS("nope"); err == nil {
		t.Error("expected error for bad ts")
	}
}

// --- Events API receiver ---

func signSlack(secret, ts string, body []byte) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte("v0:" + ts + ":"))
	mac.Write(body)
	return "v0=" + hex.EncodeToString(mac.Sum(nil))
}

func TestEventsURLVerification(t *testing.T) {
	const secret = "shh"
	h := NewEventsHandler(secret, nil, nil)
	defer h.Close()
	h.now = func() time.Time { return time.Unix(1735689600, 0) }

	body := []byte(`{"type":"url_verification","challenge":"abc123"}`)
	ts := "1735689600"
	req := httptest.NewRequest(http.MethodPost, "/slack/events", strings.NewReader(string(body)))
	req.Header.Set("X-Slack-Request-Timestamp", ts)
	req.Header.Set("X-Slack-Signature", signSlack(secret, ts, body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || rec.Body.String() != "abc123" {
		t.Errorf("challenge: code=%d body=%q want 200/abc123", rec.Code, rec.Body.String())
	}
}

func TestEventsDispatchesMessage(t *testing.T) {
	const secret = "topsecret"
	var got []MessageEvent
	h := NewEventsHandler(secret, func(_ context.Context, ev MessageEvent) error {
		got = append(got, ev)
		return nil
	}, nil)
	h.now = func() time.Time { return time.Unix(1735689600, 0) }

	body := []byte(`{"type":"event_callback","event":{"type":"message","channel":"C1","user":"U1","text":"ship it","ts":"1735689600.000100"}}`)
	ts := "1735689600"
	req := httptest.NewRequest(http.MethodPost, "/slack/events", strings.NewReader(string(body)))
	req.Header.Set("X-Slack-Request-Timestamp", ts)
	req.Header.Set("X-Slack-Signature", signSlack(secret, ts, body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	h.Close() // runs the queued re-ingest

	if rec.Code != http.StatusOK {
		t.Errorf("code=%d want 200", rec.Code)
	}
	if len(got) != 1 || got[0].ChannelID != "C1" || got[0].Text != "ship it" {
		t.Fatalf("dispatched=%+v want 1 message for C1", got)
	}
}

func TestEventsRejectsBadSignature(t *testing.T) {
	const secret = "topsecret"
	called := false
	h := NewEventsHandler(secret, func(_ context.Context, _ MessageEvent) error { called = true; return nil }, nil)
	h.now = func() time.Time { return time.Unix(1735689600, 0) }

	body := []byte(`{"type":"event_callback","event":{"type":"message","channel":"C1","user":"U1","text":"hi","ts":"1735689600.000100"}}`)
	req := httptest.NewRequest(http.MethodPost, "/slack/events", strings.NewReader(string(body)))
	req.Header.Set("X-Slack-Request-Timestamp", "1735689600")
	req.Header.Set("X-Slack-Signature", "v0=deadbeef")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	h.Close()

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("code=%d want 401", rec.Code)
	}
	if called {
		t.Error("onMessage must not be called on bad signature")
	}
}

func TestEventsRejectsStaleTimestamp(t *testing.T) {
	const secret = "topsecret"
	called := false
	h := NewEventsHandler(secret, func(_ context.Context, _ MessageEvent) error { called = true; return nil }, nil)
	h.now = func() time.Time { return time.Unix(1735689600, 0) }

	body := []byte(`{"type":"event_callback","event":{"type":"message","channel":"C1","user":"U1","text":"hi","ts":"1"}}`)
	staleTS := "1735680000" // ~2.7h earlier, beyond the 5-minute window
	req := httptest.NewRequest(http.MethodPost, "/slack/events", strings.NewReader(string(body)))
	req.Header.Set("X-Slack-Request-Timestamp", staleTS)
	req.Header.Set("X-Slack-Signature", signSlack(secret, staleTS, body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	h.Close()

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("code=%d want 401 for stale ts", rec.Code)
	}
	if called {
		t.Error("onMessage must not be called for a stale request")
	}
}

func TestEventsFailsClosedWithoutSecret(t *testing.T) {
	called := false
	h := NewEventsHandler("", func(_ context.Context, _ MessageEvent) error { called = true; return nil }, nil)
	h.now = func() time.Time { return time.Unix(1735689600, 0) }

	body := []byte(`{"type":"event_callback","event":{"type":"message","channel":"C1","user":"U1","text":"hi","ts":"1735689600.000100"}}`)
	req := httptest.NewRequest(http.MethodPost, "/slack/events", strings.NewReader(string(body)))
	req.Header.Set("X-Slack-Request-Timestamp", "1735689600")
	// Even a "correctly" computed signature over an empty secret must be rejected.
	req.Header.Set("X-Slack-Signature", signSlack("", "1735689600", body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	h.Close()

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("code=%d want 401 (fail closed when no secret configured)", rec.Code)
	}
	if called {
		t.Error("onMessage must not be called when no secret is configured")
	}
}

// --- Web API client ---

func TestWebClientFetchConversation(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/conversations.history", func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer xoxb-test" {
			t.Errorf("auth header=%q want Bearer xoxb-test", got)
		}
		writeJSON(w, map[string]any{
			"ok": true,
			"messages": []map[string]any{ // newest-first, as Slack returns
				{"type": "message", "user": "U2", "text": "second &amp; final", "ts": "1735689660.000200"},
				{"type": "message", "user": "U1", "text": "first", "ts": "1735689600.000100"},
				{"type": "message", "subtype": "channel_join", "user": "U3", "text": "joined", "ts": "1735689500.000050"},
			},
		})
	})
	mux.HandleFunc("/users.info", func(w http.ResponseWriter, r *http.Request) {
		uid := r.URL.Query().Get("user")
		writeJSON(w, map[string]any{
			"ok":   true,
			"user": map[string]any{"id": uid, "profile": map[string]any{"display_name": "name-" + uid}},
		})
	})
	mux.HandleFunc("/conversations.info", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"ok": true, "channel": map[string]any{"id": r.URL.Query().Get("channel"), "name": "eng"}})
	})
	c := newTestClient(t, mux)

	conv, err := c.FetchConversation(context.Background(), "C1", "")
	if err != nil {
		t.Fatal(err)
	}
	if conv.ChannelName != "eng" {
		t.Errorf("channel name=%q want eng", conv.ChannelName)
	}
	if len(conv.Messages) != 2 {
		t.Fatalf("messages=%d want 2 (channel_join filtered)", len(conv.Messages))
	}
	if conv.Messages[0].Text != "first" || conv.Messages[1].Text != "second & final" {
		t.Errorf("messages not chronological/cleaned: %+v", conv.Messages)
	}
	if conv.Messages[0].Name != "name-U1" {
		t.Errorf("user not resolved: %+v", conv.Messages[0])
	}
}

func TestWebClientOKFalseIsError(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/conversations.info", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"ok": true, "channel": map[string]any{"name": "eng"}})
	})
	mux.HandleFunc("/conversations.history", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"ok": false, "error": "channel_not_found"})
	})
	c := newTestClient(t, mux)

	if _, err := c.FetchConversation(context.Background(), "C1", ""); err == nil ||
		!strings.Contains(err.Error(), "channel_not_found") {
		t.Errorf("err=%v want channel_not_found", err)
	}
}

func TestWebClientPaginationAndReplyDedup(t *testing.T) {
	// conversations.replies repeats the thread parent at the top of every page;
	// the client must page via the cursor and de-dup the parent by ts.
	var calls int32
	mux := http.NewServeMux()
	mux.HandleFunc("/conversations.info", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"ok": true, "channel": map[string]any{"name": "eng"}})
	})
	mux.HandleFunc("/users.info", func(w http.ResponseWriter, r *http.Request) {
		uid := r.URL.Query().Get("user")
		writeJSON(w, map[string]any{"ok": true, "user": map[string]any{"id": uid, "profile": map[string]any{"display_name": uid}}})
	})
	mux.HandleFunc("/conversations.replies", func(w http.ResponseWriter, r *http.Request) {
		parent := map[string]any{"type": "message", "user": "U0", "text": "root", "ts": "100.0001", "thread_ts": "100.0001"}
		if r.URL.Query().Get("cursor") == "" {
			atomic.AddInt32(&calls, 1)
			writeJSON(w, map[string]any{
				"ok":                true,
				"messages":          []map[string]any{parent, {"type": "message", "user": "U1", "text": "reply-1", "ts": "101.0001"}},
				"response_metadata": map[string]any{"next_cursor": "CURSOR2"},
			})
			return
		}
		atomic.AddInt32(&calls, 1)
		writeJSON(w, map[string]any{ // page 2 repeats the parent
			"ok":       true,
			"messages": []map[string]any{parent, {"type": "message", "user": "U2", "text": "reply-2", "ts": "102.0001"}},
		})
	})
	c := newTestClient(t, mux)

	conv, err := c.FetchConversation(context.Background(), "C1", "100.0001")
	if err != nil {
		t.Fatal(err)
	}
	if atomic.LoadInt32(&calls) != 2 {
		t.Errorf("pages fetched=%d want 2", calls)
	}
	if len(conv.Messages) != 3 { // root + reply-1 + reply-2, parent de-duped
		t.Fatalf("messages=%d want 3 (parent de-duped): %+v", len(conv.Messages), conv.Messages)
	}
	if conv.Messages[0].Text != "root" || conv.Messages[2].Text != "reply-2" {
		t.Errorf("unexpected order/content: %+v", conv.Messages)
	}
}

func TestWebClientBotMessageAndResolutionFailureDoNotAbort(t *testing.T) {
	mux := http.NewServeMux()
	// All resolution endpoints fail; FetchConversation must still succeed.
	mux.HandleFunc("/conversations.info", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"ok": false, "error": "missing_scope"})
	})
	mux.HandleFunc("/users.info", func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})
	mux.HandleFunc("/conversations.history", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{
			"ok": true,
			"messages": []map[string]any{
				{"type": "message", "subtype": "bot_message", "username": "CI Bot", "text": "build green", "ts": "1735689600.000100"},
				{"type": "message", "user": "U1", "text": "thanks", "ts": "1735689660.000200"},
			},
		})
	})
	c := newTestClient(t, mux)

	conv, err := c.FetchConversation(context.Background(), "C1", "")
	if err != nil {
		t.Fatalf("resolution failures must not abort: %v", err)
	}
	if len(conv.Messages) != 2 {
		t.Fatalf("messages=%d want 2", len(conv.Messages))
	}
	if conv.Messages[0].Name != "CI Bot" { // bot username used
		t.Errorf("bot name=%q want CI Bot", conv.Messages[0].Name)
	}
	if conv.Messages[1].Name != "U1" { // failed users.info → falls back to id
		t.Errorf("fallback name=%q want U1", conv.Messages[1].Name)
	}
}

func TestResolveUserNegativeCacheRetries(t *testing.T) {
	var calls int32
	mux := http.NewServeMux()
	mux.HandleFunc("/users.info", func(w http.ResponseWriter, _ *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			http.Error(w, "boom", http.StatusInternalServerError) // first call fails
			return
		}
		writeJSON(w, map[string]any{"ok": true, "user": map[string]any{"id": "U1", "profile": map[string]any{"display_name": "Alice"}}})
	})
	c := newTestClient(t, mux)
	now := time.Unix(1735689600, 0)
	c.now = func() time.Time { return now }

	if got := c.resolveUser(context.Background(), "U1"); got != "U1" {
		t.Errorf("first resolve=%q want fallback U1", got)
	}
	// Within the TTL the negative result is reused (no new call).
	if got := c.resolveUser(context.Background(), "U1"); got != "U1" || atomic.LoadInt32(&calls) != 1 {
		t.Errorf("cached negative not reused: name=%q calls=%d", got, calls)
	}
	// After the TTL it retries and now succeeds.
	now = now.Add(2 * userNegativeTTL)
	if got := c.resolveUser(context.Background(), "U1"); got != "Alice" || atomic.LoadInt32(&calls) != 2 {
		t.Errorf("negative cache did not expire: name=%q calls=%d", got, calls)
	}
}

func TestWebClientConcurrentResolveNoRace(t *testing.T) {
	// The Events receiver shares one WebClient across handlers. Without the cache mutex this triggers the
	// runtime's "concurrent map writes" fatal error even without -race.
	mux := http.NewServeMux()
	mux.HandleFunc("/users.info", func(w http.ResponseWriter, r *http.Request) {
		uid := r.URL.Query().Get("user")
		writeJSON(w, map[string]any{"ok": true, "user": map[string]any{"id": uid, "profile": map[string]any{"display_name": "n-" + uid}}})
	})
	c := newTestClient(t, mux)

	var wg sync.WaitGroup
	for g := range 16 {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := range 50 {
				_ = c.resolveUser(context.Background(), "U"+strconv.Itoa((g+i)%8))
			}
		}(g)
	}
	wg.Wait()
	if got := c.resolveUser(context.Background(), "U0"); got != "n-U0" {
		t.Errorf("resolve after concurrency=%q want n-U0", got)
	}
}

func newTestClient(t *testing.T, mux *http.ServeMux) *WebClient {
	t.Helper()
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	c, err := NewWebClient("xoxb-test")
	if err != nil {
		t.Fatal(err)
	}
	c.base = srv.URL
	return c
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	b, _ := json.Marshal(v)
	_, _ = w.Write(b)
}

// postEvent signs body as Slack would at 1735689600 and serves it, returning the status code.
func postEvent(t *testing.T, h *EventsHandler, secret, body, retryNum string) int {
	t.Helper()
	const ts = "1735689600"
	req := httptest.NewRequest(http.MethodPost, "/slack/events", strings.NewReader(body))
	req.Header.Set("X-Slack-Request-Timestamp", ts)
	req.Header.Set("X-Slack-Signature", signSlack(secret, ts, []byte(body)))
	if retryNum != "" {
		req.Header.Set("X-Slack-Retry-Num", retryNum)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec.Code
}

// TestEventsAcksBeforeIngest checks the 200 goes out while the ingest is still blocked, well inside Slack's 3s.
func TestEventsAcksBeforeIngest(t *testing.T) {
	const secret = "topsecret"
	release := make(chan struct{})
	h := NewEventsHandler(secret, func(context.Context, MessageEvent) error { <-release; return nil }, nil)
	h.now = func() time.Time { return time.Unix(1735689600, 0) }
	body := `{"type":"event_callback","event_id":"Ev1","event":{"type":"message","channel":"C1","user":"U1","text":"hi","ts":"1735689600.000100"}}`

	done := make(chan int, 1)
	go func() { done <- postEvent(t, h, secret, body, "") }()
	select {
	case code := <-done:
		if code != http.StatusOK {
			t.Errorf("code=%d want 200", code)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("ServeHTTP blocked on the ingest; Slack retries anything not acked within 3s")
	}
	close(release)
	h.Close()
}

// TestEventsCoalescesConversationBurst checks a burst in one thread becomes a single re-ingest of the newest event.
func TestEventsCoalescesConversationBurst(t *testing.T) {
	prev := eventDebounce
	eventDebounce = time.Minute // only Close's drain runs the queue, so the burst surely coalesces
	t.Cleanup(func() { eventDebounce = prev })
	const secret = "topsecret"
	var mu sync.Mutex
	var got []MessageEvent
	h := NewEventsHandler(secret, func(_ context.Context, ev MessageEvent) error {
		mu.Lock()
		defer mu.Unlock()
		got = append(got, ev)
		return nil
	}, nil)
	h.now = func() time.Time { return time.Unix(1735689600, 0) }
	for i, ev := range []string{
		`{"type":"message","channel":"C1","user":"U1","text":"a","ts":"1735689600.000200","thread_ts":"1735689600.000100"}`,
		`{"type":"message","channel":"C1","user":"U2","text":"b","ts":"1735689600.000300","thread_ts":"1735689600.000100"}`,
		`{"type":"message","channel":"C1","user":"U1","text":"c","ts":"1735689600.000400","thread_ts":"1735689600.000100"}`,
		`{"type":"message","channel":"C2","user":"U1","text":"d","ts":"1735689600.000500"}`,
	} {
		body := `{"type":"event_callback","event_id":"Ev` + strconv.Itoa(i) + `","event":` + ev + `}`
		if code := postEvent(t, h, secret, body, ""); code != http.StatusOK {
			t.Fatalf("event %d: code=%d", i, code)
		}
	}
	h.Close()

	if len(got) != 2 {
		t.Fatalf("re-ingests = %+v, want 2 (one per conversation)", got)
	}
	for _, ev := range got {
		if ev.ChannelID == "C1" && (ev.Text != "c" || ev.Thread() != "1735689600.000100") {
			t.Errorf("thread re-ingest used %+v, want the newest event of the thread", ev)
		}
	}
}

func TestEventsDropsRedeliveredEvent(t *testing.T) {
	prev := eventDebounce
	eventDebounce = time.Millisecond
	t.Cleanup(func() { eventDebounce = prev })
	const secret = "topsecret"
	var calls atomic.Int32
	ran := make(chan struct{}, 4)
	h := NewEventsHandler(secret, func(context.Context, MessageEvent) error {
		calls.Add(1)
		ran <- struct{}{}
		return nil
	}, nil)
	h.now = func() time.Time { return time.Unix(1735689600, 0) }
	body := `{"type":"event_callback","event_id":"Ev1","event":{"type":"message","channel":"C1","user":"U1","text":"hi","ts":"1735689600.000100"}}`

	postEvent(t, h, secret, body, "")
	select {
	case <-ran:
	case <-time.After(2 * time.Second):
		t.Fatal("first delivery was never ingested")
	}
	if code := postEvent(t, h, secret, body, "1"); code != http.StatusOK {
		t.Fatalf("retry code=%d want 200 (ack so Slack stops retrying)", code)
	}
	h.Close()
	if n := calls.Load(); n != 1 {
		t.Errorf("onMessage ran %d times, want 1 (the redelivery of Ev1 must be dropped)", n)
	}
}

func TestRecentIDsEvictsOldest(t *testing.T) {
	r := newRecentIDs(2)
	if !r.add("a") || !r.add("b") || r.add("a") {
		t.Fatal("a and b must be new once, then a seen")
	}
	if !r.add("c") || !r.add("a") {
		t.Fatal("adding c must evict a, the oldest")
	}
	if !r.add("") || !r.add("") {
		t.Fatal("an empty id is always new")
	}
}

// TestEventsCloseCancelsStuckIngest checks Close aborts an ingest that outlives the grace period.
func TestEventsCloseCancelsStuckIngest(t *testing.T) {
	prev := closeGrace
	closeGrace = 20 * time.Millisecond
	prevDebounce := eventDebounce
	eventDebounce = time.Millisecond
	t.Cleanup(func() { closeGrace, eventDebounce = prev, prevDebounce })
	const secret = "topsecret"
	started := make(chan struct{})
	aborted := make(chan error, 1)
	h := NewEventsHandler(secret, func(ctx context.Context, _ MessageEvent) error {
		close(started)
		<-ctx.Done() // stands in for a 429 wait or a slow fetch
		aborted <- ctx.Err()
		return ctx.Err()
	}, nil)
	h.now = func() time.Time { return time.Unix(1735689600, 0) }
	body := `{"type":"event_callback","event_id":"Ev1","event":{"type":"message","channel":"C1","user":"U1","text":"hi","ts":"1735689600.000100"}}`
	postEvent(t, h, secret, body, "")
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("the queued ingest never started")
	}
	h.Close() // the running ingest blocks until the grace period cancels it
	select {
	case err := <-aborted:
		if err != context.Canceled {
			t.Errorf("ingest ctx err = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Close returned without aborting the stuck ingest")
	}
}
