package slack

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Eric-Fernald/chiton/internal/jobs"
)

// eventDebounce coalesces a burst of messages in one conversation into a single re-ingest. A var so tests can change it.
var eventDebounce = time.Second

// seenEventsCap bounds how many recent event_ids the handler remembers to drop Slack's redeliveries.
const seenEventsCap = 1024

// maxEventBody caps the request body read from Slack.
const maxEventBody = 4 << 20

// replayWindow bounds how far a request timestamp may deviate from now before
// it is rejected as a replay (Slack's recommended bound).
const replayWindow = 5 * time.Minute

// MessageEvent is the subset of a Slack message event chiton acts on.
type MessageEvent struct {
	ChannelID string
	User      string
	Text      string
	TS        string
	ThreadTS  string
}

// Thread returns the thread ts whose conversation ev belongs to, or "" for a top-level channel message.
func (ev MessageEvent) Thread() string {
	if ev.ThreadTS != "" && ev.ThreadTS != ev.TS {
		return ev.ThreadTS
	}
	return ""
}

// EventsHandler receives Slack Events API callbacks. It verifies each request's X-Slack-Signature (HMAC over the raw
// body, within a replay window), answers url_verification, and queues message events for onMessage after acking.
type EventsHandler struct {
	signingSecret string
	onMessage     func(ctx context.Context, ev MessageEvent) error
	log           func(format string, args ...any)
	now           func() time.Time

	queue   *jobs.Queue
	seen    *recentIDs
	mu      sync.Mutex
	pending map[string]MessageEvent // newest event per conversation, taken by the queue worker
}

// NewEventsHandler builds a handler that authenticates every request with the Slack signing secret. An empty secret
// fails closed. onMessage handles one verified message event, later, on a single background worker. Call Close when done.
func NewEventsHandler(signingSecret string, onMessage func(ctx context.Context, ev MessageEvent) error, log func(string, ...any)) *EventsHandler {
	if log == nil {
		log = func(string, ...any) {}
	}
	if signingSecret == "" {
		// Every Slack app has a signing secret, so an empty one is a misconfiguration. Fail closed rather than
		// authenticate forged requests.
		log("slack events: WARNING — no signing secret configured; ALL requests will be rejected (set CHITON_SLACK_SIGNING_SECRET)")
	}
	h := &EventsHandler{
		signingSecret: signingSecret,
		onMessage:     onMessage,
		log:           log,
		now:           time.Now,
		seen:          newRecentIDs(seenEventsCap),
		pending:       map[string]MessageEvent{},
	}
	// One worker, since every ingest writes the shared store; a background ctx, since a request's ends at the ack.
	h.queue = jobs.New(h.ingest, jobs.Options{
		Workers:  1,
		Debounce: eventDebounce,
		OnError:  func(key string, err error) { log("slack events: re-ingest of %s failed: %v", key, err) },
	})
	h.queue.Start(context.Background())
	return h
}

// Close stops accepting events and finishes every queued re-ingest. Call it after the server stops, before the store closes.
func (h *EventsHandler) Close() { h.queue.Drain() }

// enqueue records ev as its conversation's newest event and schedules a coalesced re-ingest.
func (h *EventsHandler) enqueue(ev MessageEvent) {
	key := ConversationURI(ev.ChannelID, ev.Thread())
	h.mu.Lock()
	h.pending[key] = ev
	h.mu.Unlock()
	h.queue.Enqueue(key)
}

// ingest runs onMessage for key's newest pending event. It is the queue's handler.
func (h *EventsHandler) ingest(ctx context.Context, key string) error {
	h.mu.Lock()
	ev, ok := h.pending[key]
	delete(h.pending, key)
	h.mu.Unlock()
	if !ok || h.onMessage == nil {
		return nil
	}
	return h.onMessage(ctx, ev)
}

func (h *EventsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxEventBody))
	if err != nil {
		http.Error(w, "read body", http.StatusBadRequest)
		return
	}

	// Verify the signature over the RAW body before parsing anything — this
	// authenticates both the url_verification handshake and event callbacks.
	if !h.verify(r.Header, body) {
		h.log("slack events: rejecting request with bad signature")
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	var env eventEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		http.Error(w, "invalid event", http.StatusBadRequest)
		return
	}

	switch env.Type {
	case "url_verification":
		// Echo the challenge verbatim as text/plain so Slack can confirm the
		// request URL — the analogue of the Teams validationToken handshake.
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		if _, err := io.WriteString(w, env.Challenge); err != nil {
			h.log("slack events: failed to write challenge: %v", err)
		}
	case "event_callback":
		// Only cheap work happens here; the re-ingest runs on the queue, so the ack always beats Slack's 3s deadline.
		if !h.seen.add(env.EventID) {
			h.log("slack events: dropping redelivered event %s (retry %s: %s)", env.EventID,
				r.Header.Get("X-Slack-Retry-Num"), r.Header.Get("X-Slack-Retry-Reason"))
			w.WriteHeader(http.StatusOK)
			return
		}
		var ev slackInnerEvent
		if err := json.Unmarshal(env.Event, &ev); err != nil {
			h.log("slack events: invalid inner event: %v", err)
			w.WriteHeader(http.StatusOK) // ack so Slack does not retry junk
			return
		}
		if h.shouldDispatch(ev) && h.onMessage != nil {
			h.enqueue(MessageEvent{ChannelID: ev.Channel, User: ev.User, Text: ev.Text, TS: ev.TS, ThreadTS: ev.ThreadTS})
		}
		w.WriteHeader(http.StatusOK)
	default:
		w.WriteHeader(http.StatusOK)
	}
}

// verify checks X-Slack-Signature = "v0=" + HMAC_SHA256(secret, "v0:"+timestamp+":"+body) in constant time,
// rejecting timestamps outside the replay window.
func (h *EventsHandler) verify(header http.Header, body []byte) bool {
	if h.signingSecret == "" {
		return false // fail closed: cannot authenticate without a secret
	}
	tsStr := header.Get("X-Slack-Request-Timestamp")
	sig := header.Get("X-Slack-Signature")
	if tsStr == "" || sig == "" {
		return false
	}
	tsSec, err := strconv.ParseInt(tsStr, 10, 64)
	if err != nil {
		return false
	}
	delta := h.now().Unix() - tsSec
	if delta < 0 {
		delta = -delta
	}
	if delta > int64(replayWindow/time.Second) {
		return false
	}
	mac := hmac.New(sha256.New, []byte(h.signingSecret))
	mac.Write([]byte("v0:" + tsStr + ":"))
	mac.Write(body)
	expected := "v0=" + hex.EncodeToString(mac.Sum(nil))
	return hmac.Equal([]byte(expected), []byte(sig))
}

// shouldDispatch reports whether an event should trigger a re-ingest, which only human text messages do.
// Bot messages are skipped here but still captured when the next human message re-fetches the conversation.
func (h *EventsHandler) shouldDispatch(ev slackInnerEvent) bool {
	if ev.Type != "message" {
		return false
	}
	if ev.Subtype != "" && ev.Subtype != "thread_broadcast" {
		return false
	}
	if ev.BotID != "" {
		return false
	}
	return strings.TrimSpace(ev.Text) != ""
}

// eventEnvelope is the outer Slack Events API payload.
type eventEnvelope struct {
	Type      string          `json:"type"`
	Challenge string          `json:"challenge"`
	Token     string          `json:"token"`
	TeamID    string          `json:"team_id"`
	EventID   string          `json:"event_id"`
	Event     json.RawMessage `json:"event"`
}

// slackInnerEvent is the inner event of an event_callback.
type slackInnerEvent struct {
	Type     string `json:"type"`
	Subtype  string `json:"subtype"`
	Channel  string `json:"channel"`
	User     string `json:"user"`
	BotID    string `json:"bot_id"`
	Text     string `json:"text"`
	TS       string `json:"ts"`
	ThreadTS string `json:"thread_ts"`
}

// recentIDs remembers the last n IDs added, evicting the oldest first.
type recentIDs struct {
	mu   sync.Mutex
	set  map[string]struct{}
	ring []string
	next int
}

func newRecentIDs(n int) *recentIDs {
	return &recentIDs{set: make(map[string]struct{}, n), ring: make([]string, n)}
}

// add records id and reports whether it was new. An empty id is always new.
func (r *recentIDs) add(id string) bool {
	if id == "" {
		return true
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.set[id]; ok {
		return false
	}
	if old := r.ring[r.next]; old != "" {
		delete(r.set, old)
	}
	r.ring[r.next] = id
	r.next = (r.next + 1) % len(r.ring)
	r.set[id] = struct{}{}
	return true
}
