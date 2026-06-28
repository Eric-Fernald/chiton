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
	"time"
)

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

// EventsHandler receives Slack Events API callbacks. It verifies each request's
// signature (X-Slack-Signature HMAC over the raw body, bounded by a replay
// window), performs the url_verification challenge handshake, and dispatches
// genuine message events to onMessage. It is transport-only: callers wire the
// returned handler into their own http.Server. It mirrors the Teams
// WebhookHandler, with Slack's signing-secret scheme in place of clientState.
type EventsHandler struct {
	signingSecret string
	onMessage     func(ctx context.Context, ev MessageEvent) error
	log           func(format string, args ...any)
	now           func() time.Time
}

// NewEventsHandler builds a handler. signingSecret is the Slack app signing
// secret used to authenticate every request; requests that fail verification
// are rejected. onMessage handles one verified message event (typically:
// re-ingest the channel or thread the message belongs to).
func NewEventsHandler(signingSecret string, onMessage func(ctx context.Context, ev MessageEvent) error, log func(string, ...any)) *EventsHandler {
	if log == nil {
		log = func(string, ...any) {}
	}
	if signingSecret == "" {
		// A Slack app always has a signing secret, so an empty one is a
		// misconfiguration with no legitimate use. Fail closed (verify rejects
		// everything) rather than silently authenticating forged requests.
		log("slack events: WARNING — no signing secret configured; ALL requests will be rejected (set CHITON_SLACK_SIGNING_SECRET)")
	}
	return &EventsHandler{
		signingSecret: signingSecret,
		onMessage:     onMessage,
		log:           log,
		now:           time.Now,
	}
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
		var ev slackInnerEvent
		if err := json.Unmarshal(env.Event, &ev); err != nil {
			h.log("slack events: invalid inner event: %v", err)
			w.WriteHeader(http.StatusOK) // ack so Slack does not retry junk
			return
		}
		if h.shouldDispatch(ev) && h.onMessage != nil {
			me := MessageEvent{
				ChannelID: ev.Channel,
				User:      ev.User,
				Text:      ev.Text,
				TS:        ev.TS,
				ThreadTS:  ev.ThreadTS,
			}
			if err := h.onMessage(r.Context(), me); err != nil {
				h.log("slack events: onMessage failed for channel %s: %v", ev.Channel, err)
			}
		}
		// Slack requires a fast 2xx (within 3s) or it retries; ack regardless of
		// the onMessage outcome, exactly as the Teams webhook does.
		w.WriteHeader(http.StatusOK)
	default:
		w.WriteHeader(http.StatusOK)
	}
}

// verify authenticates a request using the Slack signing-secret scheme:
// X-Slack-Signature = "v0=" + HMAC_SHA256(secret, "v0:"+timestamp+":"+body),
// constant-time compared, with the timestamp bounded to the replay window.
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

// shouldDispatch reports whether an inner event should trigger a re-ingest:
// only genuine human text messages (skips edits/deletes/joins and bot echoes).
// It is intentionally stricter than the WebClient's ingestable filter — a
// re-ingest re-fetches the whole conversation, which applies ingestable to
// every message, so bot/system messages are still captured on the next human
// message; we just don't re-fetch on bot chatter.
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
