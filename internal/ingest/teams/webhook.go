package teams

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// ChangeItem is one Microsoft Graph change notification (a transcript created
// on a meeting). Only the fields chiton acts on are decoded.
type ChangeItem struct {
	SubscriptionID string         `json:"subscriptionId"`
	ChangeType     string         `json:"changeType"`
	Resource       string         `json:"resource"`
	ClientState    string         `json:"clientState"`
	ResourceData   map[string]any `json:"resourceData"`
}

// changeNotification is the envelope Graph POSTs to the webhook.
type changeNotification struct {
	Value []ChangeItem `json:"value"`
}

// WebhookHandler receives Microsoft Graph change notifications for Teams
// transcripts. It performs the subscription-validation handshake (echoing
// ?validationToken=…) and, for genuine notifications, verifies clientState and
// dispatches each change to onChange. It is transport-only: callers wire the
// returned handler into their own http.Server.
type WebhookHandler struct {
	clientState string
	onChange    func(ctx context.Context, c ChangeItem) error
	log         func(format string, args ...any)
}

// NewWebhookHandler builds a handler. clientState is the shared secret set when
// creating the Graph subscription; notifications whose clientState does not
// match are rejected (defends against forged callbacks). onChange handles one
// validated change (typically: fetch the new transcript and ingest it).
func NewWebhookHandler(clientState string, onChange func(ctx context.Context, c ChangeItem) error, log func(string, ...any)) *WebhookHandler {
	if log == nil {
		log = func(string, ...any) {}
	}
	if clientState == "" {
		// Surface the misconfiguration rather than silently authenticating no one:
		// with no clientState every notification (including forged ones) is accepted.
		log("teams webhook: WARNING — clientState is empty; notifications are NOT authenticated")
	}
	return &WebhookHandler{clientState: clientState, onChange: onChange, log: log}
}

func (h *WebhookHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Subscription validation: Graph sends ?validationToken=… and expects it
	// echoed back verbatim as text/plain within 10s. This happens on both the
	// initial POST and on renewal.
	if token := r.URL.Query().Get("validationToken"); token != "" {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		if _, err := io.WriteString(w, token); err != nil {
			// The subscription won't be created if Graph doesn't receive the
			// echoed token — make that failure visible rather than silent.
			h.log("teams webhook: failed to write validation token: %v", err)
		}
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 4<<20))
	if err != nil {
		http.Error(w, "read body", http.StatusBadRequest)
		return
	}
	var note changeNotification
	if err := json.Unmarshal(body, &note); err != nil {
		http.Error(w, "invalid notification", http.StatusBadRequest)
		return
	}

	// Acknowledge fast (Graph requires a 2xx within 3s) and process each change.
	// clientState mismatches are dropped, not errored, so a forged callback
	// cannot learn which states are valid.
	for _, c := range note.Value {
		if !h.validState(c.ClientState) {
			h.log("teams webhook: dropping notification with bad clientState (sub %s)", c.SubscriptionID)
			continue
		}
		if h.onChange != nil {
			if err := h.onChange(r.Context(), c); err != nil {
				h.log("teams webhook: onChange failed for %s: %v", c.Resource, err)
			}
		}
	}
	w.WriteHeader(http.StatusAccepted)
}

// validState constant-time-compares the notification's clientState to ours. An
// empty configured clientState disables the check (not recommended).
func (h *WebhookHandler) validState(got string) bool {
	if h.clientState == "" {
		return true
	}
	return subtle.ConstantTimeCompare([]byte(got), []byte(h.clientState)) == 1
}

// ParseTranscriptResource extracts the (meetingID, transcriptID) from a Graph
// transcript resource string such as
// "communications/onlineMeetings('MEET')/transcripts('TID')" or
// "users('U')/onlineMeetings('MEET')/transcripts('TID')". Returns ok=false when
// the resource is not a transcript path.
func ParseTranscriptResource(resource string) (meetingID, transcriptID string, ok bool) {
	meetingID = between(resource, "onlineMeetings('", "')")
	transcriptID = between(resource, "transcripts('", "')")
	if meetingID == "" || transcriptID == "" {
		return "", "", false
	}
	return meetingID, transcriptID, true
}

// between returns the substring of s between the first occurrence of open and
// the next occurrence of closeTok after it, or "" if either is absent.
func between(s, open, closeTok string) string {
	_, after, ok := strings.Cut(s, open)
	if !ok {
		return ""
	}
	mid, _, ok := strings.Cut(after, closeTok)
	if !ok {
		return ""
	}
	return mid
}
