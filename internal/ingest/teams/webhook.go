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

// WebhookHandler receives Graph change notifications for Teams transcripts. It echoes ?validationToken=…
// handshakes and dispatches changes whose clientState matches to onChange.
type WebhookHandler struct {
	clientState string
	onChange    func(ctx context.Context, c ChangeItem) error
	log         func(format string, args ...any)
}

// NewWebhookHandler builds a handler. clientState is the shared secret from the Graph subscription, and the
// handler rejects notifications that don't match it as forged.
func NewWebhookHandler(clientState string, onChange func(ctx context.Context, c ChangeItem) error, log func(string, ...any)) *WebhookHandler {
	if log == nil {
		log = func(string, ...any) {}
	}
	if clientState == "" {
		// With no clientState the handler fails closed and rejects every notification. Callers should refuse
		// to start in that case, and this check is the library's fallback.
		log("teams webhook: WARNING — clientState is empty; every notification will be rejected")
	}
	return &WebhookHandler{clientState: clientState, onChange: onChange, log: log}
}

func (h *WebhookHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Graph expects ?validationToken=… echoed verbatim as text/plain within 10s, on creation and on renewal.
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

	// Drop mismatched clientStates without an error response, so a forged callback can't learn valid states.
	// The 202 is sent only after every onChange returns, so onChange must fit Graph's 3s ack deadline.
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

// validState compares clientState in constant time. With none configured it rejects, since accepting anything
// would let any network host inject meeting content into the graph.
func (h *WebhookHandler) validState(got string) bool {
	if h.clientState == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(got), []byte(h.clientState)) == 1
}

// ParseTranscriptResource extracts (meetingID, transcriptID) from a resource such as
// "users('U')/onlineMeetings('MEET')/transcripts('TID')". It sets ok to false for non-transcript paths.
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
