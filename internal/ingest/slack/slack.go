// Package slack ingests Slack channels and threads as meeting subgraphs, mapping messages straight to a
// meeting.Transcript without a WebVTT parse, so meeting-to-code linkage applies unchanged.
package slack

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// Message is one Slack message, with the author resolved to a display name where the token's scopes allow.
type Message struct {
	User     string // Slack user id (e.g. "U0123"); "" for some bot/system msgs
	Name     string // resolved display name (falls back to the user id)
	Text     string // message text
	TS       string // Slack ts ("1735689600.000600") — the message id in a channel
	ThreadTS string // parent thread ts when this is a threaded reply (optional)
}

// speaker returns the best available human label for the message author.
func (m Message) speaker() string {
	if m.Name != "" {
		return m.Name
	}
	return m.User
}

// Conversation is a fetched Slack channel (or a single thread within it).
type Conversation struct {
	ChannelID   string    // channel id (e.g. "C0123")
	ChannelName string    // resolved channel name (optional)
	ThreadTS    string    // "" for a whole-channel fetch; set for a single thread
	Messages    []Message // chronological order (oldest first)
}

// ConversationSource lists and fetches Slack conversations. The concrete Web
// API implementation is WebClient (web_client.go); tests use a fake.
type ConversationSource interface {
	// FetchConversation returns a channel's messages, or — when threadTS is
	// non-empty — the replies of a single thread. Messages are chronological.
	FetchConversation(ctx context.Context, channelID, threadTS string) (Conversation, error)
	// Name identifies the source kind ("slack").
	Name() string
}

// ConversationURI returns the stable source_uri slack://<channel>[/<thread-ts>]. It omits the workspace so
// the key never shifts between runs, which would break DeleteBySourceURI replacement.
func ConversationURI(channelID, threadTS string) string {
	uri := "slack://" + channelID
	if threadTS != "" {
		uri += "/" + threadTS
	}
	return uri
}

// Ingester turns Slack conversations into graph nodes via a ConversationSource.
type Ingester struct {
	store store.GraphStore
	src   ConversationSource
	now   func() time.Time
}

// NewIngester builds an Ingester over a graph store and conversation source.
func NewIngester(s store.GraphStore, src ConversationSource) *Ingester {
	return &Ingester{store: s, src: src, now: time.Now}
}

// Report summarises an ingest run.
type Report struct {
	ChannelID     string `json:"channel_id"`
	ThreadTS      string `json:"thread_ts,omitempty"`
	Messages      int    `json:"messages"`
	Turns         int    `json:"turns"`
	NodesUpserted int    `json:"nodes_upserted"`
}

// IngestConversation fetches a channel (threadTS == "") or one thread and upserts it as a meeting subgraph
// keyed on its slack:// URI, so re-ingest replaces rather than duplicates.
func (i *Ingester) IngestConversation(ctx context.Context, channelID, threadTS string) (Report, error) {
	if store.IsReadOnly(i.store) {
		return Report{}, fmt.Errorf("slack: store is read-only — restart 'chiton serve --writable' or use the 'chiton slack fetch' CLI")
	}
	if strings.TrimSpace(channelID) == "" {
		return Report{}, fmt.Errorf("slack: a channel id is required")
	}
	conv, err := i.src.FetchConversation(ctx, channelID, threadTS)
	if err != nil {
		return Report{}, fmt.Errorf("fetch conversation: %w", err)
	}
	// Keep the caller's request authoritative for the URI/report even if a
	// source does not echo the ids back.
	if conv.ChannelID == "" {
		conv.ChannelID = channelID
	}
	if conv.ThreadTS == "" {
		conv.ThreadTS = threadTS
	}
	return i.ingestOne(ctx, conv)
}

// ingestOne upserts one fetched conversation as a meeting subgraph.
func (i *Ingester) ingestOne(ctx context.Context, conv Conversation) (Report, error) {
	rep := Report{ChannelID: conv.ChannelID, ThreadTS: conv.ThreadTS, Messages: len(conv.Messages)}
	uri := ConversationURI(conv.ChannelID, conv.ThreadTS)

	// Replace any prior version of exactly this conversation before re-adding,
	// the same atomic pattern the Teams ingester and AST extractors use.
	if _, err := i.store.DeleteBySourceURI(ctx, uri); err != nil {
		return rep, fmt.Errorf("replace prior conversation %s: %w", uri, err)
	}

	tr := conversationToTranscript(conv, uri)
	if len(tr.Turns) == 0 {
		// Nothing ingestable (e.g. a channel of only join/leave notices). The
		// delete above already cleared any stale prior version.
		return rep, nil
	}
	n, err := meeting.Upsert(ctx, i.store, tr, i.now(), "conversation")
	if err != nil {
		return rep, err
	}
	rep.Turns = len(tr.Turns)
	rep.NodesUpserted = n
	return rep, nil
}

// conversationToTranscript maps each message to one Turn timed as its offset from the first message.
// It skips empty-text messages such as file shares and join or leave notices.
func conversationToTranscript(conv Conversation, uri string) *meeting.Transcript {
	tr := &meeting.Transcript{
		Title:     conversationTitle(conv),
		SourceURI: uri,
	}
	var base time.Time
	haveBase := false
	for _, m := range conv.Messages {
		if strings.TrimSpace(m.Text) == "" {
			continue
		}
		var off time.Duration
		if t, err := parseSlackTS(m.TS); err == nil {
			if !haveBase {
				base = t
				haveBase = true
				tr.Date = t
			}
			if off = t.Sub(base); off < 0 {
				off = 0
			}
		}
		tr.Turns = append(tr.Turns, meeting.Turn{
			Speaker: m.speaker(),
			Start:   off,
			End:     off,
			Text:    m.Text,
		})
	}
	return tr
}

// conversationTitle labels a conversation. Thread titles include the parent's date so two threads in one
// channel don't get the same label.
func conversationTitle(conv Conversation) string {
	name := conv.ChannelName
	if name == "" {
		name = conv.ChannelID
	}
	if conv.ThreadTS != "" {
		when := conv.ThreadTS
		if t, err := parseSlackTS(conv.ThreadTS); err == nil {
			when = t.Format("2006-01-02 15:04")
		}
		return fmt.Sprintf("Slack thread in #%s (%s)", name, when)
	}
	return fmt.Sprintf("Slack #%s", name)
}

// parseSlackTS parses a Slack ts ("1735689600.000600") as UTC. The fraction is a per-second counter, which
// it treats as microseconds to preserve ordering.
func parseSlackTS(ts string) (time.Time, error) {
	ts = strings.TrimSpace(ts)
	if ts == "" {
		return time.Time{}, fmt.Errorf("slack: empty ts")
	}
	secStr, fracStr, _ := strings.Cut(ts, ".")
	sec, err := strconv.ParseInt(secStr, 10, 64)
	if err != nil {
		return time.Time{}, fmt.Errorf("slack: bad ts %q: %w", ts, err)
	}
	var nsec int64
	if fracStr != "" {
		// Normalise the fraction to 6 microsecond digits, then to nanoseconds.
		if len(fracStr) > 6 {
			fracStr = fracStr[:6]
		}
		fracStr += strings.Repeat("0", 6-len(fracStr))
		if micros, err := strconv.ParseInt(fracStr, 10, 64); err == nil {
			nsec = micros * 1000
		}
	}
	return time.Unix(sec, nsec).UTC(), nil
}
