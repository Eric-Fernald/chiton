// Package slack ingests Slack channel and thread conversations into chiton's
// graph. Slack is the chat-shaped sibling of the Phase 4 meeting sources: where
// Teams hands you a linear WebVTT transcript, Slack hands you threaded,
// already-structured JSON. So Slack is the deliberate exception to the WebVTT
// convergence point — it does NOT ride the transcript parser (there is no
// transcript to parse). Instead a thin adapter (conversationToTranscript) maps
// Slack's structured messages straight into the same meeting.Transcript shape
// meeting.ToGraph consumes, so a Slack channel/thread becomes a
// meeting/meeting_turn/person subgraph and the meeting→code linkage
// (DECIDED_IN / DISCUSSED_IN) applies unchanged.
//
// The Web API wiring lives behind the ConversationSource interface in
// web_client.go (a thin net/http client, no vendor SDK — mirroring the Gmail
// email source); the orchestration here and the Events API receiver (events.go)
// are pure-Go and unit-testable with a fake source.
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

// Message is one Slack message, already resolved to a display name where the
// token's scopes allow. It is the provider-neutral unit a ConversationSource
// returns.
type Message struct {
	User     string // Slack user id (e.g. "U0123"); "" for some bot/system msgs
	Name     string // resolved display name (falls back to the user id)
	Text     string // message text
	TS       string // Slack ts ("1735689600.000600") — the message id in a channel
	ThreadTS string // parent thread ts when this is a threaded reply (optional)
}

// speaker returns the best available human label for the message author.
func (m Message) speaker() string {
	switch {
	case m.Name != "":
		return m.Name
	case m.User != "":
		return m.User
	default:
		return ""
	}
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

// ConversationURI is the stable source_uri for a Slack conversation, so a
// re-ingest replaces exactly that channel/thread's nodes. Form:
//
//	slack://<channel>[/<thread-ts>]
//
// The channel id is the stable identifier; it is keyed on alone (the workspace
// is intentionally NOT part of the key) so the URI never shifts between runs —
// a shifting key would defeat the DeleteBySourceURI replace-not-duplicate
// guarantee the ingester relies on.
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

// IngestConversation fetches a channel (threadTS == "") or a single thread and
// upserts it as a meeting subgraph (meeting + turns + persons), keyed on a
// stable slack:// URI so re-ingest replaces rather than duplicates. Read-only
// stores are rejected up front.
func (i *Ingester) IngestConversation(ctx context.Context, channelID, threadTS string) (Report, error) {
	if ro, ok := i.store.(store.ReadOnlyAware); ok && ro.ReadOnly() {
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
	ns, es := meeting.ToGraph(tr, i.now())
	if err := i.store.UpsertNodes(ctx, ns); err != nil {
		return rep, fmt.Errorf("upsert conversation nodes: %w", err)
	}
	if err := i.store.UpsertEdges(ctx, es); err != nil {
		return rep, fmt.Errorf("upsert conversation edges: %w", err)
	}
	rep.Turns = len(tr.Turns)
	rep.NodesUpserted = len(ns)
	return rep, nil
}

// conversationToTranscript adapts a Slack conversation into the meeting
// Transcript shape meeting.ToGraph consumes. Each message becomes one Turn
// (chat has no speaker "turns" to coalesce — every message stands alone), with
// Start/End set to the message's offset from the first message so the graph
// keeps relative timing. Messages with empty text are skipped (file shares,
// reactions, and join/leave notices carry no prose worth linking to code).
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

// conversationTitle builds a human label for a Slack conversation. Thread
// titles carry a date discriminator (derived from the parent ts) so two
// threads in the same channel don't collapse to an identical label — the same
// disambiguation the Teams transcript title uses.
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

// parseSlackTS parses a Slack message timestamp ("1735689600.000600") into a
// UTC time. The fractional component is Slack's per-second message counter; we
// treat it as microseconds, which preserves ordering and yields a sensible
// sub-second time.
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
		for len(fracStr) < 6 {
			fracStr += "0"
		}
		if micros, err := strconv.ParseInt(fracStr, 10, 64); err == nil {
			nsec = micros * 1000
		}
	}
	return time.Unix(sec, nsec).UTC(), nil
}
