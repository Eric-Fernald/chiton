// Package teams ingests Microsoft Teams meeting transcripts: fetched as WebVTT, they go through the same
// meeting parser as a local .vtt, so meeting→code linkage applies unchanged.
package teams

import (
	"context"
	"fmt"
	"time"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// TranscriptMeta describes one transcript available for an online meeting.
type TranscriptMeta struct {
	ID        string    // transcript id
	MeetingID string    // online meeting id it belongs to
	CreatedAt time.Time // when the transcript was created (zero if unknown)
}

// TranscriptSource lists and fetches Teams transcripts. The concrete Microsoft
// Graph implementation is GraphSource (graph_client.go); tests use a fake.
type TranscriptSource interface {
	// ListTranscripts returns the transcripts available for an online meeting.
	ListTranscripts(ctx context.Context, meetingID string) ([]TranscriptMeta, error)
	// FetchContent returns a transcript's content as WebVTT bytes.
	FetchContent(ctx context.Context, meetingID, transcriptID string) ([]byte, error)
}

// GraphConfig configures the Microsoft Graph transcript source (app-only auth); shared by both msgraph builds.
type GraphConfig struct {
	TenantID     string
	ClientID     string
	ClientSecret string
	// UserID is the meeting organizer's id or UPN: transcripts are read under
	// /users/{UserID}/onlineMeetings/... (app-only has no /me).
	UserID string
}

// Ingester turns Teams transcripts into graph nodes via a TranscriptSource.
type Ingester struct {
	store store.GraphStore
	src   TranscriptSource
	now   func() time.Time
}

// NewIngester builds an Ingester over a graph store and transcript source.
func NewIngester(s store.GraphStore, src TranscriptSource) *Ingester {
	return &Ingester{store: s, src: src, now: time.Now}
}

// Report summarises an ingest run.
type Report struct {
	MeetingID    string `json:"meeting_id"`
	Transcripts  int    `json:"transcripts"`
	NodesUpdated int    `json:"nodes_updated"`
	Turns        int    `json:"turns"`
}

// MeetingTranscriptURI is the stable source_uri for a single Teams transcript,
// so a re-ingest replaces exactly that transcript's nodes.
func MeetingTranscriptURI(meetingID, transcriptID string) string {
	return fmt.Sprintf("teams://meeting/%s/transcript/%s", meetingID, transcriptID)
}

// IngestMeeting ingests every transcript of a meeting as its own meeting subgraph, keyed on a teams:// URI so
// re-ingest replaces rather than duplicates.
func (i *Ingester) IngestMeeting(ctx context.Context, meetingID string) (Report, error) {
	if store.IsReadOnly(i.store) {
		return Report{}, fmt.Errorf("teams: store is read-only — restart 'chiton serve --writable' or use the 'chiton teams fetch' CLI")
	}
	metas, err := i.src.ListTranscripts(ctx, meetingID)
	if err != nil {
		return Report{}, fmt.Errorf("list transcripts: %w", err)
	}
	rep := Report{MeetingID: meetingID}
	for _, m := range metas {
		content, err := i.src.FetchContent(ctx, meetingID, m.ID)
		if err != nil {
			return rep, fmt.Errorf("fetch transcript %s: %w", m.ID, err)
		}
		n, turns, err := i.ingestOne(ctx, m, content)
		if err != nil {
			return rep, err
		}
		rep.Transcripts++
		rep.NodesUpdated += n
		rep.Turns += turns
	}
	return rep, nil
}

// ingestOne parses one transcript's VTT and upserts its meeting subgraph.
func (i *Ingester) ingestOne(ctx context.Context, m TranscriptMeta, content []byte) (nodes, turns int, err error) {
	uri := MeetingTranscriptURI(m.MeetingID, m.ID)
	tr, err := meeting.Parse(uri, content)
	if err != nil {
		return 0, 0, fmt.Errorf("parse transcript %s: %w", m.ID, err)
	}
	tr.SourceURI = uri
	tr.Title = transcriptTitle(m)
	tr.Date = m.CreatedAt

	if _, err := i.store.DeleteBySourceURI(ctx, uri); err != nil {
		return 0, 0, fmt.Errorf("replace prior transcript %s: %w", uri, err)
	}
	n, err := meeting.Upsert(ctx, i.store, tr, i.now(), "meeting")
	if err != nil {
		return 0, 0, err
	}
	return n, len(tr.Turns), nil
}

// transcriptTitle builds a human label for a Teams transcript.
func transcriptTitle(m TranscriptMeta) string {
	if !m.CreatedAt.IsZero() {
		return fmt.Sprintf("Teams meeting %s", m.CreatedAt.UTC().Format("2006-01-02 15:04"))
	}
	return fmt.Sprintf("Teams meeting %s", shortID(m.MeetingID))
}

func shortID(s string) string {
	return s[:min(len(s), 12)]
}
