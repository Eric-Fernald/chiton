package meeting

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// URI schemes for meeting-domain nodes.
const (
	// MeetingURIScheme prefixes a meeting's (and its turns' and decisions')
	// source_uri, so DeleteBySourceURI can replace a whole meeting atomically.
	MeetingURIScheme = "meeting://"
	// PersonURIScheme prefixes a speaker/attendee identity, shared across
	// meetings so the same person is one node graph-wide.
	PersonURIScheme = "person://"
)

// MeetingURI returns a stable meeting:// URI from a human id (title or external
// id). The slug keeps it readable and collision-resistant for distinct titles.
func MeetingURI(id string) string {
	return MeetingURIScheme + slug(id)
}

// PersonURI returns the shared person:// identity URI for a speaker name.
func PersonURI(name string) string {
	return PersonURIScheme + slug(name)
}

// slug is model.Slug with this package's fallback, for readable meeting:// / person:// URIs and decision IDs.
func slug(s string) string { return model.Slug(s, "meeting") }

// PersonID returns the stable node ID for a speaker, so the linkage submit path resolves an action-item
// assignee to the same node ToGraph created.
func PersonID(name string) string {
	return model.NewNodeID(PersonURI(name), name)
}

// ToGraph converts a Transcript into EXTRACTED meeting, meeting_turn, and person nodes (nodes[0] is the
// meeting). Turns share the meeting's source_uri so a re-ingest replaces them; person nodes survive it.
func ToGraph(t *Transcript, now time.Time) ([]model.Node, []model.Edge) {
	uri := t.SourceURI
	if uri == "" {
		uri = MeetingURI(t.Title)
	}
	meetingID := model.NewNodeID(uri, t.Title)

	// Slot 0 is reserved for the meeting node, filled in once its attrs are known.
	nodes := make([]model.Node, 1, 1+len(t.Turns))
	edges := make([]model.Edge, 0, len(t.Turns))

	personID := map[string]string{}
	var participants []string

	for i, turn := range t.Turns {
		speakerID := ""
		if turn.Speaker != "" {
			pid, ok := personID[turn.Speaker]
			if !ok {
				pURI := PersonURI(turn.Speaker)
				pid = model.NewNodeID(pURI, turn.Speaker)
				personID[turn.Speaker] = pid
				participants = append(participants, turn.Speaker)
				nodes = append(nodes, model.Node{
					ID:        pid,
					Kind:      model.KindPerson,
					Label:     turn.Speaker,
					SourceURI: pURI,
					// Speaker names are PII (security plan): excluded from default
					// query results unless the caller opts in with include_pii.
					Attrs:      map[string]any{"pii": true},
					Provenance: model.ProvExtracted,
					Confidence: 1.0,
					CreatedAt:  now,
					UpdatedAt:  now,
				})
				edges = append(edges, mentionedEdge(meetingID, pid, now))
			}
			speakerID = pid
		}

		// One meeting_turn node per turn. ID is keyed on the turn index so a
		// re-ingest of identical content reproduces the same IDs.
		turnID := model.NewNodeID(uri, "turn-"+strconv.Itoa(i))
		attrs := map[string]any{
			"index":    i,
			"text":     turn.Text,
			"start_ms": turn.Start.Milliseconds(),
			"end_ms":   turn.End.Milliseconds(),
		}
		if turn.Speaker != "" {
			attrs["speaker"] = turn.Speaker
			attrs["person_id"] = speakerID
		}
		nodes = append(nodes, model.Node{
			ID:         turnID,
			Kind:       model.KindMeetingTurn,
			Label:      turnLabel(i, turn),
			SourceURI:  uri,
			Attrs:      attrs,
			Provenance: model.ProvExtracted,
			Confidence: 1.0,
			CreatedAt:  now,
			UpdatedAt:  now,
		})
		edges = append(edges, containsEdge(meetingID, turnID, now))
	}

	meetingAttrs := map[string]any{
		"text":         strings.TrimRight(t.Text(), "\n"),
		"participants": participants,
		"turn_count":   len(t.Turns),
	}
	if !t.Date.IsZero() {
		meetingAttrs["date"] = t.Date.UTC().Format(time.RFC3339)
	}
	if d := transcriptDuration(t); d > 0 {
		meetingAttrs["duration_ms"] = d.Milliseconds()
	}

	nodes[0] = model.Node{
		ID:         meetingID,
		Kind:       model.KindMeeting,
		Label:      t.Title,
		SourceURI:  uri,
		Attrs:      meetingAttrs,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		CreatedAt:  now,
		UpdatedAt:  now,
	}
	return nodes, edges
}

// Upsert writes tr's ToGraph subgraph to gs and returns the node count; noun names it in errors ("upsert <noun>
// nodes: ..."). Callers first delete the prior version under tr.SourceURI.
func Upsert(ctx context.Context, gs store.GraphStore, tr *Transcript, now time.Time, noun string) (int, error) {
	nodes, edges := ToGraph(tr, now)
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		return 0, fmt.Errorf("upsert %s nodes: %w", noun, err)
	}
	if err := gs.UpsertEdges(ctx, edges); err != nil {
		return 0, fmt.Errorf("upsert %s edges: %w", noun, err)
	}
	return len(nodes), nil
}

// transcriptDuration returns the span from the first turn's start to the last
// turn's end (or last start when ends are absent).
func transcriptDuration(t *Transcript) time.Duration {
	if len(t.Turns) == 0 {
		return 0
	}
	start := t.Turns[0].Start
	var end time.Duration
	for _, turn := range t.Turns {
		if turn.End > end {
			end = turn.End
		}
		if turn.Start > end {
			end = turn.Start
		}
	}
	if end <= start {
		return 0
	}
	return end - start
}

// turnLabel builds a concise, human-readable label for a turn: the speaker and
// timestamp when known, falling back to the turn index.
func turnLabel(i int, t Turn) string {
	who := t.Speaker
	if who == "" {
		who = fmt.Sprintf("turn %d", i)
	}
	if t.End > 0 || t.Start > 0 {
		return fmt.Sprintf("%s @%s", who, fmtClock(t.Start))
	}
	return who
}

// fmtClock renders a Duration as H:MM:SS (or M:SS under an hour).
func fmtClock(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	total := int(d / time.Second)
	h := total / 3600
	m := (total % 3600) / 60
	s := total % 60
	if h > 0 {
		return fmt.Sprintf("%d:%02d:%02d", h, m, s)
	}
	return fmt.Sprintf("%d:%02d", m, s)
}

// --- edge builders (EXTRACTED structural facts) ---

func containsEdge(meetingID, childID string, now time.Time) model.Edge {
	return model.Edge{
		ID:         model.NewEdgeID(meetingID, childID, model.EdgeContains),
		Src:        meetingID,
		Dst:        childID,
		Kind:       model.EdgeContains,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		CreatedAt:  now,
	}
}

// mentionedEdge records a participant: a person mentioned (spoke) in a meeting.
func mentionedEdge(meetingID, personID string, now time.Time) model.Edge {
	return model.Edge{
		ID:         model.NewEdgeID(meetingID, personID, model.EdgeMentioned),
		Src:        meetingID,
		Dst:        personID,
		Kind:       model.EdgeMentioned,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		CreatedAt:  now,
	}
}
