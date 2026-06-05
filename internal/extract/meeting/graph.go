package meeting

import (
	"fmt"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
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

// PersonID returns the stable node ID for a speaker, so other packages
// (the linkage submit path) can resolve an action-item assignee to the same
// node ToGraph created.
func PersonID(name string) string {
	return model.NewNodeID(PersonURI(name), name)
}

// ToGraph converts a parsed Transcript into graph nodes and edges.
//
// Everything emitted here is EXTRACTED with confidence 1.0: turns, speakers,
// timings, and participants are read directly from the transcript, never
// inferred. The meeting node and its meeting_turn children share the meeting's
// source_uri, so a re-ingest can DeleteBySourceURI(meetingURI) then re-add the
// whole meeting atomically (the same pattern the AST extractors use for a file
// and its symbols). person nodes use a separate, shared person:// identity and
// survive a re-ingest.
//
// The decision/rationale layer (DECIDED_IN, DISCUSSED_IN, …) is INFERRED and is
// added later by LinkService — never here.
//
// now is injected for deterministic timestamps in tests.
func ToGraph(t *Transcript, now time.Time) ([]model.Node, []model.Edge) {
	uri := t.SourceURI
	if uri == "" {
		uri = MeetingURI(t.Title)
	}
	meetingID := model.NewNodeID(uri, t.Title)

	var nodes []model.Node
	var edges []model.Edge

	personID := map[string]string{}
	var participants []string
	var fullText strings.Builder

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
		turnID := model.NewNodeID(uri, fmt.Sprintf("turn-%d", i))
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

		if turn.Speaker != "" {
			fullText.WriteString(turn.Speaker)
			fullText.WriteString(": ")
		}
		fullText.WriteString(turn.Text)
		fullText.WriteByte('\n')
	}

	meetingAttrs := map[string]any{
		"text":         strings.TrimRight(fullText.String(), "\n"),
		"participants": participants,
		"turn_count":   len(t.Turns),
	}
	if !t.Date.IsZero() {
		meetingAttrs["date"] = t.Date.UTC().Format(time.RFC3339)
	}
	if d := transcriptDuration(t); d > 0 {
		meetingAttrs["duration_ms"] = d.Milliseconds()
	}

	meeting := model.Node{
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
	// Prepend the meeting node so callers can rely on nodes[0] being it.
	nodes = append([]model.Node{meeting}, nodes...)
	return nodes, edges
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

// slug lowercases and collapses runs of non-alphanumerics to single hyphens,
// for readable meeting:// / person:// URIs. Empty input yields "meeting".
func slug(s string) string {
	var b strings.Builder
	lastHyphen := false
	for _, r := range strings.ToLower(s) {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'):
			b.WriteRune(r)
			lastHyphen = false
		default:
			if !lastHyphen {
				b.WriteByte('-')
				lastHyphen = true
			}
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "meeting"
	}
	return out
}
