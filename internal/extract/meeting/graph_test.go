package meeting

import (
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func sampleTranscript() *Transcript {
	return &Transcript{
		Title: "Retry Policy Sync",
		Turns: []Turn{
			{Speaker: "Alice", Start: 0, End: 3 * time.Second, Text: "We should cap retries at 3."},
			{Speaker: "Bob", Start: 3 * time.Second, End: 6 * time.Second, Text: "Agreed, with backoff."},
			{Speaker: "Alice", Start: 6 * time.Second, End: 9 * time.Second, Text: "I'll update fetchAll."},
		},
	}
}

func TestToGraphNodeShape(t *testing.T) {
	now := time.Date(2026, 6, 5, 12, 0, 0, 0, time.UTC)
	nodes, edges := ToGraph(sampleTranscript(), now)

	// nodes[0] must be the meeting.
	m := nodes[0]
	if m.Kind != model.KindMeeting {
		t.Fatalf("nodes[0] kind=%q want meeting", m.Kind)
	}
	if m.SourceURI != "meeting://retry-policy-sync" {
		t.Errorf("meeting uri=%q", m.SourceURI)
	}
	if m.Provenance != model.ProvExtracted || m.Confidence != 1.0 {
		t.Errorf("meeting provenance/conf = %q/%v want EXTRACTED/1", m.Provenance, m.Confidence)
	}
	if tc, _ := m.Attrs["turn_count"].(int); tc != 3 {
		t.Errorf("turn_count=%v want 3", m.Attrs["turn_count"])
	}
	parts, _ := m.Attrs["participants"].([]string)
	if len(parts) != 2 || parts[0] != "Alice" || parts[1] != "Bob" {
		t.Errorf("participants=%v want [Alice Bob]", parts)
	}
	if dur, _ := m.Attrs["duration_ms"].(int64); dur != 9000 {
		t.Errorf("duration_ms=%v want 9000", m.Attrs["duration_ms"])
	}

	// Count node kinds.
	kinds := map[model.NodeKind]int{}
	for _, n := range nodes {
		kinds[n.Kind]++
	}
	if kinds[model.KindMeetingTurn] != 3 {
		t.Errorf("meeting_turn count=%d want 3", kinds[model.KindMeetingTurn])
	}
	if kinds[model.KindPerson] != 2 {
		t.Errorf("person count=%d want 2 (deduped)", kinds[model.KindPerson])
	}

	// person nodes are PII-tagged.
	for _, n := range nodes {
		if n.Kind == model.KindPerson {
			if pii, _ := n.Attrs["pii"].(bool); !pii {
				t.Errorf("person %q missing pii=true", n.Label)
			}
		}
	}

	// Edges: 3 CONTAINS (meeting→turn) + 2 MENTIONED (meeting→person).
	ek := map[model.EdgeKind]int{}
	for _, e := range edges {
		ek[e.Kind]++
	}
	if ek[model.EdgeContains] != 3 {
		t.Errorf("CONTAINS=%d want 3", ek[model.EdgeContains])
	}
	if ek[model.EdgeMentioned] != 2 {
		t.Errorf("MENTIONED=%d want 2", ek[model.EdgeMentioned])
	}
}

func TestToGraphStableIDs(t *testing.T) {
	now := time.Now()
	a, _ := ToGraph(sampleTranscript(), now)
	b, _ := ToGraph(sampleTranscript(), now)
	if len(a) != len(b) {
		t.Fatalf("node count differs: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].ID != b[i].ID {
			t.Errorf("node %d ID unstable: %q vs %q", i, a[i].ID, b[i].ID)
		}
	}
	// PersonID helper resolves to the same node the conversion created.
	wantAlice := PersonID("Alice")
	found := false
	for _, n := range a {
		if n.Kind == model.KindPerson && n.Label == "Alice" {
			found = n.ID == wantAlice
		}
	}
	if !found {
		t.Errorf("PersonID(Alice)=%q does not match the emitted Alice node", wantAlice)
	}
}

func TestToGraphSharedSourceURI(t *testing.T) {
	// Meeting and its turns share the meeting URI so DeleteBySourceURI replaces
	// the whole meeting; persons have their own person:// URI.
	now := time.Now()
	nodes, _ := ToGraph(sampleTranscript(), now)
	muri := nodes[0].SourceURI
	for _, n := range nodes {
		switch n.Kind {
		case model.KindMeeting, model.KindMeetingTurn:
			if n.SourceURI != muri {
				t.Errorf("%s node uri=%q want %q", n.Kind, n.SourceURI, muri)
			}
		case model.KindPerson:
			if n.SourceURI == muri {
				t.Errorf("person node shares meeting uri %q (should be person://)", muri)
			}
		}
	}
}

func TestToGraphRespectsSourceURI(t *testing.T) {
	now := time.Now()
	tr := sampleTranscript()
	tr.SourceURI = "teams://meeting/abc123"
	nodes, _ := ToGraph(tr, now)
	if nodes[0].SourceURI != "teams://meeting/abc123" {
		t.Errorf("meeting uri=%q want the provided teams:// uri", nodes[0].SourceURI)
	}
}
