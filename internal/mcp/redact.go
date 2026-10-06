package mcpsrv

import (
	"maps"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

const (
	redactedName   = "[redacted]"
	redactedPerson = "[redacted person]"
)

// personAttrKeys are attrs that name or address people: meeting participants and speakers, email
// correspondents, and a person's own address.
var personAttrKeys = []string{"participants", "speaker", "person_id", "from", "from_name", "to", "cc", "email"}

// redactor projects nodes for tool output. Unless includePII (serve --include-pii), it hides who people are
// while keeping what was said and decided.
type redactor struct{ includePII bool }

// isPerson reports whether n identifies a person: a person node, or any node tagged pii.
func isPerson(n model.Node) bool {
	pii, _ := n.Attrs["pii"].(bool)
	return n.Kind == model.KindPerson || pii
}

// label is n's label with any person's name removed.
func (r redactor) label(n model.Node) string {
	if r.includePII {
		return n.Label
	}
	if isPerson(n) {
		return redactedPerson
	}
	if sp := model.AttrString(n.Attrs, "speaker"); sp != "" && n.Kind == model.KindMeetingTurn {
		if rest, ok := strings.CutPrefix(n.Label, sp); ok {
			return redactedName + rest
		}
	}
	return n.Label
}

// hidden reports whether a label lookup must skip n, because matching its label would itself reveal a name.
func (r redactor) hidden(n model.Node) bool { return r.label(n) != n.Label }

// node projects n to its lean shape, redacted.
func (r redactor) node(n model.Node) leanNode {
	ln := leanNode{
		Kind: n.Kind, Label: r.label(n), SourceURI: n.SourceURI, Span: n.Span,
		Attrs: n.Attrs, Provenance: n.Provenance, Confidence: n.Confidence,
	}
	switch {
	case r.includePII:
	case isPerson(n):
		ln.SourceURI = meeting.PersonURIScheme + redactedName
		ln.Attrs = map[string]any{"pii": true}
	default:
		ln.Attrs = redactAttrs(n.Attrs)
	}
	return ln
}

// nodes projects nodes in order. The result is never nil, so JSON renders [], not null.
func (r redactor) nodes(nodes []model.Node) []leanNode {
	out := make([]leanNode, len(nodes))
	for i, n := range nodes {
		out[i] = r.node(n)
	}
	return out
}

// labels maps node ID to redacted label, for naming edge endpoints.
func (r redactor) labels(nodes []model.Node) map[string]string {
	m := make(map[string]string, len(nodes))
	for _, n := range nodes {
		m[n.ID] = r.label(n)
	}
	return m
}

// overlayLabel is a cluster result's label, hidden for a person (recognised by its person:// source URI).
func (r redactor) overlayLabel(nr cluster.NodeResult) string {
	if !r.includePII && strings.HasPrefix(nr.SourceURI, meeting.PersonURIScheme) {
		return redactedPerson
	}
	return nr.Label
}

// redactAttrs returns attrs without person identifiers, cloning only when there is something to drop.
func redactAttrs(attrs map[string]any) map[string]any {
	_, hasActions := attrs["action_items"]
	if !hasActions && !hasAnyKey(attrs, personAttrKeys) {
		return attrs
	}
	out := maps.Clone(attrs)
	for _, k := range personAttrKeys {
		delete(out, k)
	}
	if ps := model.AttrStrings(attrs, "participants"); len(ps) > 0 {
		out["participant_count"] = len(ps)
		if text, ok := out["text"].(string); ok {
			out["text"] = redactSpeakers(text, ps)
		}
	}
	if items, ok := out["action_items"].([]any); ok {
		out["action_items"] = actionTexts(items)
	}
	return out
}

func hasAnyKey(m map[string]any, keys []string) bool {
	for _, k := range keys {
		if _, ok := m[k]; ok {
			return true
		}
	}
	return false
}

// redactSpeakers replaces the "<participant>: " line prefixes that meeting.Transcript.Text writes.
func redactSpeakers(text string, participants []string) string {
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		for _, p := range participants {
			if rest, ok := strings.CutPrefix(line, p+": "); ok {
				lines[i] = redactedName + ": " + rest
				break
			}
		}
	}
	return strings.Join(lines, "\n")
}

// actionTexts copies decision action items keeping only their text, since assignees are people.
func actionTexts(items []any) []any {
	out := make([]any, 0, len(items))
	for _, it := range items {
		if m, ok := it.(map[string]any); ok {
			out = append(out, map[string]any{"text": m["text"]})
		}
	}
	return out
}
