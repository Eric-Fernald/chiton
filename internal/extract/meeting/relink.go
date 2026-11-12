package meeting

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// attrGoverns is the decision attr listing every reference the decision named, so a later pass can tell what it
// governed after the symbol's node (and with it the edge) is gone.
const attrGoverns = "governs"

// governRef is one entry of a decision's governs attr. NodeID is "" while no node matches Label.
type governRef struct {
	Label  string
	NodeID string
}

// governed reads n's governs attr as the store returns it. recorded is false for a decision written before the
// attr existed.
func governed(n model.Node) (refs []governRef, recorded bool) {
	raw, ok := n.Attrs[attrGoverns]
	if !ok {
		return nil, false
	}
	list, _ := raw.([]any)
	for _, item := range list {
		m, _ := item.(map[string]any)
		if label := model.AttrString(m, "label"); label != "" {
			refs = append(refs, governRef{Label: label, NodeID: model.AttrString(m, "node_id")})
		}
	}
	return refs, true
}

// governsAttr encodes refs for the governs attr. It is never nil, so an empty list still reads back as recorded.
func governsAttr(refs []governRef) []map[string]any {
	out := make([]map[string]any, len(refs))
	for i, r := range refs {
		out[i] = map[string]any{"label": r.Label, "node_id": r.NodeID}
	}
	return out
}

// governEdges links a decision to a symbol it governs: decision DECIDED_IN symbol, symbol DISCUSSED_IN meeting,
// and meeting MENTIONED symbol, so "what did this meeting mention?" also covers decision-governed symbols.
func governEdges(decID, symID, meetingID string, prov model.Provenance, conf float32, now time.Time) []model.Edge {
	return []model.Edge{
		typedEdge(decID, symID, model.EdgeDecidedIn, prov, conf, now, nil),
		typedEdge(symID, meetingID, model.EdgeDiscussedIn, prov, conf, now, nil),
		typedEdge(meetingID, symID, model.EdgeMentioned, prov, conf, now, nil),
	}
}

// RelinkReport summarises what Relink changed.
type RelinkReport struct {
	Relinked   int      `json:"relinked"`
	Unresolved []string `json:"unresolved,omitempty"` // labels no node matches by exact name
}

// Relink restores decision→symbol links that deleting a symbol's node removed. A symbol that moved files gets a
// new node ID, so each governs entry without a live DECIDED_IN edge is looked up again by exact label. A label
// with no exact match stays unlinked and is reported. Decisions linked before governs existed have it filled in
// from the edges they still have.
func (s *LinkService) Relink(ctx context.Context) (RelinkReport, error) {
	if store.IsReadOnly(s.store) {
		return RelinkReport{}, fmt.Errorf("relink: store is read-only — stop 'chiton serve' or restart it with --writable")
	}
	decisions, err := store.NodesMatching(ctx, s.store, store.NodeFilter{Kinds: []model.NodeKind{model.KindDecision}})
	if err != nil {
		return RelinkReport{}, fmt.Errorf("load decisions: %w", err)
	}
	now := s.now()
	var rep RelinkReport
	var nodes []model.Node
	var edges []model.Edge
	unresolved := map[string]bool{}
	for _, d := range decisions {
		sub, err := s.store.Neighbors(ctx, d.ID, 1, []model.EdgeKind{model.EdgeDecidedIn})
		if err != nil {
			return RelinkReport{}, fmt.Errorf("load links of decision %s: %w", d.ID, err)
		}
		// A decision's DECIDED_IN edges reach its meeting and the symbols it governs.
		linked := map[string]string{} // governed node ID → label
		for _, n := range sub.Nodes {
			if n.ID != d.ID && n.Kind != model.KindMeeting {
				linked[n.ID] = n.Label
			}
		}
		refs, recorded := governed(d)
		changed := !recorded
		if !recorded {
			for _, id := range slices.Sorted(maps.Keys(linked)) {
				refs = append(refs, governRef{Label: linked[id], NodeID: id})
			}
		}
		meetingID := model.AttrString(d.Attrs, "meeting_id")
		for i, ref := range refs {
			if _, ok := linked[ref.NodeID]; ok {
				continue
			}
			id, _ := s.matchSymbol(ctx, ref.Label)
			if id == "" || meetingID == "" {
				unresolved[ref.Label] = true
				changed = changed || ref.NodeID != ""
				refs[i].NodeID = ""
				continue
			}
			edges = append(edges, governEdges(d.ID, id, meetingID, d.Provenance, d.Confidence, now)...)
			refs[i].NodeID = id
			changed = true
			rep.Relinked++
		}
		if changed {
			if d.Attrs == nil {
				d.Attrs = map[string]any{}
			}
			d.Attrs[attrGoverns] = governsAttr(refs)
			nodes = append(nodes, d)
		}
	}
	if len(nodes) > 0 {
		if err := s.store.UpsertNodes(ctx, nodes); err != nil {
			return RelinkReport{}, fmt.Errorf("update decisions: %w", err)
		}
	}
	if len(edges) > 0 {
		if err := s.store.UpsertEdges(ctx, edges); err != nil {
			return RelinkReport{}, fmt.Errorf("restore decision links: %w", err)
		}
	}
	rep.Unresolved = sortedKeys(unresolved)
	return rep, nil
}

// matchSymbol looks label up among code-side nodes. exact is a case-insensitive whole-label match, preferring a
// symbol over another kind. contains is the first node whose label merely contains label.
func (s *LinkService) matchSymbol(ctx context.Context, label string) (exact, contains string) {
	label = strings.TrimSpace(label)
	if label == "" {
		return "", ""
	}
	nodes, err := s.store.SearchNodes(ctx, label, 25)
	if err != nil {
		return "", ""
	}
	lower := strings.ToLower(label)
	for _, n := range nodes {
		if n.Kind == model.KindMeeting || n.Kind == model.KindMeetingTurn ||
			n.Kind == model.KindPerson || n.Kind == model.KindDecision ||
			n.Kind == model.KindRationale {
			continue // never link a decision to another meeting-domain node
		}
		if strings.EqualFold(n.Label, label) {
			if n.Kind == model.KindSymbol {
				return n.ID, contains // best: exact symbol match
			}
			if exact == "" {
				exact = n.ID
			}
			continue
		}
		if contains == "" && strings.Contains(strings.ToLower(n.Label), lower) {
			contains = n.ID
		}
	}
	return exact, contains
}
