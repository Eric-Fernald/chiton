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

// governRef is one entry of a decision's governs attr. NodeID is "" while no node matches Label. Rejected marks
// a reference whose link a verification pass removed, which Relink must not restore.
type governRef struct {
	Label    string
	NodeID   string
	Rejected bool
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
			rejected, _ := m["rejected"].(bool)
			refs = append(refs, governRef{Label: label, NodeID: model.AttrString(m, "node_id"), Rejected: rejected})
		}
	}
	return refs, true
}

// governsAttr encodes refs for the governs attr. It is never nil, so an empty list still reads back as recorded.
func governsAttr(refs []governRef) []map[string]any {
	out := make([]map[string]any, len(refs))
	for i, r := range refs {
		out[i] = map[string]any{"label": r.Label, "node_id": r.NodeID}
		if r.Rejected {
			out[i]["rejected"] = true
		}
	}
	return out
}

// governsOf returns d's governs entries and the nodes it has a live DECIDED_IN edge to (ID → label), leaving out
// its meeting. For a decision written before the attr existed, recorded is false and refs lists the live links.
func (s *LinkService) governsOf(ctx context.Context, d model.Node) (refs []governRef, linked map[string]string, recorded bool, err error) {
	sub, err := s.store.Neighbors(ctx, d.ID, 1, []model.EdgeKind{model.EdgeDecidedIn})
	if err != nil {
		return nil, nil, false, fmt.Errorf("load links of decision %s: %w", d.ID, err)
	}
	linked = map[string]string{}
	for _, n := range sub.Nodes {
		if n.ID != d.ID && n.Kind != model.KindMeeting {
			linked[n.ID] = n.Label
		}
	}
	if refs, recorded = governed(d); !recorded {
		for _, id := range slices.Sorted(maps.Keys(linked)) {
			refs = append(refs, governRef{Label: linked[id], NodeID: id})
		}
	}
	return refs, linked, recorded, nil
}

// RejectLink removes the DECIDED_IN edge from a decision to a node it does not govern and marks the reference
// rejected, so Relink leaves it removed. Re-submitting the meeting's linkage starts over.
func (s *LinkService) RejectLink(ctx context.Context, decisionID, symbolID string) error {
	if store.IsReadOnly(s.store) {
		return fmt.Errorf("reject link: store is read-only — stop 'chiton serve' or restart it with --writable")
	}
	d, err := s.store.GetNode(ctx, decisionID)
	if err != nil {
		return fmt.Errorf("load decision: %w", err)
	}
	if d == nil || d.Kind != model.KindDecision {
		return fmt.Errorf("reject link: %q is not a decision", decisionID)
	}
	refs, linked, _, err := s.governsOf(ctx, *d)
	if err != nil {
		return err
	}
	label, ok := linked[symbolID]
	if !ok {
		return fmt.Errorf("reject link: decision %s has no link to %s", decisionID, symbolID)
	}
	i := slices.IndexFunc(refs, func(r governRef) bool { return r.NodeID == symbolID })
	if i < 0 {
		refs = append(refs, governRef{Label: label})
		i = len(refs) - 1
	}
	refs[i].NodeID, refs[i].Rejected = "", true
	if d.Attrs == nil {
		d.Attrs = map[string]any{}
	}
	d.Attrs[attrGoverns] = governsAttr(refs)
	if _, err := store.DeleteEdges(ctx, s.store, []string{model.NewEdgeID(decisionID, symbolID, model.EdgeDecidedIn)}); err != nil {
		return fmt.Errorf("delete rejected link: %w", err)
	}
	if err := s.store.UpsertNodes(ctx, []model.Node{*d}); err != nil {
		return fmt.Errorf("update decision: %w", err)
	}
	return nil
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
	Ambiguous  []string `json:"ambiguous,omitempty"`  // labels several nodes share, left unlinked
}

// Relink restores decision→symbol links that deleting a symbol's node removed. A symbol that moved files gets a
// new node ID, so each governs entry without a live DECIDED_IN edge is looked up again by exact label. A label
// with no exact match stays unlinked and is reported, and a rejected entry is left alone. Decisions linked before
// governs existed have it filled in from the edges they still have.
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
	unresolved, ambiguous := map[string]bool{}, map[string]bool{}
	for _, d := range decisions {
		refs, linked, recorded, err := s.governsOf(ctx, d)
		if err != nil {
			return RelinkReport{}, err
		}
		changed := !recorded
		meetingID := model.AttrString(d.Attrs, "meeting_id")
		for i, ref := range refs {
			if _, ok := linked[ref.NodeID]; ok || ref.Rejected {
				continue
			}
			id, shared := s.uniqueMatch(ctx, ref.Label)
			if id == "" || meetingID == "" {
				if shared {
					ambiguous[ref.Label] = true
				} else {
					unresolved[ref.Label] = true
				}
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
	rep.Unresolved, rep.Ambiguous = sortedKeys(unresolved), sortedKeys(ambiguous)
	return rep, nil
}

// uniqueMatch returns the one code-side node whose label is exactly label (case-insensitive), preferring a
// definition over other kinds and ignoring the reference stubs that share a function's label. When several
// qualify it returns "" with shared set: names like Open and New exist in many packages, and a decision is not
// attached to a guess. It reads every match, so the true one can't fall outside a result cap.
func (s *LinkService) uniqueMatch(ctx context.Context, label string) (id string, shared bool) {
	nodes, err := s.store.SearchNodes(ctx, label, 0)
	if err != nil {
		return "", false
	}
	var symbols, others []string
	for _, n := range nodes {
		switch {
		case !codeSide(n) || model.IsExternal(n) || !strings.EqualFold(n.Label, label):
		case n.Kind == model.KindSymbol:
			symbols = append(symbols, n.ID)
		default:
			others = append(others, n.ID)
		}
	}
	candidates := symbols
	if len(candidates) == 0 {
		candidates = others
	}
	if len(candidates) == 1 {
		return candidates[0], false
	}
	return "", len(candidates) > 1
}

// codeSide reports whether a decision may be linked to n. Meeting-domain nodes are never link targets.
func codeSide(n model.Node) bool {
	switch n.Kind {
	case model.KindMeeting, model.KindMeetingTurn, model.KindPerson, model.KindDecision, model.KindRationale:
		return false
	}
	return true
}

// matchSymbol looks label up among code-side nodes. exact is a case-insensitive whole-label match, preferring a
// symbol over another kind. contains is the first node whose label merely contains label. Like uniqueMatch it
// reads every match, so a result cap can't hide the symbol behind same-named nodes.
func (s *LinkService) matchSymbol(ctx context.Context, label string) (exact, contains string, err error) {
	label = strings.TrimSpace(label)
	if label == "" {
		return "", "", nil
	}
	nodes, err := s.store.SearchNodes(ctx, label, 0)
	if err != nil {
		return "", "", fmt.Errorf("look up %q: %w", label, err)
	}
	lower := strings.ToLower(label)
	for _, n := range nodes {
		if !codeSide(n) {
			continue
		}
		if strings.EqualFold(n.Label, label) {
			if n.Kind == model.KindSymbol {
				return n.ID, contains, nil // best: exact symbol match
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
	return exact, contains, nil
}
