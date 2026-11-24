package meeting

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// KindMeetingLinkage is the subagent task kind for meeting→code linkage.
const KindMeetingLinkage = "meeting_linkage"

// linkageStatusDone marks a linked meeting so Pending skips it. It shares the extraction_status attr with
// semantic extraction without collision, since each Pending scans only its own node kinds.
const linkageStatusDone = "done"

// defaultMaxCandidates caps a task spec's candidate-symbol menu to protect the host's context budget. The
// menu is only a hint, because submit resolves any returned label against the whole graph.
const defaultMaxCandidates = 300

// defaultDiscussedConfidence scores a "merely discussed" symbol link, since the host gives no per-item score.
// The link goes through model.ResolveProvenance like every other INFERRED edge.
const defaultDiscussedConfidence float32 = 0.7

// LinkService runs meeting-to-code linkage in subagent mode. Pending emits a task spec per unlinked meeting,
// and Submit writes the host's decisions as INFERRED (or AMBIGUOUS) nodes and edges. chiton calls no LLM.
type LinkService struct {
	store         store.GraphStore
	threshold     float32
	maxCandidates int
	now           func() time.Time
}

// NewLinkService builds a LinkService. threshold<=0 uses
// model.DefaultConfidenceThreshold.
func NewLinkService(s store.GraphStore, threshold float32) *LinkService {
	if threshold <= 0 {
		threshold = model.DefaultConfidenceThreshold
	}
	return &LinkService{store: s, threshold: threshold, maxCandidates: defaultMaxCandidates, now: time.Now}
}

// LinkTaskSpec is one unit of linkage work: the transcript text, a hint menu of code-symbol labels it seems
// to reference, and the Schema and Instructions that define the JSON returned via submit_meeting_links.
type LinkTaskSpec struct {
	ID               string   `json:"id"`   // meeting node ID (also the submit task_id)
	Kind             string   `json:"kind"` // meeting_linkage
	Label            string   `json:"label"`
	SourceURI        string   `json:"source_uri"`
	Input            string   `json:"input"`
	CandidateSymbols []string `json:"candidate_symbols"`
	Schema           string   `json:"schema"`
	Instructions     string   `json:"instructions"`
}

// ActionItem is a follow-up assigned (optionally) to a person.
type ActionItem struct {
	Text     string `json:"text"`
	Assignee string `json:"assignee"`
}

// DecisionResult is one decision the host assistant extracted from a meeting.
type DecisionResult struct {
	Summary     string       `json:"summary"`
	Rationale   string       `json:"rationale"`
	Confidence  float32      `json:"confidence"`
	References  []string     `json:"references"`
	ActionItems []ActionItem `json:"action_items"`
}

// LinkResult is the host assistant's structured answer for one meeting.
type LinkResult struct {
	TaskID    string           `json:"task_id"`
	Decisions []DecisionResult `json:"decisions"`
	Discussed []string         `json:"discussed"`
}

// LinkReport summarises what Submit wrote.
type LinkReport struct {
	TaskID         string   `json:"task_id"`
	DecisionsAdded int      `json:"decisions_added"`
	EdgesAdded     int      `json:"edges_added"`
	EdgesRemoved   int      `json:"edges_removed"` // links an earlier submit wrote that this one omits
	LinkedSymbols  int      `json:"linked_symbols"`
	Ambiguous      int      `json:"ambiguous"`
	UnresolvedRefs []string `json:"unresolved_refs,omitempty"`
}

// Pending returns linkage task specs for meetings not yet linked (meeting nodes
// whose attrs["extraction_status"] != "done"). limit<=0 means no cap.
func (s *LinkService) Pending(ctx context.Context, limit int) ([]LinkTaskSpec, error) {
	meetings, err := store.NodesMatching(ctx, s.store, store.NodeFilter{Kinds: []model.NodeKind{model.KindMeeting}})
	if err != nil {
		return nil, fmt.Errorf("load nodes: %w", err)
	}

	// Find the pending meetings first, so a call with nothing to do skips loading the symbol menu.
	var pending []model.Node
	for _, n := range meetings {
		if model.AttrString(n.Attrs, "extraction_status") == linkageStatusDone {
			continue
		}
		if strings.TrimSpace(model.AttrString(n.Attrs, "text")) == "" {
			continue
		}
		pending = append(pending, n)
		if limit > 0 && len(pending) >= limit {
			break
		}
	}
	if len(pending) == 0 {
		return nil, nil
	}

	// Collect candidate code-symbol labels once for the whole corpus.
	code, err := store.NodesMatching(ctx, s.store, store.NodeFilter{Kinds: []model.NodeKind{model.KindSymbol, model.KindFile}})
	if err != nil {
		return nil, fmt.Errorf("load nodes: %w", err)
	}
	var symbolLabels []string
	seenLabel := map[string]bool{}
	for _, n := range code {
		lbl := strings.TrimSpace(n.Label)
		if lbl == "" || seenLabel[lbl] {
			continue
		}
		seenLabel[lbl] = true
		symbolLabels = append(symbolLabels, lbl)
	}
	sort.Strings(symbolLabels)
	candidates := newSymbolCandidates(symbolLabels)

	specs := make([]LinkTaskSpec, 0, len(pending))
	for _, n := range pending {
		text := model.AttrString(n.Attrs, "text")
		specs = append(specs, LinkTaskSpec{
			ID:               n.ID,
			Kind:             KindMeetingLinkage,
			Label:            n.Label,
			SourceURI:        n.SourceURI,
			Input:            text,
			CandidateSymbols: candidatesFor(text, candidates, s.maxCandidates),
			Schema:           linkSchema,
			Instructions:     linkInstructions,
		})
	}
	return specs, nil
}

// Submit ingests one meeting's linkage results idempotently (stable IDs). It deletes the meeting's prior
// decision and rationale nodes absent from the new result, which supersedes them.
func (s *LinkService) Submit(ctx context.Context, r LinkResult) (LinkReport, error) {
	if r.TaskID == "" {
		return LinkReport{}, fmt.Errorf("submit: task_id is required")
	}
	if store.IsReadOnly(s.store) {
		return LinkReport{}, fmt.Errorf("submit: store is read-only — restart 'chiton serve --writable' or use 'chiton meeting submit'")
	}
	meeting, err := s.store.GetNode(ctx, r.TaskID)
	if err != nil {
		return LinkReport{}, fmt.Errorf("load meeting node: %w", err)
	}
	if meeting == nil {
		return LinkReport{}, fmt.Errorf("submit: no node with id %q (call pending first)", r.TaskID)
	}
	if meeting.Kind != model.KindMeeting {
		return LinkReport{}, fmt.Errorf("submit: node %q is a %s, not a meeting", r.TaskID, meeting.Kind)
	}

	now := s.now()
	meetingURI := meeting.SourceURI
	decisionURI := DecisionURI(meetingURI)
	rep := LinkReport{TaskID: r.TaskID}

	var nodes []model.Node
	var edges []model.Edge
	keep := map[string]bool{} // decision/rationale node IDs in this submission
	linkedSymbols := map[string]bool{}
	unresolved := map[string]bool{}
	seenPersons := map[string]bool{}

	for i, d := range r.Decisions {
		summary := strings.TrimSpace(d.Summary)
		if summary == "" {
			continue
		}
		prov, conf := model.ResolveProvenance(d.Confidence, s.threshold)
		if prov == model.ProvAmbiguous {
			rep.Ambiguous++
		}
		// Key on position as well as summary so identical summaries in one meeting stay distinct nodes.
		// deleteSupersededDecisions prunes any prior decision not in this submission's keep set.
		disambig := fmt.Sprintf("%d:%s", i, slug(summary))
		decID := model.NewNodeID(meetingURI, "decision:"+disambig)
		keep[decID] = true
		decAttrs := map[string]any{
			"summary":    summary,
			"meeting_id": r.TaskID,
		}
		if rat := strings.TrimSpace(d.Rationale); rat != "" {
			decAttrs["rationale"] = rat
		}
		nodes = append(nodes, model.Node{
			ID:         decID,
			Kind:       model.KindDecision,
			Label:      truncateLabel(summary),
			SourceURI:  decisionURI,
			Attrs:      decAttrs,
			Provenance: prov,
			Confidence: conf,
			CreatedAt:  now,
			UpdatedAt:  now,
		})
		rep.DecisionsAdded++

		// decision --DECIDED_IN--> meeting (where it was decided).
		edges = append(edges, typedEdge(decID, r.TaskID, model.EdgeDecidedIn, prov, conf, now, nil))
		rep.EdgesAdded++

		// Optional rationale node: rationale --RATIONALE_FOR--> decision.
		if rat := strings.TrimSpace(d.Rationale); rat != "" {
			ratID := model.NewNodeID(meetingURI, "rationale:"+disambig)
			keep[ratID] = true
			nodes = append(nodes, model.Node{
				ID:         ratID,
				Kind:       model.KindRationale,
				Label:      truncateLabel(rat),
				SourceURI:  decisionURI,
				Attrs:      map[string]any{"text": rat, "decision_id": decID},
				Provenance: prov,
				Confidence: conf,
				CreatedAt:  now,
				UpdatedAt:  now,
			})
			edges = append(edges, typedEdge(ratID, decID, model.EdgeRationaleFor, prov, conf, now, nil))
			rep.EdgesAdded++
		}

		// References → code symbols the decision governs. Every named reference is recorded, linked or not,
		// so Relink can retry it.
		var governs []governRef
		named := map[string]bool{}
		for _, ref := range d.References {
			ref = strings.TrimSpace(ref)
			if ref == "" || named[ref] {
				continue
			}
			named[ref] = true
			symID, err := s.resolveSymbol(ctx, ref)
			if err != nil {
				return LinkReport{}, fmt.Errorf("submit: %w", err)
			}
			governs = append(governs, governRef{Label: ref, NodeID: symID})
			if symID == "" {
				unresolved[ref] = true
				continue
			}
			linkedSymbols[symID] = true
			edges = append(edges, governEdges(decID, symID, r.TaskID, prov, conf, now)...)
			rep.EdgesAdded += 3
		}
		decAttrs[attrGoverns] = governsAttr(governs)

		// Action items → ACTION_FOR a person (created if new).
		for _, ai := range d.ActionItems {
			text := strings.TrimSpace(ai.Text)
			if text == "" {
				continue
			}
			assignee := strings.TrimSpace(ai.Assignee)
			if assignee == "" {
				continue // unassigned action: kept only in decision attrs below
			}
			pid := PersonID(assignee)
			// Emit each assignee's person node once per submission, though they may own several items.
			if !seenPersons[pid] {
				seenPersons[pid] = true
				nodes = append(nodes, personNode(assignee, pid, now))
			}
			edges = append(edges, typedEdge(decID, pid, model.EdgeActionFor, prov, conf, now,
				map[string]any{"text": text}))
			rep.EdgesAdded++
		}
		if items := nonEmptyActions(d.ActionItems); len(items) > 0 {
			decAttrs["action_items"] = items
		}
	}

	// Symbols merely discussed (no decision): symbol --DISCUSSED_IN--> meeting
	// and meeting --MENTIONED--> symbol.
	for _, ref := range r.Discussed {
		symID, err := s.resolveSymbol(ctx, ref)
		if err != nil {
			return LinkReport{}, fmt.Errorf("submit: %w", err)
		}
		if symID == "" {
			if rr := strings.TrimSpace(ref); rr != "" {
				unresolved[rr] = true
			}
			continue
		}
		linkedSymbols[symID] = true
		prov, conf := model.ResolveProvenance(defaultDiscussedConfidence, s.threshold)
		if prov == model.ProvAmbiguous {
			rep.Ambiguous++
		}
		edges = append(edges, typedEdge(symID, r.TaskID, model.EdgeDiscussedIn, prov, conf, now, nil))
		edges = append(edges, typedEdge(r.TaskID, symID, model.EdgeMentioned, prov, conf, now, nil))
		rep.EdgesAdded += 2
	}

	// Mark the meeting linked (preserve its other attrs).
	if meeting.Attrs == nil {
		meeting.Attrs = map[string]any{}
	}
	meeting.Attrs["extraction_status"] = linkageStatusDone
	meeting.UpdatedAt = now
	nodes = append(nodes, *meeting)

	// Resolved/superseded lifecycle: drop prior decision/rationale nodes for this
	// meeting that are absent from this submission.
	if err := s.deleteSupersededDecisions(ctx, meetingURI, keep); err != nil {
		return LinkReport{}, err
	}

	if err := s.store.UpsertNodes(ctx, nodes); err != nil {
		return LinkReport{}, fmt.Errorf("upsert linkage nodes: %w", err)
	}
	if len(edges) > 0 {
		if err := s.store.UpsertEdges(ctx, edges); err != nil {
			return LinkReport{}, fmt.Errorf("upsert linkage edges: %w", err)
		}
	}
	// New edges are in before the superseded ones go, so a failure here loses nothing.
	stale, err := s.staleLinkEdges(ctx, r.TaskID, keep, edges)
	if err != nil {
		return LinkReport{}, err
	}
	if rep.EdgesRemoved, err = store.DeleteEdges(ctx, s.store, stale); err != nil {
		return LinkReport{}, fmt.Errorf("delete superseded links: %w", err)
	}

	rep.LinkedSymbols = len(linkedSymbols)
	rep.UnresolvedRefs = sortedKeys(unresolved)
	return rep, nil
}

// linkKinds are the edge kinds that Submit, and a verification verdict on its decisions, write around a meeting.
var linkKinds = []model.EdgeKind{
	model.EdgeDecidedIn, model.EdgeDiscussedIn, model.EdgeMentioned,
	model.EdgeActionFor, model.EdgeRationaleFor, model.EdgeContradicts,
}

// staleLinkEdges returns the IDs of linkage edges around the meeting and its kept decisions that written omits.
// A submission is the meeting's full linkage: a decision keeps its ID when its summary is unchanged, so without
// this its links to references the new result dropped would stay. Ingest's own edges (meeting to participant)
// are EXTRACTED and stay, and so does recorded drift for a symbol the result still links.
func (s *LinkService) staleLinkEdges(ctx context.Context, meetingID string, keep map[string]bool, written []model.Edge) ([]string, error) {
	kept := make(map[string]bool, len(written))
	governs := map[[2]string]bool{} // {decision, symbol}
	for _, e := range written {
		kept[e.ID] = true
		if e.Kind == model.EdgeDecidedIn {
			governs[[2]string{e.Src, e.Dst}] = true
		}
	}
	seen := map[string]bool{}
	var stale []string
	for _, id := range append(sortedKeys(keep), meetingID) {
		sub, err := s.store.Neighbors(ctx, id, 1, linkKinds)
		if err != nil {
			return nil, fmt.Errorf("load prior links: %w", err)
		}
		for _, e := range sub.Edges {
			if e.Provenance == model.ProvExtracted || kept[e.ID] || seen[e.ID] {
				continue
			}
			if e.Kind == model.EdgeContradicts && governs[[2]string{e.Dst, e.Src}] {
				continue
			}
			seen[e.ID] = true
			stale = append(stale, e.ID)
		}
	}
	return stale, nil
}

// resolveSymbol resolves a label to a node ID by exact case-insensitive symbol match, then exact match on
// another node kind, then substring match. It returns "" rather than invent a link to an unknown symbol.
func (s *LinkService) resolveSymbol(ctx context.Context, label string) (string, error) {
	exact, contains, err := s.matchSymbol(ctx, label)
	if exact != "" || err != nil {
		return exact, err
	}
	return contains, nil
}

// deleteSupersededDecisions removes the meeting's decision and rationale nodes not in keep, under DecisionURI or
// (written before it existed) the meeting's own URI. Stores without NodeDeleter keep stale ones until a rebuild.
func (s *LinkService) deleteSupersededDecisions(ctx context.Context, meetingURI string, keep map[string]bool) error {
	var stale []string
	for _, uri := range []string{DecisionURI(meetingURI), meetingURI} {
		prior, err := store.NodesMatching(ctx, s.store, store.NodeFilter{
			Kinds:     []model.NodeKind{model.KindDecision, model.KindRationale},
			SourceURI: uri,
		})
		if err != nil {
			return fmt.Errorf("load nodes for supersede: %w", err)
		}
		for _, n := range prior {
			// The filter ignores an empty SourceURI, so the check stays for a meeting without one.
			if n.SourceURI == uri && !keep[n.ID] {
				stale = append(stale, n.ID)
			}
		}
	}
	if len(stale) == 0 {
		return nil
	}
	if d, ok := s.store.(store.NodeDeleter); ok {
		if _, err := d.DeleteByIDs(ctx, stale); err != nil {
			return fmt.Errorf("delete superseded decisions: %w", err)
		}
	}
	return nil
}

// personNode builds (or re-builds, idempotently) a PII-tagged person node.
func personNode(name, id string, now time.Time) model.Node {
	return model.Node{
		ID:         id,
		Kind:       model.KindPerson,
		Label:      name,
		SourceURI:  PersonURI(name),
		Attrs:      map[string]any{"pii": true},
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		CreatedAt:  now,
		UpdatedAt:  now,
	}
}

// typedEdge builds an INFERRED/AMBIGUOUS meeting-domain edge with optional attrs.
func typedEdge(src, dst string, kind model.EdgeKind, prov model.Provenance, conf float32, now time.Time, attrs map[string]any) model.Edge {
	return model.Edge{
		ID:         model.NewEdgeID(src, dst, kind),
		Src:        src,
		Dst:        dst,
		Kind:       kind,
		Provenance: prov,
		Confidence: conf,
		Attrs:      attrs,
		CreatedAt:  now,
	}
}

// symbolCandidate is a symbol label lowercased and tokenized once, then matched against every meeting.
type symbolCandidate struct {
	label  string
	lower  string
	tokens []string // lowercased identifier tokens of at least 4 bytes
}

func newSymbolCandidates(labels []string) []symbolCandidate {
	out := make([]symbolCandidate, len(labels))
	for i, lbl := range labels {
		c := symbolCandidate{label: lbl, lower: strings.ToLower(lbl)}
		for _, tok := range identifierTokens(lbl) {
			if len(tok) >= 4 {
				c.tokens = append(c.tokens, strings.ToLower(tok))
			}
		}
		out[i] = c
	}
	return out
}

// candidatesFor returns up to max symbol labels that appear to be referenced in
// the transcript text (substring or identifier-token match), as a hint menu.
func candidatesFor(text string, symbols []symbolCandidate, max int) []string {
	lower := strings.ToLower(text)
	words := tokenSet(lower)
	var out []string
	for _, c := range symbols {
		hit := len(c.lower) >= 3 && strings.Contains(lower, c.lower)
		if !hit {
			for _, tok := range c.tokens {
				if words[tok] {
					hit = true
					break
				}
			}
		}
		if hit {
			out = append(out, c.label)
			if max > 0 && len(out) >= max {
				break
			}
		}
	}
	return out
}

// identifierTokens splits a symbol label into tokens at camelCase, snake_case, dot, and slash boundaries,
// so fetchAll gives [fetch All] and pkg.Method gives [pkg Method].
func identifierTokens(s string) []string {
	var tokens []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			tokens = append(tokens, cur.String())
			cur.Reset()
		}
	}
	var prev rune
	for _, r := range s {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'):
			cur.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			if prev >= 'a' && prev <= 'z' {
				flush() // camelCase boundary
			}
			cur.WriteRune(r)
		default:
			flush()
		}
		prev = r
	}
	flush()
	return tokens
}

func tokenSet(lowerText string) map[string]bool {
	set := map[string]bool{}
	for _, f := range strings.FieldsFunc(lowerText, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	}) {
		set[f] = true
	}
	return set
}

// nonEmptyActions returns the action items with non-empty text, for storage in
// a decision's attrs.
func nonEmptyActions(items []ActionItem) []map[string]any {
	var out []map[string]any
	for _, ai := range items {
		text := strings.TrimSpace(ai.Text)
		if text == "" {
			continue
		}
		m := map[string]any{"text": text}
		if a := strings.TrimSpace(ai.Assignee); a != "" {
			m["assignee"] = a
		}
		out = append(out, m)
	}
	return out
}

func truncateLabel(s string) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	const max = 80
	if len(s) <= max {
		return s
	}
	// Cut on a rune boundary so a multi-byte character is never split, which
	// would persist an invalid-UTF-8 label.
	cut := max
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return strings.TrimSpace(s[:cut]) + "…"
}

func sortedKeys(set map[string]bool) []string {
	if len(set) == 0 {
		return nil
	}
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
