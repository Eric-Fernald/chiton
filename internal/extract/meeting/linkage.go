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

// linkageStatusDone marks a meeting whose linkage is complete, so Pending will
// not re-emit it. Reuses the same attr key as semantic extraction; meeting and
// doc/image nodes never collide because each Pending only scans its own kinds.
const linkageStatusDone = "done"

// DefaultConfidenceThreshold splits INFERRED from AMBIGUOUS when none is set.
const DefaultConfidenceThreshold = 0.5

// defaultMaxCandidates caps the candidate-symbol hint menu in a task spec, so a
// large corpus does not blow the host assistant's context budget. The menu is a
// hint only — submit resolves any returned label against the whole graph.
const defaultMaxCandidates = 300

// defaultDiscussedConfidence is the confidence assigned to a "merely discussed"
// symbol link, which the host reports without a per-item score. It is a real
// confidence (not the threshold), run through resolveProvenance like every other
// INFERRED edge so it splits INFERRED/AMBIGUOUS consistently.
const defaultDiscussedConfidence float32 = 0.7

// LinkService runs chiton's meeting→code linkage in subagent mode: it emits one
// task spec per un-linked meeting (the transcript text plus a menu of candidate
// code-symbol labels), and ingests the host assistant's structured decisions,
// writing decision/rationale nodes and DECIDED_IN / DISCUSSED_IN / ACTION_FOR /
// MENTIONED / RATIONALE_FOR edges tagged INFERRED (or AMBIGUOUS below the
// confidence threshold). chiton calls no LLM itself — exactly mirroring the
// doc/image semantic-extraction and LLM-rule paths. This is the capability the
// plan calls out as unique: meeting decisions linked back to the code symbols
// they reference.
type LinkService struct {
	store         store.GraphStore
	threshold     float32
	maxCandidates int
	now           func() time.Time
}

// NewLinkService builds a LinkService. threshold<=0 uses
// DefaultConfidenceThreshold.
func NewLinkService(s store.GraphStore, threshold float32) *LinkService {
	if threshold <= 0 {
		threshold = DefaultConfidenceThreshold
	}
	return &LinkService{store: s, threshold: threshold, maxCandidates: defaultMaxCandidates, now: time.Now}
}

// LinkTaskSpec is one unit of linkage work for the host assistant. Input is the
// transcript text; CandidateSymbols is a hint menu of code-symbol labels that
// appear to be referenced. Schema/Instructions pin the JSON the assistant must
// return via submit_meeting_links / `chiton meeting submit`.
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
	LinkedSymbols  int      `json:"linked_symbols"`
	Ambiguous      int      `json:"ambiguous"`
	UnresolvedRefs []string `json:"unresolved_refs,omitempty"`
}

// Pending returns linkage task specs for meetings not yet linked (meeting nodes
// whose attrs["extraction_status"] != "done"). limit<=0 means no cap.
func (s *LinkService) Pending(ctx context.Context, limit int) ([]LinkTaskSpec, error) {
	nodes, err := s.store.AllNodes(ctx)
	if err != nil {
		return nil, fmt.Errorf("load nodes: %w", err)
	}

	// Collect candidate code-symbol labels once for the whole corpus.
	var symbolLabels []string
	seenLabel := map[string]bool{}
	for _, n := range nodes {
		if n.Kind != model.KindSymbol && n.Kind != model.KindFile {
			continue
		}
		lbl := strings.TrimSpace(n.Label)
		if lbl == "" || seenLabel[lbl] {
			continue
		}
		seenLabel[lbl] = true
		symbolLabels = append(symbolLabels, lbl)
	}
	sort.Strings(symbolLabels)

	var specs []LinkTaskSpec
	for _, n := range nodes {
		if n.Kind != model.KindMeeting {
			continue
		}
		if attrString(n, "extraction_status") == linkageStatusDone {
			continue
		}
		text := attrString(n, "text")
		if strings.TrimSpace(text) == "" {
			continue
		}
		specs = append(specs, LinkTaskSpec{
			ID:               n.ID,
			Kind:             KindMeetingLinkage,
			Label:            n.Label,
			SourceURI:        n.SourceURI,
			Input:            text,
			CandidateSymbols: candidatesFor(text, symbolLabels, s.maxCandidates),
			Schema:           linkSchema,
			Instructions:     linkInstructions,
		})
		if limit > 0 && len(specs) >= limit {
			break
		}
	}
	return specs, nil
}

// Submit ingests one meeting's linkage results. It is idempotent: stable IDs
// make re-submitting the same result a no-op, and prior decision/rationale
// nodes for the meeting that are absent from the new result are deleted (the
// decision is resolved/superseded), exactly like the LLM-rule resolved
// lifecycle.
func (s *LinkService) Submit(ctx context.Context, r LinkResult) (LinkReport, error) {
	if r.TaskID == "" {
		return LinkReport{}, fmt.Errorf("submit: task_id is required")
	}
	if ro, ok := s.store.(store.ReadOnlyAware); ok && ro.ReadOnly() {
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
		prov, conf := resolveProvenance(d.Confidence, s.threshold)
		if prov == model.ProvAmbiguous {
			rep.Ambiguous++
		}
		// Key the decision (and its rationale) on its position as well as its
		// summary, so two decisions with identical summaries in one meeting do not
		// collide onto a single node. Re-submit stays clean because
		// deleteSupersededDecisions prunes any prior decision/rationale not in this
		// submission's keep set.
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
			SourceURI:  meetingURI,
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
				SourceURI:  meetingURI,
				Attrs:      map[string]any{"text": rat, "decision_id": decID},
				Provenance: prov,
				Confidence: conf,
				CreatedAt:  now,
				UpdatedAt:  now,
			})
			edges = append(edges, typedEdge(ratID, decID, model.EdgeRationaleFor, prov, conf, now, nil))
			rep.EdgesAdded++
		}

		// References → code symbols the decision governs.
		for _, ref := range d.References {
			symID := s.resolveSymbol(ctx, ref)
			if symID == "" {
				if r := strings.TrimSpace(ref); r != "" {
					unresolved[r] = true
				}
				continue
			}
			linkedSymbols[symID] = true
			// decision --DECIDED_IN--> symbol (the code the decision governs).
			edges = append(edges, typedEdge(decID, symID, model.EdgeDecidedIn, prov, conf, now, nil))
			// symbol --DISCUSSED_IN--> meeting (code discussed in the meeting).
			edges = append(edges, typedEdge(symID, r.TaskID, model.EdgeDiscussedIn, prov, conf, now, nil))
			// meeting --MENTIONED--> symbol, so "what did this meeting mention?"
			// is complete for decision-governed symbols too, matching the
			// discussed-only path below.
			edges = append(edges, typedEdge(r.TaskID, symID, model.EdgeMentioned, prov, conf, now, nil))
			rep.EdgesAdded += 3
		}

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
			// Dedupe the person node within this submission (the same assignee may
			// own several action items); UpsertNodes also dedupes by ID, but this
			// keeps the batch clean, matching the file's other dedup maps.
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
		symID := s.resolveSymbol(ctx, ref)
		if symID == "" {
			if rr := strings.TrimSpace(ref); rr != "" {
				unresolved[rr] = true
			}
			continue
		}
		linkedSymbols[symID] = true
		prov, conf := resolveProvenance(defaultDiscussedConfidence, s.threshold)
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

	rep.LinkedSymbols = len(linkedSymbols)
	rep.UnresolvedRefs = sortedKeys(unresolved)
	return rep, nil
}

// resolveSymbol resolves a code-symbol label to a node ID, preferring an exact
// (case-insensitive) label match on a symbol node, then a file/other node, then
// an exact-substring match. Returns "" when nothing resolves — chiton never
// fabricates a link to a symbol it cannot find (the same discipline the AST
// extractor applies to CALLS edges).
func (s *LinkService) resolveSymbol(ctx context.Context, label string) string {
	label = strings.TrimSpace(label)
	if label == "" {
		return ""
	}
	nodes, err := s.store.SearchNodes(ctx, label, 25)
	if err != nil || len(nodes) == 0 {
		return ""
	}
	lower := strings.ToLower(label)
	var exactOther, contains string
	for _, n := range nodes {
		if n.Kind == model.KindMeeting || n.Kind == model.KindMeetingTurn ||
			n.Kind == model.KindPerson || n.Kind == model.KindDecision ||
			n.Kind == model.KindRationale {
			continue // never link a decision to another meeting-domain node
		}
		if strings.EqualFold(n.Label, label) {
			if n.Kind == model.KindSymbol {
				return n.ID // best: exact symbol match
			}
			if exactOther == "" {
				exactOther = n.ID
			}
			continue
		}
		if contains == "" && strings.Contains(strings.ToLower(n.Label), lower) {
			contains = n.ID
		}
	}
	if exactOther != "" {
		return exactOther
	}
	return contains
}

// deleteSupersededDecisions removes decision/rationale nodes under meetingURI
// that are not in keep, giving decisions a resolved/superseded lifecycle on
// re-submit. Uses NodeDeleter (removes adjacent DECIDED_IN/RATIONALE_FOR/
// ACTION_FOR edges too) when available; otherwise it is a no-op fallback so a
// minimal store still works (stale decisions then linger until rebuild).
func (s *LinkService) deleteSupersededDecisions(ctx context.Context, meetingURI string, keep map[string]bool) error {
	all, err := s.store.AllNodes(ctx)
	if err != nil {
		return fmt.Errorf("load nodes for supersede: %w", err)
	}
	var stale []string
	for _, n := range all {
		if n.Kind != model.KindDecision && n.Kind != model.KindRationale {
			continue
		}
		if n.SourceURI != meetingURI || keep[n.ID] {
			continue
		}
		stale = append(stale, n.ID)
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

// candidatesFor returns up to max symbol labels that appear to be referenced in
// the transcript text (substring or identifier-token match), as a hint menu.
func candidatesFor(text string, symbols []string, max int) []string {
	lower := strings.ToLower(text)
	words := tokenSet(lower)
	var out []string
	for _, lbl := range symbols {
		low := strings.ToLower(lbl)
		hit := false
		if len(low) >= 3 && strings.Contains(lower, low) {
			hit = true
		} else {
			for _, tok := range identifierTokens(lbl) {
				if len(tok) >= 4 && words[strings.ToLower(tok)] {
					hit = true
					break
				}
			}
		}
		if hit {
			out = append(out, lbl)
			if max > 0 && len(out) >= max {
				break
			}
		}
	}
	return out
}

// identifierTokens splits a symbol label into word tokens, handling camelCase,
// snake_case, dotted, and slashed names (fetchAll → [fetch All], do_work →
// [do work], pkg.Method → [pkg Method]).
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

// resolveProvenance maps a host-supplied confidence to a provenance tag and a
// clamped confidence. At or above threshold → INFERRED; below → AMBIGUOUS.
func resolveProvenance(conf, threshold float32) (model.Provenance, float32) {
	if conf < 0 {
		conf = 0
	}
	if conf > 1 {
		conf = 1
	}
	if conf >= threshold {
		return model.ProvInferred, conf
	}
	return model.ProvAmbiguous, conf
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

func attrString(n model.Node, key string) string {
	if n.Attrs == nil {
		return ""
	}
	if v, ok := n.Attrs[key].(string); ok {
		return v
	}
	return ""
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
