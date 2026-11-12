// Package verify re-checks INFERRED claims against the sources they came from. Pending emits a task for each
// claim whose evidence changed since it was last checked, and Submit applies the host assistant's verdict. Like
// the other subagent flows it calls no LLM, and ingested text reaches the assistant only as task evidence.
package verify

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// Task kinds.
const (
	KindConceptSupport = "concept_support" // does the text a concept came from still support it?
	KindDecisionDrift  = "decision_drift"  // does the code a decision governs still match it?
)

// Verdict is the host assistant's answer for one task.
type Verdict string

const (
	Supported   Verdict = "supported"   // concept: the text supports it. decision: the code matches it
	Revised     Verdict = "revised"     // concept only: real, but the description needed correcting
	Unsupported Verdict = "unsupported" // concept only: nothing in the evidence backs it
	Drifted     Verdict = "drifted"     // decision only: the code now does something the decision rules out
	Mislinked   Verdict = "mislinked"   // decision only: the symbol is not what the decision is about
)

// verdicts lists what each task kind accepts.
var verdicts = map[string][]Verdict{
	KindConceptSupport: {Supported, Revised, Unsupported},
	KindDecisionDrift:  {Supported, Drifted, Mislinked},
}

// Attrs Submit keeps on a checked claim: on the concept node, or on a decision's DECIDED_IN edge to the symbol.
const (
	attrVerifiedFP        = "verified_fp" // fingerprint of the claim and evidence the verdict was given for
	attrVerifiedAt        = "verified_at" // RFC 3339
	attrVerdict           = "verdict"
	attrVerdictConfidence = "verdict_confidence"
	attrQuote             = "evidence_quote"
	attrEvidenceID        = "evidence_id" // node the quote came from
	attrNote              = "note"        // drifted: what the code does now
)

const (
	maxEvidence      = 6    // evidence items per task
	maxEvidenceChars = 6000 // per item
	minQuoteChars    = 8
)

// Service emits verification tasks and applies their verdicts.
type Service struct {
	store     store.GraphStore
	links     *meeting.LinkService
	root      string
	threshold float32
	now       func() time.Time
}

// New builds a Service. root confines the source files it reads for decision tasks: a symbol whose file lies
// outside it gets no task. threshold<=0 uses model.DefaultConfidenceThreshold.
func New(s store.GraphStore, root string, threshold float32) *Service {
	if threshold <= 0 {
		threshold = model.DefaultConfidenceThreshold
	}
	return &Service{store: s, links: meeting.NewLinkService(s, threshold), root: root, threshold: threshold, now: time.Now}
}

// Evidence is one source the claim is checked against: a doc section's text, or a symbol's current source lines.
type Evidence struct {
	NodeID    string `json:"node_id"`
	Label     string `json:"label"`
	SourceURI string `json:"source_uri"`
	Text      string `json:"text"`
}

// TaskSpec is one claim to check. Claim and Evidence are ingested data. Schema and Instructions are constants.
type TaskSpec struct {
	ID          string     `json:"id"` // the submit task_id: a concept node ID, or "<decision ID>:<symbol ID>"
	Kind        string     `json:"kind"`
	Label       string     `json:"label"`
	Claim       string     `json:"claim"`
	Evidence    []Evidence `json:"evidence"`
	Fingerprint string     `json:"fingerprint"` // echoed on submit, so a verdict on outdated evidence is refused
	// Truncated is set when the task carries only part of the evidence. A negative verdict is then recorded
	// but not acted on, since the missing part may hold the support.
	Truncated    bool   `json:"truncated,omitempty"`
	Schema       string `json:"schema"`
	Instructions string `json:"instructions"`
}

// claim is one checkable claim as Pending and Submit both derive it from the store.
type claim struct {
	spec   TaskSpec
	node   model.Node // the concept, or the decision
	symbol model.Node // decision claims: the governed node
	link   model.Edge // decision claims: the DECIDED_IN edge from decision to symbol
}

// state returns the attrs holding the claim's verification state.
func (c claim) state() map[string]any {
	if c.spec.Kind == KindDecisionDrift {
		return c.link.Attrs
	}
	return c.node.Attrs
}

// Pending returns tasks for claims never checked, or whose claim or evidence changed since their last verdict.
// kind filters to one task kind (""=both), and limit<=0 means no cap.
func (s *Service) Pending(ctx context.Context, kind string, limit int) ([]TaskSpec, error) {
	var specs []TaskSpec
	full := func() bool { return limit > 0 && len(specs) >= limit }
	add := func(c claim) {
		if model.AttrString(c.state(), attrVerifiedFP) != c.spec.Fingerprint {
			specs = append(specs, c.spec)
		}
	}
	if kind == "" || kind == KindConceptSupport {
		concepts, err := store.NodesMatching(ctx, s.store, store.NodeFilter{Kinds: []model.NodeKind{model.KindConcept}})
		if err != nil {
			return nil, fmt.Errorf("load concepts: %w", err)
		}
		for _, n := range concepts {
			c, ok, err := s.conceptClaim(ctx, n)
			if err != nil {
				return nil, err
			}
			if ok {
				add(c)
			}
			if full() {
				return specs, nil
			}
		}
	}
	if kind == "" || kind == KindDecisionDrift {
		decisions, err := store.NodesMatching(ctx, s.store, store.NodeFilter{Kinds: []model.NodeKind{model.KindDecision}})
		if err != nil {
			return nil, fmt.Errorf("load decisions: %w", err)
		}
		files := sourceFiles{}
		for _, d := range decisions {
			claims, err := s.decisionClaims(ctx, d, files)
			if err != nil {
				return nil, err
			}
			for _, c := range claims {
				add(c)
				if full() {
					return specs, nil
				}
			}
		}
	}
	return specs, nil
}

// Result is the host assistant's structured answer for one task.
type Result struct {
	TaskID      string  `json:"task_id"`
	Fingerprint string  `json:"fingerprint"`
	Verdict     Verdict `json:"verdict"`
	Quote       string  `json:"quote"`       // passage copied from the evidence; required for supported, revised, drifted
	Description string  `json:"description"` // revised: the corrected description
	Note        string  `json:"note"`        // drifted: what the code does now
	Confidence  float32 `json:"confidence"`
}

// Report says what Submit did with a verdict.
type Report struct {
	TaskID  string  `json:"task_id"`
	Verdict Verdict `json:"verdict"`
	Applied bool    `json:"applied"` // false when the verdict was only recorded
	Effect  string  `json:"effect"`
}

// Submit applies one verdict. It refuses a fingerprint that no longer matches the store and a quote that is not
// in the evidence. A verdict below the confidence threshold is recorded, which stops the task coming back, but
// changes nothing else. Re-sending a verdict already recorded is a no-op.
func (s *Service) Submit(ctx context.Context, r Result) (Report, error) {
	if r.TaskID == "" {
		return Report{}, fmt.Errorf("submit: task_id is required")
	}
	if store.IsReadOnly(s.store) {
		return Report{}, fmt.Errorf("submit: store is read-only — restart 'chiton serve --writable' or use 'chiton verify submit'")
	}
	c, err := s.claimFor(ctx, r.TaskID)
	if err != nil {
		return Report{}, err
	}
	if allowed := verdicts[c.spec.Kind]; !slices.Contains(allowed, r.Verdict) {
		return Report{}, fmt.Errorf("submit: verdict %q is not valid for a %s task (want one of %v)", r.Verdict, c.spec.Kind, allowed)
	}
	if r.Fingerprint != c.spec.Fingerprint {
		return Report{}, fmt.Errorf("submit: the claim or evidence for %s changed since the task was issued; call pending_verifications again", r.TaskID)
	}
	rep := Report{TaskID: r.TaskID, Verdict: r.Verdict}
	if state := c.state(); model.AttrString(state, attrVerifiedFP) == r.Fingerprint && model.AttrString(state, attrVerdict) == string(r.Verdict) {
		rep.Effect = "already recorded"
		return rep, nil
	}

	r.Quote, r.Description, r.Note = strings.TrimSpace(r.Quote), strings.TrimSpace(r.Description), strings.TrimSpace(r.Note)
	r.Confidence = min(max(r.Confidence, 0), 1)
	evidenceID := ""
	if r.Verdict == Supported || r.Verdict == Revised || r.Verdict == Drifted {
		if evidenceID, err = quotedFrom(c.spec.Evidence, r.Quote); err != nil {
			return Report{}, fmt.Errorf("submit: %w", err)
		}
	} else {
		r.Quote = ""
	}
	if r.Verdict == Revised && r.Description == "" {
		return Report{}, fmt.Errorf("submit: a revised verdict needs the corrected description")
	}
	if r.Verdict == Drifted && r.Note == "" {
		return Report{}, fmt.Errorf("submit: a drifted verdict needs a note saying what the code does now")
	}

	// The verifier sees only what the task carries, so "nothing here supports it" says nothing about the part
	// left out. A negative verdict on partial evidence is recorded, never acted on.
	confident := r.Confidence >= s.threshold
	negative := r.Verdict == Unsupported || r.Verdict == Mislinked
	rep.Applied = confident && !(negative && c.spec.Truncated)
	record := func(attrs map[string]any, fp string) {
		attrs[attrVerifiedFP] = fp
		attrs[attrVerifiedAt] = s.now().UTC().Format(time.RFC3339)
		attrs[attrVerdict] = string(r.Verdict)
		attrs[attrVerdictConfidence] = r.Confidence
		setOrDelete(attrs, attrQuote, r.Quote)
		setOrDelete(attrs, attrEvidenceID, evidenceID)
	}
	if c.spec.Kind == KindDecisionDrift {
		rep.Effect, err = s.applyToDecision(ctx, c, r, rep.Applied, record)
	} else {
		rep.Effect, err = s.applyToConcept(ctx, c, r, rep.Applied, record)
	}
	if err != nil {
		return Report{}, err
	}
	if !rep.Applied {
		rep.Effect = "recorded only: confidence is below the threshold"
		if confident {
			rep.Effect = "recorded only: the task carried part of the evidence, so a negative verdict is not acted on"
		}
	}
	return rep, nil
}

// claimFor re-derives the claim a task ID names, so Submit judges it against the store as it is now.
func (s *Service) claimFor(ctx context.Context, taskID string) (claim, error) {
	nodeID, symbolID, isPair := strings.Cut(taskID, ":")
	n, err := s.store.GetNode(ctx, nodeID)
	if err != nil {
		return claim{}, fmt.Errorf("load claim: %w", err)
	}
	unknown := fmt.Errorf("submit: no verification task %q (call pending_verifications first)", taskID)
	switch {
	case n == nil:
		return claim{}, unknown
	case isPair && n.Kind == model.KindDecision:
		claims, err := s.decisionClaims(ctx, *n, sourceFiles{})
		if err != nil {
			return claim{}, err
		}
		for _, c := range claims {
			if c.symbol.ID == symbolID {
				return c, nil
			}
		}
		return claim{}, unknown
	case !isPair && n.Kind == model.KindConcept:
		c, ok, err := s.conceptClaim(ctx, *n)
		if err != nil {
			return claim{}, err
		}
		if !ok {
			return claim{}, fmt.Errorf("submit: concept %q has no text evidence to check it against", taskID)
		}
		return c, nil
	}
	return claim{}, fmt.Errorf("submit: node %q is a %s, not a claim a verification task covers", nodeID, n.Kind)
}

// fingerprint identifies a claim together with the evidence it is checked against.
func fingerprint(kind, claimText string, evidence []Evidence) string {
	h := sha256.New()
	write := func(parts ...string) {
		for _, p := range parts {
			io.WriteString(h, p)
			h.Write([]byte{0})
		}
	}
	write(kind, claimText)
	for _, e := range evidence {
		write(e.NodeID, e.Text)
	}
	return hex.EncodeToString(h.Sum(nil)[:16])
}

// quotedFrom returns the ID of the evidence item quote was copied from. Runs of whitespace compare equal, so a
// quote that reflows lines still matches, but every other character must.
func quotedFrom(evidence []Evidence, quote string) (string, error) {
	want := squeeze(quote)
	if utf8.RuneCountInString(want) < minQuoteChars {
		return "", fmt.Errorf("quote must be at least %d characters copied from the evidence", minQuoteChars)
	}
	for _, e := range evidence {
		if strings.Contains(squeeze(e.Text), want) {
			return e.NodeID, nil
		}
	}
	return "", fmt.Errorf("quote is not in the evidence: copy a passage from it character for character")
}

// squeeze collapses each run of whitespace in s to one space.
func squeeze(s string) string { return strings.Join(strings.Fields(s), " ") }

// clip cuts text to maxEvidenceChars bytes on a rune boundary and reports whether it cut anything.
func clip(text string) (string, bool) {
	if len(text) <= maxEvidenceChars {
		return text, false
	}
	n := maxEvidenceChars
	for n > 0 && !utf8.RuneStart(text[n]) {
		n--
	}
	return text[:n], true
}

func setOrDelete(attrs map[string]any, key, value string) {
	if value == "" {
		delete(attrs, key)
		return
	}
	attrs[key] = value
}
