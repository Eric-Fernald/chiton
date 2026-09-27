package analyze

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// Finding is one rule violation. Stable IDs let diff_findings compare scans.
type Finding struct {
	RuleID   string
	Severity Severity
	Category Category
	Message  string
	Fix      string // optional patch suggestion (free-form text for now)

	// At least one must be set: TargetURI is the file's source_uri, TargetNodeID the symbol/file node.
	TargetURI    string
	TargetNodeID string

	// Optional positional info; finding ID hashes Span to keep stability.
	Span *model.SourceSpan

	// Set only for LLM-rule findings; deterministic findings leave them zero (see resolved).
	NodeProvenance model.Provenance
	Confidence     float32
	RuleProvenance RuleProvenance
}

// resolved maps zero provenance fields to EXTRACTED/1.0/DETERMINISTIC; setting either
// NodeProvenance or RuleProvenance marks the finding LLM-sourced, keeping its confidence.
func (f Finding) resolved() (model.Provenance, float32, RuleProvenance) {
	if f.NodeProvenance == "" && f.RuleProvenance == "" {
		return model.ProvExtracted, 1.0, ProvenanceDeterministic
	}
	prov := f.NodeProvenance
	if prov == "" {
		prov = model.ProvInferred
	}
	ruleProv := f.RuleProvenance
	if ruleProv == "" {
		ruleProv = ProvenanceLLM
	}
	return prov, f.Confidence, ruleProv
}

// FindingID returns the stable ID: a hash of rule, target, and start line (not message), so the
// same offence at the same place reuses its node across scans.
func (f Finding) FindingID() string {
	loc := f.TargetURI
	if f.Span != nil {
		loc += "#L" + strconv.Itoa(f.Span.StartLine)
	}
	h := sha256.Sum256([]byte(f.RuleID + "::" + loc))
	return hex.EncodeToString(h[:16])
}

// asNode renders the finding as a graph node; id is f.FindingID(), passed in so callers hash it once.
func (f Finding) asNode(id string, now time.Time) model.Node {
	uri := "finding://" + id
	label := f.RuleID
	if f.Span != nil {
		label += "@L" + strconv.Itoa(f.Span.StartLine)
	}
	prov, conf, ruleProv := f.resolved()
	attrs := map[string]any{
		"rule_id":         f.RuleID,
		"severity":        string(f.Severity),
		"category":        string(f.Category),
		"message":         f.Message,
		"target_uri":      f.TargetURI,
		"target_node_id":  f.TargetNodeID,
		"rule_provenance": string(ruleProv),
	}
	if f.Fix != "" {
		attrs["fix_suggestion"] = f.Fix
	}
	if ruleProv == ProvenanceLLM {
		attrs["confidence_score"] = conf
	}
	return model.Node{
		ID:         id,
		Kind:       model.KindFinding,
		Label:      label,
		SourceURI:  uri,
		Span:       f.Span,
		Attrs:      attrs,
		Provenance: prov,
		Confidence: conf,
		CreatedAt:  now,
		UpdatedAt:  now,
	}
}

// violatesEdge returns the finding→rule edge; id and rule are f.FindingID() and RuleNodeID(f.RuleID).
func (f Finding) violatesEdge(id, rule string, now time.Time) model.Edge {
	prov, conf, _ := f.resolved()
	return model.Edge{
		ID:         model.NewEdgeID(id, rule, model.EdgeViolates),
		Src:        id,
		Dst:        rule,
		Kind:       model.EdgeViolates,
		Provenance: prov,
		Confidence: conf,
		CreatedAt:  now,
	}
}

// affectsEdge returns the finding→target edge, or a zero Edge if no target node is set (callers should
// check Edge.ID != ""); id is f.FindingID().
func (f Finding) affectsEdge(id string, now time.Time) model.Edge {
	target := f.TargetNodeID
	if target == "" {
		return model.Edge{}
	}
	prov, conf, _ := f.resolved()
	return model.Edge{
		ID:         model.NewEdgeID(id, target, model.EdgeAffects),
		Src:        id,
		Dst:        target,
		Kind:       model.EdgeAffects,
		Provenance: prov,
		Confidence: conf,
		CreatedAt:  now,
	}
}

// appendGraph appends f's node, VIOLATES edge, and AFFECTS edge (when it has a target node); id and rule
// are f.FindingID() and RuleNodeID(f.RuleID).
func (f Finding) appendGraph(nodes []model.Node, edges []model.Edge, id, rule string, now time.Time) ([]model.Node, []model.Edge) {
	nodes = append(nodes, f.asNode(id, now))
	edges = append(edges, f.violatesEdge(id, rule, now))
	if ae := f.affectsEdge(id, now); ae.ID != "" {
		edges = append(edges, ae)
	}
	return nodes, edges
}

// RuleNodeID returns the stable graph-node ID for a rule.
func RuleNodeID(ruleID string) string {
	h := sha256.Sum256([]byte("rule::" + ruleID))
	return hex.EncodeToString(h[:16])
}

// RuleAsNode renders rule metadata as a graph node, so list_findings can join
// findings to their rule via the VIOLATES edge.
func RuleAsNode(m Meta, now time.Time) model.Node {
	id := RuleNodeID(m.ID)
	attrs := map[string]any{
		"rule_id":    m.ID,
		"category":   string(m.Category),
		"severity":   string(m.Severity),
		"summary":    m.Summary,
		"provenance": string(m.Provenance),
		"source":     "built-in",
	}
	if m.Fix != "" {
		attrs["fix_hint"] = m.Fix
	}
	if len(m.Languages) > 0 {
		attrs["languages"] = m.Languages
	}
	return model.Node{
		ID:         id,
		Kind:       model.KindRule,
		Label:      m.ID,
		SourceURI:  "rule://" + m.ID,
		Attrs:      attrs,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		CreatedAt:  now,
		UpdatedAt:  now,
	}
}
