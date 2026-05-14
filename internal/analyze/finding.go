package analyze

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// Finding is one rule violation. Stable IDs let diff_findings compare scans.
type Finding struct {
	RuleID    string
	Severity  Severity
	Category  Category
	Message   string
	Fix       string // optional patch suggestion (free-form text for now)

	// At least one of TargetURI or TargetNodeID must be set.
	// TargetURI is the source_uri of the file the finding refers to.
	// TargetNodeID points at the specific symbol/file node (if known).
	TargetURI    string
	TargetNodeID string

	// Optional positional info; finding ID hashes Span to keep stability.
	Span *model.SourceSpan
}

// FindingID returns the stable ID for this finding.
// Hash includes rule, target, and start line so the same offence at the
// same place reuses its node across scans.
func (f Finding) FindingID() string {
	loc := f.TargetURI
	if f.Span != nil {
		loc = fmt.Sprintf("%s#L%d", loc, f.Span.StartLine)
	}
	h := sha256.Sum256([]byte(f.RuleID + "::" + loc))
	return hex.EncodeToString(h[:16])
}

// AsNode renders the finding as a graph node.
func (f Finding) AsNode(now time.Time) model.Node {
	id := f.FindingID()
	uri := "finding://" + id
	label := f.RuleID
	if f.Span != nil {
		label = fmt.Sprintf("%s@L%d", f.RuleID, f.Span.StartLine)
	}
	attrs := map[string]any{
		"rule_id":         f.RuleID,
		"severity":        string(f.Severity),
		"category":        string(f.Category),
		"message":         f.Message,
		"target_uri":      f.TargetURI,
		"target_node_id":  f.TargetNodeID,
		"rule_provenance": string(ProvenanceDeterministic),
	}
	if f.Fix != "" {
		attrs["fix_suggestion"] = f.Fix
	}
	return model.Node{
		ID:         id,
		Kind:       model.KindFinding,
		Label:      label,
		SourceURI:  uri,
		Span:       f.Span,
		Attrs:      attrs,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		CreatedAt:  now,
		UpdatedAt:  now,
	}
}

// ViolatesEdge returns the finding→rule edge.
func (f Finding) ViolatesEdge(now time.Time) model.Edge {
	return model.Edge{
		ID:         model.NewEdgeID(f.FindingID(), RuleNodeID(f.RuleID), model.EdgeViolates),
		Src:        f.FindingID(),
		Dst:        RuleNodeID(f.RuleID),
		Kind:       model.EdgeViolates,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		CreatedAt:  now,
	}
}

// AffectsEdge returns the finding→target edge, or a zero Edge if no target
// node is set (callers should check Edge.ID != "").
func (f Finding) AffectsEdge(now time.Time) model.Edge {
	target := f.TargetNodeID
	if target == "" {
		return model.Edge{}
	}
	return model.Edge{
		ID:         model.NewEdgeID(f.FindingID(), target, model.EdgeAffects),
		Src:        f.FindingID(),
		Dst:        target,
		Kind:       model.EdgeAffects,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		CreatedAt:  now,
	}
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
		"rule_id":     m.ID,
		"category":    string(m.Category),
		"severity":    string(m.Severity),
		"summary":     m.Summary,
		"provenance":  string(m.Provenance),
		"source":      "built-in",
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
