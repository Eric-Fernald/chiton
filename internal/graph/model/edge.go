package model

import (
	"crypto/sha256"
	"encoding/hex"
	"time"
)

type EdgeKind string

// Structural edges — derived from AST or git history.
const (
	EdgeContains   EdgeKind = "CONTAINS"
	EdgeImports    EdgeKind = "IMPORTS"
	EdgeCalls      EdgeKind = "CALLS"
	EdgeExtends    EdgeKind = "EXTENDS"
	EdgeImplements EdgeKind = "IMPLEMENTS"
	EdgeReferences EdgeKind = "REFERENCES"
	EdgeDefinedIn  EdgeKind = "DEFINED_IN"
	EdgeModifiedBy EdgeKind = "MODIFIED_BY"
)

// Semantic edges — derived from LLM extraction.
const (
	EdgeDescribes   EdgeKind = "DESCRIBES"
	EdgeRationaleFor EdgeKind = "RATIONALE_FOR"
	EdgeSimilarTo   EdgeKind = "SIMILAR_TO"
	EdgeContradicts EdgeKind = "CONTRADICTS"
	EdgeElaborates  EdgeKind = "ELABORATES"
	EdgeDerivedFrom EdgeKind = "DERIVED_FROM"
)

// Meeting/discussion edges.
const (
	EdgeDiscussedIn EdgeKind = "DISCUSSED_IN"
	EdgeDecidedIn   EdgeKind = "DECIDED_IN"
	EdgeActionFor   EdgeKind = "ACTION_FOR"
	EdgeMentioned   EdgeKind = "MENTIONED"
)

// Hyperedge membership.
const (
	EdgeMember EdgeKind = "MEMBER"
)

type Edge struct {
	ID         string         `json:"id"`
	Src        string         `json:"src"`
	Dst        string         `json:"dst"`
	Kind       EdgeKind       `json:"kind"`
	Provenance Provenance     `json:"provenance"`
	Confidence float32        `json:"confidence"`
	Attrs      map[string]any `json:"attrs,omitempty"`
	CreatedAt  time.Time      `json:"created_at"`
}

// NewEdgeID returns a stable ID from src, dst, and kind.
func NewEdgeID(src, dst string, kind EdgeKind) string {
	h := sha256.Sum256([]byte(src + "::" + string(kind) + "::" + dst))
	return hex.EncodeToString(h[:16])
}
