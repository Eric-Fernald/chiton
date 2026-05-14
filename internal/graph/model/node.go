package model

import (
	"crypto/sha256"
	"encoding/hex"
	"time"
)

type NodeKind string

const (
	KindFile        NodeKind = "file"
	KindSymbol      NodeKind = "symbol"
	KindConcept     NodeKind = "concept"
	KindDocSection  NodeKind = "doc_section"
	KindImage       NodeKind = "image"
	KindMeeting     NodeKind = "meeting"
	KindMeetingTurn NodeKind = "meeting_turn"
	KindDecision    NodeKind = "decision"
	KindPerson      NodeKind = "person"
	KindURL         NodeKind = "url"
	KindRationale   NodeKind = "rationale"
	KindRule        NodeKind = "rule"
	KindFinding     NodeKind = "finding"
)

type Provenance string

const (
	ProvExtracted Provenance = "EXTRACTED"
	ProvInferred  Provenance = "INFERRED"
	ProvAmbiguous Provenance = "AMBIGUOUS"
)

type SourceSpan struct {
	StartLine  int   `json:"start_line"`
	EndLine    int   `json:"end_line"`
	ByteOffset int64 `json:"byte_offset"`
}

type Node struct {
	ID            string         `json:"id"`
	Kind          NodeKind       `json:"kind"`
	Label         string         `json:"label"`
	SourceURI     string         `json:"source_uri"`
	Span          *SourceSpan    `json:"span,omitempty"`
	Embedding     []float32      `json:"embedding,omitempty"`
	Attrs         map[string]any `json:"attrs,omitempty"`
	Provenance    Provenance     `json:"provenance"`
	Confidence    float32        `json:"confidence"`
	CommunityID   *int           `json:"community_id,omitempty"`
	ContentSHA256 string         `json:"content_sha256,omitempty"`
	CreatedAt     time.Time      `json:"created_at"`
	UpdatedAt     time.Time      `json:"updated_at"`
}

// NewNodeID returns a stable 16-byte hex ID derived from sourceURI and label.
func NewNodeID(sourceURI, canonicalLabel string) string {
	h := sha256.Sum256([]byte(sourceURI + "::" + canonicalLabel))
	return hex.EncodeToString(h[:16])
}
