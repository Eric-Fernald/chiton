package model

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
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
	KindEmail       NodeKind = "email"
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

// ContentSHA returns the lowercase hex SHA-256 of b, the form stored in Node.ContentSHA256.
func ContentSHA(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// IsExternal reports whether n is an extractor stub for something outside the scanned tree, such as an
// imported package, which extractors mark with attrs.external=true.
func IsExternal(n Node) bool {
	ext, _ := n.Attrs["external"].(bool)
	return ext
}

// Go file nodes carry these attrs when the file sits inside a module: the package's import path and the
// module path. They tie a function's definition to the pkg:// reference its callers point at.
const (
	AttrImportPath = "import_path"
	AttrModule     = "module"
)

// PkgFuncRefID is the ID of the pkg:// reference node a Go call names when the called package-level function
// is defined outside the calling file: funcName in the package at importPath.
func PkgFuncRefID(importPath, funcName string) string {
	return NewNodeID("pkg://"+importPath, funcName)
}

// PkgRefImportPath returns the import path of a pkg:// reference node, and false for any other node.
func PkgRefImportPath(n Node) (string, bool) {
	p, ok := strings.CutPrefix(n.SourceURI, "pkg://")
	return p, ok && IsExternal(n)
}
