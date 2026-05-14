// Package analyze runs deterministic rules over the extracted graph and
// records rule violations as Finding nodes in the graph store.
package analyze

import (
	"context"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// Severity ranks a finding's importance. Ordering mirrors TrueCourse's model.
type Severity string

const (
	SeverityInfo     Severity = "info"
	SeverityLow      Severity = "low"
	SeverityMedium   Severity = "medium"
	SeverityHigh     Severity = "high"
	SeverityCritical Severity = "critical"
)

// SeverityRank returns an integer ordering for severity comparison.
// Higher = more severe.
func SeverityRank(s Severity) int {
	switch s {
	case SeverityCritical:
		return 4
	case SeverityHigh:
		return 3
	case SeverityMedium:
		return 2
	case SeverityLow:
		return 1
	case SeverityInfo:
		return 0
	}
	return -1
}

// Category is one of the eight fixed buckets. Free-form tags are not supported.
type Category string

const (
	CategoryArchitecture Category = "architecture"
	CategorySecurity     Category = "security"
	CategoryBugs         Category = "bugs"
	CategoryPerformance  Category = "performance"
	CategoryReliability  Category = "reliability"
	CategoryDatabase     Category = "database"
	CategoryQuality      Category = "quality"
	CategoryStyle        Category = "style"
)

// RuleProvenance distinguishes deterministic Go rules from LLM-driven rules.
// All rules in Phase 1.5 are DETERMINISTIC.
type RuleProvenance string

const (
	ProvenanceDeterministic RuleProvenance = "DETERMINISTIC"
	ProvenanceLLM           RuleProvenance = "LLM"
)

// Scope determines how the analyzer dispatches a rule.
// FileScope rules run once per matching file. GraphScope rules run once over
// the entire extracted graph.
type Scope int

const (
	FileScope Scope = iota
	GraphScope
)

// Meta is the metadata a rule registers with the engine. All fields are
// required except Languages (nil = applies to all) and Fix (optional hint).
type Meta struct {
	ID         string
	Category   Category
	Severity   Severity
	Scope      Scope
	Languages  []string // file extensions without the dot, e.g. {"go","py"}. nil = all
	Summary    string
	Fix        string
	Provenance RuleProvenance
}

// FileTarget is the input to a FileScope rule.
type FileTarget struct {
	Path     string
	URI      string // canonical source_uri for the file node
	Content  []byte
	FileNode model.Node    // file node already in the graph
	Nodes    []model.Node  // symbols extracted from this file (excluding the file node)
	Edges    []model.Edge  // edges whose Src or Dst is in this file
}

// GraphTarget is the input to a GraphScope rule.
type GraphTarget struct {
	Nodes []model.Node
	Edges []model.Edge
}

// Rule is the unified rule interface. The analyzer calls Meta() once at
// registration time to learn the rule's category/scope/languages, then calls
// either MatchFile or MatchGraph depending on Meta.Scope.
type Rule interface {
	Meta() Meta
}

// FileRule is a rule with FileScope. Match returns zero or more findings.
type FileRule interface {
	Rule
	MatchFile(ctx context.Context, t FileTarget) []Finding
}

// GraphRule is a rule with GraphScope.
type GraphRule interface {
	Rule
	MatchGraph(ctx context.Context, t GraphTarget) []Finding
}
