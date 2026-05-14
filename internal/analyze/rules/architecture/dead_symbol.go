package architecture

import (
	"context"
	"fmt"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DeadSymbol flags non-exported symbols with zero non-CONTAINS in-edges.
// A symbol still imported via CONTAINS (its enclosing file) is fine — that
// edge isn't a "use" of the symbol, just structural.
//
// Heuristics:
//   - Test files contribute their CALLS edges (so symbols used only from
//     tests count as live). Some shops do want to flag test-only code but
//     that's a separate rule.
//   - Exported names (capitalized in Go, public in others) are skipped to
//     avoid noise on API surfaces a chiton run doesn't see callers for.
type DeadSymbol struct{}

func (DeadSymbol) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "architecture/dead-symbol",
		Category:   analyze.CategoryArchitecture,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.GraphScope,
		Summary:    "Non-exported symbol has no incoming references.",
		Fix:        "Remove the symbol or confirm it is intentionally reachable only via reflection.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (DeadSymbol) MatchGraph(_ context.Context, t analyze.GraphTarget) []analyze.Finding {
	nodes := indexNodesByID(t.Nodes)

	// Count non-CONTAINS in-edges for each symbol.
	inCount := map[string]int{}
	for _, e := range t.Edges {
		if e.Kind == model.EdgeContains {
			continue
		}
		inCount[e.Dst]++
	}

	var out []analyze.Finding
	for _, n := range t.Nodes {
		if n.Kind != model.KindSymbol {
			continue
		}
		if isExternal(n) {
			continue
		}
		if isExportedName(n.Label) {
			continue
		}
		if inCount[n.ID] > 0 {
			continue
		}
		// Skip symbols defined in test files.
		if isTestSource(n.SourceURI) {
			continue
		}
		// Make sure the node is a function-like — flagging unused types adds
		// too much noise for a starter rule.
		lk, _ := n.Attrs["lang_kind"].(string)
		if !isFuncLike(lk) {
			continue
		}
		_ = nodes // currently unused, retained for future neighbour lookups
		out = append(out, analyze.Finding{
			RuleID:       "architecture/dead-symbol",
			Severity:     analyze.SeverityLow,
			Category:     analyze.CategoryArchitecture,
			Message:      fmt.Sprintf("symbol %q has no incoming references", n.Label),
			Fix:          "Remove if unused, or confirm reflective access.",
			TargetURI:    n.SourceURI,
			TargetNodeID: n.ID,
			Span:         n.Span,
		})
	}
	return out
}

func isFuncLike(langKind string) bool {
	switch langKind {
	case "func", "method", "arrow":
		return true
	}
	return false
}

// isExportedName is a language-agnostic heuristic: a leading uppercase letter
// (Go), a leading underscore being absent (Python convention), or a public
// keyword on the same line — we only have the label here so use the cheapest
// signal that catches Go exports.
func isExportedName(label string) bool {
	if label == "" {
		return false
	}
	// Strip a leading qualifier like "Class." or "package.".
	if idx := strings.LastIndex(label, "."); idx >= 0 && idx+1 < len(label) {
		label = label[idx+1:]
	}
	c := label[0]
	return c >= 'A' && c <= 'Z'
}

func isTestSource(uri string) bool {
	low := strings.ToLower(uri)
	switch {
	case strings.HasSuffix(low, "_test.go"):
		return true
	case strings.Contains(low, ".test."), strings.Contains(low, ".spec."):
		return true
	case strings.Contains(low, "/tests/"), strings.Contains(low, "\\tests\\"):
		return true
	}
	return false
}

func init() { analyze.Register(DeadSymbol{}) }
