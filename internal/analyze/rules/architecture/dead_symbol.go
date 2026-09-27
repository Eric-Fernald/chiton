package architecture

import (
	"context"
	"fmt"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DeadSymbol flags non-exported func-like symbols with no in-edges other than CONTAINS. CALLS from tests
// count as uses. It skips exported names, since their callers may lie outside the graph.
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
		// init and main are invoked by the runtime, never via graph edges.
		if n.Label == "init" || n.Label == "main" {
			continue
		}
		// Skip methods ("Type.method") because the Go extractor emits no CALLS edges for receiver calls,
		// so every private method would look dead.
		if strings.Contains(n.Label, ".") {
			continue
		}
		if inCount[n.ID] > 0 {
			continue
		}
		// Skip symbols defined in test files.
		if analyze.IsTestSource(n.SourceURI) {
			continue
		}
		// Make sure the node is a function-like — flagging unused types adds
		// too much noise for a starter rule.
		lk, _ := n.Attrs["lang_kind"].(string)
		if !analyze.IsFuncKind(lk) {
			continue
		}
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

// isExportedName reports whether the label's last dotted segment starts with an ASCII capital
// (the Go export rule), the only signal a bare label offers.
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

func init() { analyze.Register(DeadSymbol{}) }
