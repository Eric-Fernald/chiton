// Package quality contains deterministic rules for the code-quality category.
package quality

import (
	"context"
	"fmt"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// LongFunction flags functions exceeding LongFunctionLines lines.
// Reads from the extracted symbol nodes' spans, so no source re-walk needed.
const LongFunctionLines = 80

type LongFunction struct{}

func (LongFunction) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "quality/long-function",
		Category:   analyze.CategoryQuality,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Summary:    fmt.Sprintf("Function exceeds %d lines.", LongFunctionLines),
		Fix:        "Extract helpers or split the function along logical seams.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (LongFunction) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for _, n := range t.Nodes {
		if n.Kind != model.KindSymbol || n.Span == nil {
			continue
		}
		// Only treat nodes the extractor explicitly tagged as a function-like
		// kind; class/struct/etc. nodes also live in symbol-space.
		lk, _ := n.Attrs["lang_kind"].(string)
		if !isFuncLike(lk) {
			continue
		}
		lines := n.Span.EndLine - n.Span.StartLine + 1
		if lines <= LongFunctionLines {
			continue
		}
		out = append(out, analyze.Finding{
			RuleID:       "quality/long-function",
			Severity:     analyze.SeverityLow,
			Category:     analyze.CategoryQuality,
			Message:      fmt.Sprintf("function %q is %d lines (limit %d)", n.Label, lines, LongFunctionLines),
			Fix:          "Extract helper functions or split along logical seams.",
			TargetURI:    t.URI,
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

func init() { analyze.Register(LongFunction{}) }
