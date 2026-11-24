package bugs

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// LooseEquality flags JS/TS `==` and `!=` in place of the strict `===` and `!==`. Loose equality coerces
// types (0 == "", null == undefined), a frequent source of subtle bugs.
type LooseEquality struct{}

func (LooseEquality) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "bugs/loose-equality",
		Category:   analyze.CategoryBugs,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"js", "ts", "jsx", "tsx"},
		Summary:    "Loose equality (== / !=) performs type coercion; prefer === / !==.",
		Fix:        "Use === or !== for strict comparison.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// looseEqualityRe matches `==` or `!=` that is not part of a longer `=` run (`===`, `!==`) and not
// preceded by `<`, `>`, or `!`.
var looseEqualityRe = regexp.MustCompile(`[^=!<>]([!=]=)([^=]|$)`)

func (LooseEquality) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	// Whole-file masking keeps `==` and `!=` inside strings, templates, regex literals, and comments
	// (SQL fragments, base64 padding) away from the operator regex.
	masked := maskJSLiteralsAndComments(t.Content)
	for line, text := range analyze.Lines(masked) {
		if looseEqualityRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "bugs/loose-equality",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryBugs,
				Message:      "loose equality (== or !=) coerces types; use === / !==",
				Fix:          "Replace == with === and != with !==.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(LooseEquality{}) }
