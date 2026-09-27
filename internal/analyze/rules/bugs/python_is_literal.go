package bugs

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// PythonIsLiteral flags `is` / `is not` against a string or numeric literal (`x is "foo"`, `n is 5`):
// CPython's small-int/interned-string caching makes it seem to work, then break. None/True/False pass.
type PythonIsLiteral struct{}

func (PythonIsLiteral) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "bugs/python-is-literal",
		Category:   analyze.CategoryBugs,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"py"},
		Summary:    "`is` / `is not` compared to a literal tests identity, not equality; use ==.",
		Fix:        "Use == / != for value comparison (reserve is/is not for None).",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// `is` / `is not` then a string or numeric literal; the non-dot/word lead char skips `.is(...)` calls
// and identifiers like `axis`.
var pyIsLiteralRe = regexp.MustCompile(
	`(^|[^.\w])is(\s+not)?\s+(["']|[+\-]?\d+(\.\d+)?\b|\.\d+\b)`)

func (PythonIsLiteral) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	// Whole-file masking so `is <literal>` inside a string or (multi-line) docstring never matches.
	masked := maskPyStringsAndComments(t.Content)
	for line, text := range analyze.Lines(masked) {
		if pyIsLiteralRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "bugs/python-is-literal",
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryBugs,
				Message:      "`is`/`is not` compared to a literal tests identity, not value; use ==",
				Fix:          "Use == / != for value comparison.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(PythonIsLiteral{}) }
