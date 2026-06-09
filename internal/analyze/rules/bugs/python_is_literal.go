package bugs

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// PythonIsLiteral flags Python identity comparisons (`is` / `is not`) against a
// string or numeric literal, e.g. `x is "foo"` or `n is 5`. `is` compares object
// identity, not value; CPython caches small ints / interned strings so it may
// appear to work, then break unpredictably. `==` is almost always intended.
// (Comparisons to None/True/False are correct and NOT flagged.)
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

// `is` or `is not` followed by a string literal (', ") or a numeric literal.
// A leading word boundary plus a non-dot/word char prevents matching `.is(...)`
// method calls or identifiers like `axis`.
var pyIsLiteralRe = regexp.MustCompile(
	`(^|[^.\w])is(\s+not)?\s+(["']|[+\-]?\d+(\.\d+)?\b|\.\d+\b)`)

func (PythonIsLiteral) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	// Mask Python string-literal bodies (incl. single- and multi-line
	// triple-quoted strings) and `#` comments across the whole file, so the word
	// `is` followed by a literal INSIDE a string or docstring no longer matches.
	masked := maskPyStringsAndComments(t.Content)
	sc := bufio.NewScanner(bytes.NewReader(masked))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
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
