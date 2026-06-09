package bugs

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// LooseEquality flags JS/TS use of `==` / `!=` instead of the strict
// `===` / `!==`. Loose equality triggers type coercion (e.g. 0 == "",
// null == undefined) and is a frequent source of subtle bugs.
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

// Match `==` or `!=` that is NOT part of `===`, `!==`, `==>`, `>=`, `<=`,
// or an assignment-equality boundary. The surrounding negative classes keep
// `===`/`!==` (extra `=`) from matching, and exclude `=>` arrow boundaries.
var looseEqualityRe = regexp.MustCompile(`[^=!<>]([!=]=)([^=]|$)`)

func (LooseEquality) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	// Mask string, template, and regex literal bodies plus comments across the
	// whole file first, so `==`/`!=` inside a SQL string, a base64 blob, a
	// template, or a regex literal (/.../) can never reach the operator regex.
	masked := maskJSLiteralsAndComments(t.Content)
	sc := bufio.NewScanner(bytes.NewReader(masked))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
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
