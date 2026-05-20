package security

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// SQLStringConcat flags lines where a SQL keyword is concatenated with a
// variable — a near-canonical SQL-injection shape regardless of language.
type SQLStringConcat struct{}

func (SQLStringConcat) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/sql-string-concat",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.FileScope,
		Summary:    "SQL keyword built by string concatenation with a variable (SQL injection risk).",
		Fix:        "Use parameterised queries / prepared statements instead of string building.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// Matches a SQL keyword followed by string-concat punctuation that injects a
// variable name (Go-style `+ x`, Python f-strings, JS template literals).
var sqlConcatRe = regexp.MustCompile(
	`(?i)("[^"]*\b(SELECT|INSERT|UPDATE|DELETE|WHERE|FROM)\b[^"]*"\s*\+\s*\w)|` +
		`(f["'][^"']*\b(SELECT|INSERT|UPDATE|DELETE|WHERE|FROM)\b[^"']*\{[^}]+\})|` +
		"(`[^`]*\\$\\{[^}]+\\}[^`]*\\b(SELECT|INSERT|UPDATE|DELETE|WHERE|FROM)\\b[^`]*`)",
)

func (SQLStringConcat) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		if sqlConcatRe.MatchString(sc.Text()) {
			out = append(out, analyze.Finding{
				RuleID:       "security/sql-string-concat",
				Severity:     analyze.SeverityHigh,
				Category:     analyze.CategorySecurity,
				Message:      "SQL keyword interpolated with a variable — use parameterised queries",
				Fix:          "Switch to bound parameters (e.g. `?` / `$1`) instead of string concatenation.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(SQLStringConcat{}) }
