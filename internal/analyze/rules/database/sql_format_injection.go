package database

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// SQLFormatInjection flags SQL built by formatting instead of binding: Go fmt.Sprintf, Python
// f-strings and %-format, JS/TS template literals; py/js need a real substitution to fire.
type SQLFormatInjection struct{}

func (SQLFormatInjection) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "database/sql-format-injection",
		Category:   analyze.CategoryDatabase,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts"},
		Summary:    "SQL built via string formatting/interpolation — use parameterized queries.",
		Fix:        "Use bound parameters (placeholders) instead of formatting values into SQL.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// SQL skeleton (SELECT…FROM, UPDATE…SET, INSERT INTO, DELETE FROM) so a string merely containing
// "select"/"update" doesn't fire; the SELECT/UPDATE gap cannot cross a quote or backtick.
const sqlGap = `[^"'` + "`" + `]*?`
const sqlVerb = `(?:SELECT\b` + sqlGap + `\bFROM|INSERT\s+INTO|UPDATE\b` + sqlGap + `\bSET|DELETE\s+FROM)`

var (
	// go: fmt.Sprintf("... SELECT ...", ...)
	goSprintfSQLRe = regexp.MustCompile(`(?i)\bfmt\.Sprintf\s*\(\s*["` + "`" + `][^"` + "`" + `]*\b` + sqlVerb + `\b`)
	// py: f-string with a SQL verb and an interpolation: f"... SELECT ... {x}"
	pyFStringSQLRe  = regexp.MustCompile(`(?i)\bf"[^"]*\b` + sqlVerb + `\b[^"]*\{[^}]+\}`)
	pyFString2SQLRe = regexp.MustCompile(`(?i)\bf'[^']*\b` + sqlVerb + `\b[^']*\{[^}]+\}`)
	// py: "...SELECT..." % (...)   percent-format
	pyPercentSQLRe = regexp.MustCompile(`(?i)["'][^"']*\b` + sqlVerb + `\b[^"']*["']\s*%\s*[\w(]`)
	// js/ts: template literal containing SQL and a ${...} interpolation.
	jsTemplateSQLRe = regexp.MustCompile(`(?i)` + "`" + `[^` + "`" + `]*\b` + sqlVerb + `\b[^` + "`" + `]*\$\{[^}]+\}`)
)

func (SQLFormatInjection) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	ext := analyze.FileExt(t.URI)
	var res []*regexp.Regexp
	switch ext {
	case "go":
		res = []*regexp.Regexp{goSprintfSQLRe}
	case "py":
		res = []*regexp.Regexp{pyFStringSQLRe, pyFString2SQLRe, pyPercentSQLRe}
	case "js", "ts":
		res = []*regexp.Regexp{jsTemplateSQLRe}
	default:
		return nil
	}

	var out []analyze.Finding
	py := ext == "py"
	for line, raw := range analyze.Lines(maskedDB(t, py)) {
		text := stripDBComment(raw, py)
		for _, re := range res {
			if re.MatchString(text) {
				out = append(out, analyze.Finding{
					RuleID:       "database/sql-format-injection",
					Severity:     analyze.SeverityHigh,
					Category:     analyze.CategoryDatabase,
					Message:      "SQL assembled via string formatting/interpolation",
					Fix:          "Use parameterized queries with bound placeholders.",
					TargetURI:    t.URI,
					TargetNodeID: t.FileNode.ID,
					Span:         &model.SourceSpan{StartLine: line, EndLine: line},
				})
				break
			}
		}
	}
	return out
}

func init() { analyze.Register(SQLFormatInjection{}) }
