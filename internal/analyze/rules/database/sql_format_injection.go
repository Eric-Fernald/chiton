package database

import (
	"bufio"
	"bytes"
	"context"
	"path"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// SQLFormatInjection flags SQL that is assembled with string formatting or
// interpolation rather than parameter binding — the classic SQL-injection
// vector. Per language:
//   - go: fmt.Sprintf(...) whose format string contains a SQL verb.
//   - py: an f-string ( f"...SELECT..." ) or %-format ( "...SELECT..." % ... )
//     containing a SQL verb.
//   - js/ts: a template literal ( `...SELECT...${x}...` ) with an interpolation
//     embedded in SQL.
//
// The interpolation/format marker must be present, so a plain constant SQL
// string with no substitution does not fire.
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

// SQL skeleton that anchors a string as a query rather than arbitrary text.
// Crucially, the bare words SELECT and UPDATE are NOT enough on their own — an
// interpolated log/template string that merely contains "select" or "update"
// must not fire. We require the structural follow-on keyword:
//   - SELECT ... FROM
//   - UPDATE ... SET
//   - INSERT INTO   (verb alone is already specific)
//   - DELETE FROM   (verb alone is already specific)
//
// The gap between SELECT/UPDATE and its follow-on keyword may not cross a string
// delimiter (", ', or `), so it stays within a single literal.
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
	ext := strings.ToLower(strings.TrimPrefix(path.Ext(t.URI), "."))
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
	sc := bufio.NewScanner(bytes.NewReader(maskDBBlocks(t.Content, t.URI)))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripDBComment(sc.Text(), t.URI)
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
