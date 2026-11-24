package database

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DeleteUpdateNoWhere flags a DELETE or UPDATE statement that starts and ends on one line with no WHERE.
// It skips statements that continue onto the next line.
type DeleteUpdateNoWhere struct{}

func (DeleteUpdateNoWhere) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "database/delete-update-no-where",
		Category:   analyze.CategoryDatabase,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "java"},
		Summary:    "DELETE/UPDATE with no WHERE clause — mutates every row in the table.",
		Fix:        "Add a WHERE clause, or confirm a full-table mutation is intended.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var (
	// DELETE FROM <table>  ... terminated on the same line with no WHERE.
	deleteNoWhereRe = regexp.MustCompile(`(?i)\bDELETE\s+FROM\s+[\w."` + "`" + `]+\s*[)"';]`)
	// UPDATE <table> SET ... terminated on the same line with no WHERE.
	updateNoWhereRe = regexp.MustCompile(`(?i)\bUPDATE\s+[\w."` + "`" + `]+\s+SET\s+.+[)"';]`)
	whereClauseRe   = regexp.MustCompile(`(?i)\bWHERE\b`)
)

func (DeleteUpdateNoWhere) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	py := analyze.FileExt(t.URI) == "py"
	for line, raw := range analyze.LinesOf(maskedDBText(t, py)) {
		text := stripDBComment(raw, py)
		// A WHERE anywhere on the line means the statement is bounded.
		if whereClauseRe.MatchString(text) {
			continue
		}
		// A trailing concatenation or continuation token means the WHERE may be on a later line.
		if continuesToNextLine(text) {
			continue
		}
		if deleteNoWhereRe.MatchString(text) || updateNoWhereRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "database/delete-update-no-where",
				Severity:     analyze.SeverityHigh,
				Category:     analyze.CategoryDatabase,
				Message:      "DELETE/UPDATE without WHERE mutates the whole table",
				Fix:          "Add a WHERE clause to scope the mutation.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

// continuesToNextLine reports whether s ends (ignoring trailing blanks) with `+`, `||`, `,`, `.`,
// or `\`, joining it to the next line.
func continuesToNextLine(s string) bool {
	s = strings.TrimRight(s, " \t")
	if s == "" {
		return false
	}
	for _, suf := range []string{"+", "||", ",", ".", "\\"} {
		if strings.HasSuffix(s, suf) {
			return true
		}
	}
	return false
}

func init() { analyze.Register(DeleteUpdateNoWhere{}) }
