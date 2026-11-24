// Package database contains deterministic rules in the database category. They fire only on
// unambiguous patterns to keep false positives low.
package database

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// SelectStarFromUnbounded flags SELECT * without WHERE or LIMIT inline.
// A common cause of accidental full-table scans in dev that ship to prod.
type SelectStarFromUnbounded struct{}

func (SelectStarFromUnbounded) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "database/select-star-unbounded",
		Category:   analyze.CategoryDatabase,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "java"},
		Summary:    "SELECT * FROM <table> without a WHERE or LIMIT — risks accidental full scans.",
		Fix:        "Add WHERE/LIMIT, or select explicit columns to make intent explicit.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var selectStarRe = regexp.MustCompile(`(?i)SELECT\s+\*\s+FROM\s+[\w."` + "`" + `]+(?:\s*[)"';,]|\s*$)`)

func (SelectStarFromUnbounded) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	py := analyze.FileExt(t.URI) == "py"
	for line, raw := range analyze.LinesOf(maskedDBText(t, py)) {
		text := stripDBComment(raw, py)
		if !selectStarRe.MatchString(text) {
			continue
		}
		// Skip if the same line also bounds the query.
		if hasBoundingClause(text) {
			continue
		}
		out = append(out, analyze.Finding{
			RuleID:       "database/select-star-unbounded",
			Severity:     analyze.SeverityMedium,
			Category:     analyze.CategoryDatabase,
			Message:      "SELECT * without WHERE/LIMIT",
			Fix:          "Add WHERE/LIMIT, or name the columns you actually need.",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		})
	}
	return out
}

var boundsRe = regexp.MustCompile(`(?i)\b(WHERE|LIMIT|JOIN)\b`)

func hasBoundingClause(s string) bool { return boundsRe.MatchString(s) }

func init() { analyze.Register(SelectStarFromUnbounded{}) }
