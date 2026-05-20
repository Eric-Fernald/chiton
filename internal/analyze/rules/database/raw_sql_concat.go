// Package database contains deterministic rules in the database category.
//
// These rules are deliberately conservative: they only fire on patterns that
// are unambiguous (e.g. SELECT/INSERT keywords adjacent to interpolation).
package database

import (
	"bufio"
	"bytes"
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
		Summary:    "SELECT * FROM <table> without a WHERE or LIMIT — risks accidental full scans.",
		Fix:        "Add WHERE/LIMIT, or select explicit columns to make intent explicit.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var selectStarRe = regexp.MustCompile(`(?i)SELECT\s+\*\s+FROM\s+[\w."` + "`" + `]+(?:\s*[)"';,]|\s*$)`)

func (SelectStarFromUnbounded) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
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
