package database

import (
	"bufio"
	"bytes"
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DeleteUpdateNoWhere flags DELETE/UPDATE statements that clearly have no WHERE
// clause — a mass mutation that wipes or rewrites an entire table. To stay
// low-false-positive the rule only fires when the *whole statement* is visible
// on one line and is terminated (closing quote/paren/semicolon) without a WHERE
// appearing anywhere on that line. A statement that continues onto the next
// line (no terminator) is skipped, since its WHERE may live there.
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
	sc := bufio.NewScanner(bytes.NewReader(maskDBBlocks(t.Content, t.URI)))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripDBComment(sc.Text(), t.URI)
		// A WHERE anywhere on the line means the statement is bounded.
		if whereClauseRe.MatchString(text) {
			continue
		}
		// A line that ends in a string-concatenation / continuation operator is
		// part of a multi-line statement whose WHERE may live on a later line —
		// skip it rather than risk a false positive.
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

// continuesToNextLine reports whether a (comment-stripped) line ends with a
// token that joins it to the next line — string concatenation (`+`, `||`, `,`)
// or a trailing dot/backslash. Such a statement is not "clearly" WHERE-less.
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
