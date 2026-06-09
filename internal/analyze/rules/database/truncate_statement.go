package database

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// TruncateStatement flags TRUNCATE TABLE in source. TRUNCATE empties a table
// instantly and (on most engines) cannot be rolled back the way a DELETE can;
// it belongs in a controlled migration or admin task, not application code.
type TruncateStatement struct{}

func (TruncateStatement) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "database/truncate-statement",
		Category:   analyze.CategoryDatabase,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "java"},
		Summary:    "TRUNCATE TABLE in source — empties a table irreversibly.",
		Fix:        "Move TRUNCATE into a reviewed migration/admin task, or use a scoped DELETE.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// Require the TABLE keyword to avoid matching unrelated uses of "truncate"
// (string truncation helpers, file truncation, etc.).
var truncateStmtRe = regexp.MustCompile(`(?i)\bTRUNCATE\s+TABLE\b`)

func (TruncateStatement) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(maskDBBlocks(t.Content, t.URI)))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripDBComment(sc.Text(), t.URI)
		if truncateStmtRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "database/truncate-statement",
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryDatabase,
				Message:      "TRUNCATE TABLE empties the table irreversibly",
				Fix:          "Move TRUNCATE into a reviewed migration, or use a scoped DELETE.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(TruncateStatement{}) }
