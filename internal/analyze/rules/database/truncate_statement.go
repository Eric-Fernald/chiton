package database

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// TruncateStatement flags TRUNCATE TABLE in source: it empties a table and, on most engines,
// cannot be rolled back like a DELETE.
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
	py := analyze.FileExt(t.URI) == "py"
	for line, raw := range analyze.Lines(maskedDB(t, py)) {
		text := stripDBComment(raw, py)
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
