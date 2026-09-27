package database

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DropStatement flags DROP TABLE/DATABASE/SCHEMA in source: irreversible DDL that belongs in
// reviewed migrations, not application code.
type DropStatement struct{}

func (DropStatement) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "database/drop-statement",
		Category:   analyze.CategoryDatabase,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "java"},
		Summary:    "DROP TABLE/DATABASE/SCHEMA in source — destructive DDL.",
		Fix:        "Move destructive DDL into a reviewed migration, not application code.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// Require an object keyword after DROP so we don't match the English word
// "drop" or method names like dropFirst(); `DROP TABLE IF EXISTS` still matches.
var dropStmtRe = regexp.MustCompile(`(?i)\bDROP\s+(?:TABLE|DATABASE|SCHEMA)\b`)

func (DropStatement) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	py := analyze.FileExt(t.URI) == "py"
	for line, raw := range analyze.LinesOf(maskedDBText(t, py)) {
		text := stripDBComment(raw, py)
		if dropStmtRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "database/drop-statement",
				Severity:     analyze.SeverityHigh,
				Category:     analyze.CategoryDatabase,
				Message:      "destructive DROP statement in source",
				Fix:          "Move DROP into a reviewed, versioned migration.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(DropStatement{}) }
