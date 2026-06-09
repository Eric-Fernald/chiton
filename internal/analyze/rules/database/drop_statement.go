package database

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DropStatement flags DROP TABLE / DROP DATABASE / DROP SCHEMA in source. These
// are destructive, irreversible DDL statements; they belong in reviewed,
// versioned migrations, not embedded in application code.
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
// "drop" or method names like dropFirst(). IF EXISTS is allowed between.
var dropStmtRe = regexp.MustCompile(`(?i)\bDROP\s+(?:TABLE|DATABASE|SCHEMA)\b`)

func (DropStatement) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(maskDBBlocks(t.Content, t.URI)))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripDBComment(sc.Text(), t.URI)
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
