package database

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// SelectStar flags `SELECT * FROM` in a SQL string literal. Selecting every
// column couples callers to the schema, ships unused bytes over the wire, and
// breaks when columns are added or reordered. Distinct from
// select-star-unbounded (which is about missing WHERE/LIMIT): this fires on any
// SELECT *, including bounded queries, at a lower severity.
type SelectStar struct{}

func (SelectStar) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "database/select-star",
		Category:   analyze.CategoryDatabase,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "java"},
		Summary:    "SELECT * in a SQL string — couples code to schema and fetches unused columns.",
		Fix:        "List the columns you actually need instead of `*`.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// Requires SELECT * FROM so we only match real queries, not e.g. `count(*)`
// or a multiplication. The FROM keyword anchors it to a table read.
var selectStarLiteralRe = regexp.MustCompile(`(?i)\bSELECT\s+\*\s+FROM\s`)

func (SelectStar) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(maskDBBlocks(t.Content, t.URI)))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripDBComment(sc.Text(), t.URI)
		if selectStarLiteralRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "database/select-star",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryDatabase,
				Message:      "SELECT * — name the columns you need",
				Fix:          "Replace `*` with an explicit column list.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(SelectStar{}) }
