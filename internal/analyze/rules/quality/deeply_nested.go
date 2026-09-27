package quality

import (
	"context"
	"fmt"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DeeplyNested flags lines indented deeper than MaxIndent levels (a tab or four spaces per level), once per
// contiguous deep run. Deep indentation correlates with high cognitive complexity.
type DeeplyNested struct{}

// MaxIndent is the threshold: 5 levels = `func / for / if / for / if / line`.
const MaxIndent = 5

func (DeeplyNested) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "quality/deeply-nested",
		Category:   analyze.CategoryQuality,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Summary:    fmt.Sprintf("Code indented deeper than %d levels — high cognitive complexity.", MaxIndent),
		Fix:        "Extract inner blocks into helper functions, or invert conditions with early returns.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (DeeplyNested) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	lastDeep := -1 // most recent deep line, to emit once per contiguous deep block
	for line, text := range analyze.FileLines(t) {
		if strings.TrimSpace(text) == "" {
			continue
		}
		level := indentLevel(text)
		if level <= MaxIndent {
			continue
		}
		prevDeep := lastDeep == line-1
		lastDeep = line
		if prevDeep {
			continue
		}
		out = append(out, analyze.Finding{
			RuleID:       "quality/deeply-nested",
			Severity:     analyze.SeverityLow,
			Category:     analyze.CategoryQuality,
			Message:      fmt.Sprintf("indented to level %d (threshold %d)", level, MaxIndent),
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		})
	}
	return out
}

// indentLevel maps the leading whitespace of s into chiton's indent units:
// one tab = 1 level, four spaces = 1 level.
func indentLevel(s string) int {
	tabs, spaces := 0, 0
	for _, c := range s {
		switch c {
		case '\t':
			tabs++
		case ' ':
			spaces++
		default:
			return tabs + spaces/4
		}
	}
	return tabs + spaces/4
}

func init() { analyze.Register(DeeplyNested{}) }
