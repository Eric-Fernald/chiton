package quality

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DeeplyNested flags any function that contains a line indented deeper than
// MaxIndent levels (treating one tab or four spaces as a level). High
// indentation correlates with high cognitive complexity.
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
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	emitted := map[int]bool{} // line of the deepest run, to dedupe blocks
	for sc.Scan() {
		line++
		text := sc.Text()
		if strings.TrimSpace(text) == "" {
			continue
		}
		level := indentLevel(text)
		if level <= MaxIndent {
			continue
		}
		// Emit at most one finding per contiguous deep block.
		if emitted[line-1] {
			emitted[line] = true
			continue
		}
		emitted[line] = true
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
