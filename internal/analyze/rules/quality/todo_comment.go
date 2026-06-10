package quality

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// TodoComment flags TODO/FIXME/XXX/HACK comments — useful when running
// chiton over a maturing codebase to triage debt hotspots.
type TodoComment struct{}

func (TodoComment) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "quality/todo-comment",
		Category:   analyze.CategoryQuality,
		Severity:   analyze.SeverityInfo,
		Scope:      analyze.FileScope,
		Summary:    "TODO/FIXME/XXX/HACK comment marker.",
		Fix:        "Either resolve the marker or replace with a tracked issue link.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var todoRe = regexp.MustCompile(`(?i)\b(TODO|FIXME|XXX|HACK)\b`)

func (TodoComment) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	// Test fixtures often embed marker text in multi-line raw strings, which
	// per-line blanking can't see — skip test sources entirely.
	if analyze.IsTestSource(t.URI) {
		return nil
	}
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := blankStringLiterals(sc.Text())
		if m := todoRe.FindString(text); m != "" {
			out = append(out, analyze.Finding{
				RuleID:       "quality/todo-comment",
				Severity:     analyze.SeverityInfo,
				Category:     analyze.CategoryQuality,
				Message:      m + " marker",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

// blankStringLiterals overwrites string-literal contents with spaces so
// todoRe only matches markers in comments and code — without this the rule
// flags its own pattern strings. Scanning stops at a line-comment start
// (`//` or `#`): comment text must stay matchable, and apostrophes in prose
// ("don't") must not be mistaken for string openers.
func blankStringLiterals(s string) string {
	b := []byte(s)
	var inStr byte
	for i := 0; i < len(b); i++ {
		c := b[i]
		if inStr != 0 {
			if c == '\\' && i+1 < len(b) {
				b[i], b[i+1] = ' ', ' '
				i++
				continue
			}
			if c == inStr {
				inStr = 0
				continue
			}
			b[i] = ' '
			continue
		}
		switch c {
		case '#':
			return string(b)
		case '/':
			if i+1 < len(b) && b[i+1] == '/' {
				return string(b)
			}
		case '"', '\'', '`':
			inStr = c
		}
	}
	return string(b)
}

func init() { analyze.Register(TodoComment{}) }
