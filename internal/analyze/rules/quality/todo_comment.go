package quality

import (
	"context"
	"regexp"
	"strings"

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
	for line, raw := range analyze.FileLines(t) {
		text := blankStringLiterals(raw)
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

// blankStringLiterals spaces out string-literal contents so markers inside strings don't match. It stops at a
// `//` or `#` comment so its text stays matchable and prose apostrophes ("don't") open no string.
func blankStringLiterals(s string) string {
	if strings.IndexAny(s, "\"'`") < 0 {
		return s
	}
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
