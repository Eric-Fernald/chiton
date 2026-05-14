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
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
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

func init() { analyze.Register(TodoComment{}) }
