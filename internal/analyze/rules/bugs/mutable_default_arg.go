package bugs

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// MutableDefaultArg flags Python `def foo(x=[])` / `def foo(x={})` patterns —
// the canonical Python gotcha where the default value is shared across calls.
type MutableDefaultArg struct{}

func (MutableDefaultArg) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "bugs/mutable-default-arg",
		Category:   analyze.CategoryBugs,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.FileScope,
		Languages:  []string{"py"},
		Summary:    "Python function default argument is a mutable literal ([], {}, set()).",
		Fix:        "Use None as the default and create the mutable inside the function body.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var mutableDefaultRe = regexp.MustCompile(`^\s*(async\s+)?def\s+\w+\s*\([^)]*?=\s*(\[\s*\]|\{\s*\}|set\s*\(\s*\))`)

func (MutableDefaultArg) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		if mutableDefaultRe.MatchString(sc.Text()) {
			out = append(out, analyze.Finding{
				RuleID:       "bugs/mutable-default-arg",
				Severity:     analyze.SeverityHigh,
				Category:     analyze.CategoryBugs,
				Message:      "mutable default argument is shared across calls",
				Fix:          "Use None as the default and build the container inside the function.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(MutableDefaultArg{}) }
