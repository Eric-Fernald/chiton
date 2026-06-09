package style

import (
	"bufio"
	"bytes"
	"context"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// MixedIndent flags lines whose leading whitespace mixes tabs and spaces.
// One of the easiest ways to ship invisible diffs and confuse Python.
type MixedIndent struct{}

func (MixedIndent) Meta() analyze.Meta {
	return analyze.Meta{
		ID:       "style/mixed-indent",
		Category: analyze.CategoryStyle,
		Severity: analyze.SeverityLow,
		Scope:    analyze.FileScope,
		// Restrict to source languages: Markdown and image/binary files should
		// not be scanned for indentation style.
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "rs", "java"},
		Summary:    "Leading whitespace mixes tabs and spaces.",
		Fix:        "Pick one (Go: tabs, Python/JS: spaces) and let the formatter normalise.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (MixedIndent) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
		lead := leadingWhitespace(text)
		if strings.ContainsRune(lead, '\t') && strings.ContainsRune(lead, ' ') {
			out = append(out, analyze.Finding{
				RuleID:       "style/mixed-indent",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryStyle,
				Message:      "leading whitespace mixes tabs and spaces",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func leadingWhitespace(s string) string {
	for i, r := range s {
		if r != ' ' && r != '\t' {
			return s[:i]
		}
	}
	return s
}

func init() { analyze.Register(MixedIndent{}) }
