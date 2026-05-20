// Package style contains deterministic rules in the style category.
package style

import (
	"bufio"
	"bytes"
	"context"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// TrailingWhitespace flags trailing spaces/tabs at end-of-line. The most
// pedestrian style finding, but cheap, deterministic, and surfaces editors
// that aren't configured to strip on save.
type TrailingWhitespace struct{}

func (TrailingWhitespace) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "style/trailing-whitespace",
		Category:   analyze.CategoryStyle,
		Severity:   analyze.SeverityInfo,
		Scope:      analyze.FileScope,
		Summary:    "Line ends with trailing whitespace.",
		Fix:        "Configure your editor to strip trailing whitespace on save.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (TrailingWhitespace) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
		if text == "" {
			continue
		}
		trimmed := strings.TrimRight(text, " \t")
		if len(trimmed) == len(text) {
			continue
		}
		out = append(out, analyze.Finding{
			RuleID:       "style/trailing-whitespace",
			Severity:     analyze.SeverityInfo,
			Category:     analyze.CategoryStyle,
			Message:      "trailing whitespace",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		})
	}
	return out
}

func init() { analyze.Register(TrailingWhitespace{}) }
