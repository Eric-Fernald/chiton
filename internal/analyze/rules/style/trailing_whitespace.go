// Package style contains deterministic rules in the style category.
package style

import (
	"context"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// sourceLangs restricts style rules to source code: Markdown uses two trailing spaces as a hard
// line break, and image/binary files have no meaningful "lines". Shared; never mutate.
var sourceLangs = []string{"go", "py", "js", "ts", "jsx", "tsx", "rs", "java"}

// TrailingWhitespace flags trailing spaces/tabs at end-of-line, surfacing editors that aren't
// configured to strip on save.
type TrailingWhitespace struct{}

func (TrailingWhitespace) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "style/trailing-whitespace",
		Category:   analyze.CategoryStyle,
		Severity:   analyze.SeverityInfo,
		Scope:      analyze.FileScope,
		Languages:  sourceLangs,
		Summary:    "Line ends with trailing whitespace.",
		Fix:        "Configure your editor to strip trailing whitespace on save.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (TrailingWhitespace) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, text := range analyze.Lines(t.Content) {
		if text == "" {
			continue
		}
		if last := text[len(text)-1]; last != ' ' && last != '\t' {
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
