// Package style contains deterministic rules in the style category.
package style

import (
	"context"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// sourceLangs limits style rules to source code, since Markdown uses two trailing spaces as a hard line
// break and image or binary files have no real lines.
var sourceLangs = analyze.SourceLangs

// TrailingWhitespace flags trailing spaces or tabs at the end of a line, a sign of an editor that isn't
// configured to strip them on save.
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
	for line, text := range analyze.FileLines(t) {
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
