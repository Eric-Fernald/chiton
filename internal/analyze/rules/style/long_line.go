package style

import (
	"context"
	"unicode/utf8"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// LongLineLimit is the maximum line length (in runes) before a finding fires.
const LongLineLimit = 120

// LongLine flags source lines longer than LongLineLimit characters. The limit is generous, so ordinary
// long lines pass.
type LongLine struct{}

func (LongLine) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "style/long-line",
		Category:   analyze.CategoryStyle,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  sourceLangs,
		Summary:    "Line exceeds 120 characters.",
		Fix:        "Wrap or refactor the line so it fits within 120 columns.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (LongLine) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, text := range analyze.FileLines(t) {
		// Count runes, not bytes, so multi-byte characters aren't over-counted.
		if len(text) <= LongLineLimit || utf8.RuneCountInString(text) <= LongLineLimit {
			continue
		}
		out = append(out, analyze.Finding{
			RuleID:       "style/long-line",
			Severity:     analyze.SeverityLow,
			Category:     analyze.CategoryStyle,
			Message:      "line exceeds 120 characters",
			Fix:          "Wrap or refactor the line so it fits within 120 columns.",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		})
	}
	return out
}

func init() { analyze.Register(LongLine{}) }
