package style

import (
	"context"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// MaxConsecutiveBlankLines is the longest allowed run of blank lines: two are idiomatic (e.g.
// between Go top-level decls); three or more are almost always accidental.
const MaxConsecutiveBlankLines = 2

// MultipleBlankLines flags runs of 3+ blank (empty or whitespace-only) lines, once per run at
// the first excess line, to stay low-noise.
type MultipleBlankLines struct{}

func (MultipleBlankLines) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "style/multiple-blank-lines",
		Category:   analyze.CategoryStyle,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  sourceLangs,
		Summary:    "Three or more consecutive blank lines.",
		Fix:        "Collapse the run down to a single blank line.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (MultipleBlankLines) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	blankRun := 0
	for line, text := range analyze.Lines(t.Content) {
		if strings.TrimSpace(text) == "" {
			blankRun++
			// Fire exactly once per run, at the first line beyond the limit.
			if blankRun == MaxConsecutiveBlankLines+1 {
				out = append(out, analyze.Finding{
					RuleID:       "style/multiple-blank-lines",
					Severity:     analyze.SeverityLow,
					Category:     analyze.CategoryStyle,
					Message:      "3 or more consecutive blank lines",
					Fix:          "Collapse the run down to a single blank line.",
					TargetURI:    t.URI,
					TargetNodeID: t.FileNode.ID,
					Span:         &model.SourceSpan{StartLine: line, EndLine: line},
				})
			}
			continue
		}
		blankRun = 0
	}
	return out
}

func init() { analyze.Register(MultipleBlankLines{}) }
