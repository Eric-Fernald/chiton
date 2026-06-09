package style

import (
	"bufio"
	"bytes"
	"context"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// MaxConsecutiveBlankLines is the number of consecutive blank lines that
// triggers a finding once exceeded. Two blanks are common and idiomatic
// (e.g. between Go top-level decls); three or more are almost always accidental.
const MaxConsecutiveBlankLines = 2

// MultipleBlankLines flags runs of 3 or more consecutive blank (empty or
// whitespace-only) lines. Reports one finding per run, anchored at the first
// excess blank line, to stay low-noise.
type MultipleBlankLines struct{}

func (MultipleBlankLines) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "style/multiple-blank-lines",
		Category:   analyze.CategoryStyle,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "rs", "java"},
		Summary:    "Three or more consecutive blank lines.",
		Fix:        "Collapse the run down to a single blank line.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (MultipleBlankLines) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	blankRun := 0
	for sc.Scan() {
		line++
		if strings.TrimSpace(sc.Text()) == "" {
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
