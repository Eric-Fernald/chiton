package performance

import (
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// RangeLen flags Python `for i in range(len(x))`; only the single-argument form matches, so
// index-arithmetic loops like `range(len(x) - 1)` are left alone.
type RangeLen struct{}

func (RangeLen) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "performance/range-len",
		Category:   analyze.CategoryPerformance,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"py"},
		Summary:    "`for i in range(len(x))`; prefer `enumerate(x)` or direct iteration.",
		Fix:        "Use `for i, item in enumerate(x):` or iterate over x directly.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// `for <var> in range(len(<expr>))` where len(...) is range's only argument.
var rangeLenRe = regexp.MustCompile(`\bfor\s+\w+\s+in\s+range\s*\(\s*len\s*\([^()]*\)\s*\)`)

func (RangeLen) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	const ruleID = "performance/range-len"
	// Exact prefilter: masking writes only 'x', so a masked `range` is a raw one too.
	if !bytes.Contains(t.Content, []byte("range")) {
		return nil
	}
	var out []analyze.Finding
	// Mask strings (multi-line docstrings included) and # comments file-wide before scanning.
	for line, stripped := range analyze.Lines(maskPyStrings(t.Content)) {
		if rangeLenRe.MatchString(stripped) {
			out = append(out, analyze.Finding{
				RuleID:       ruleID,
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryPerformance,
				Message:      "for i in range(len(x)); use enumerate(x) instead",
				Fix:          "Use `for i, item in enumerate(x):`.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(RangeLen{}) }
