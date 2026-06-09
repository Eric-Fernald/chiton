package performance

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// RangeLen flags the Python `for i in range(len(x))` idiom, which is almost
// always better written with `enumerate(x)` (or direct iteration). Only the
// single-argument `range(len(...))` form is matched so genuine index-arithmetic
// loops like `range(len(x) - 1)` are left alone.
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

// Matches `for <var> in range(len(<expr>))` where the range() call has exactly
// one argument (the len(...) call), i.e. the close-paren of len is immediately
// followed by the close-paren of range, with no extra args.
var rangeLenRe = regexp.MustCompile(`\bfor\s+\w+\s+in\s+range\s*\(\s*len\s*\([^()]*\)\s*\)`)

func (RangeLen) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	const ruleID = "performance/range-len"
	var out []analyze.Finding
	// Mask Python string literals (including multi-line """/''' docstrings) and
	// # comments across the whole file before scanning, so a `range(len(x))`
	// mentioned inside a docstring or comment is not flagged.
	masked := maskPyStrings(t.Content)
	sc := bufio.NewScanner(bytes.NewReader(masked))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		stripped := sc.Text()
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
