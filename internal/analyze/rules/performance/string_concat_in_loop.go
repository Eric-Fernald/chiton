// Package performance contains deterministic rules in the performance category.
package performance

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// StringConcatInLoop flags Go `s += "..."` or `s = s + ...` inside a `for` body, which
// accumulates in O(n²).
type StringConcatInLoop struct{}

func (StringConcatInLoop) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "performance/string-concat-in-loop",
		Category:   analyze.CategoryPerformance,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go"},
		Summary:    "String concatenation inside a loop allocates each iteration (O(n²)).",
		Fix:        "Use strings.Builder, bytes.Buffer, or pre-allocate a []byte.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var (
	stringPlusEqRe = regexp.MustCompile(`(^|\s)\w+\s*\+=\s*"`)
	forStartRe     = regexp.MustCompile(`^\s*for\b`)
	// Match `name = name + ...` with a varying name. Go's RE2 has no
	// backreferences, so we check the equality of capture groups manually.
	stringSelfPlusRe = regexp.MustCompile(`(^|\s)(\w+)\s*=\s*(\w+)\s*\+`)
)

func (StringConcatInLoop) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	depth := 0
	loopDepth := 0
	for line, text := range analyze.Lines(t.Content) {
		if forStartRe.MatchString(text) {
			loopDepth = depth + 1
		}
		depth += strings.Count(text, "{") - strings.Count(text, "}")
		if depth < loopDepth {
			loopDepth = 0
		}
		if loopDepth == 0 {
			continue
		}
		if stringPlusEqRe.MatchString(text) || isSelfPlus(text) {
			out = append(out, analyze.Finding{
				RuleID:       "performance/string-concat-in-loop",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryPerformance,
				Message:      "string accumulation inside loop is O(n²)",
				Fix:          "Use strings.Builder.WriteString instead.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

// isSelfPlus reports whether s contains a `name = name + ...` assignment
// — RE2 has no backreferences, so we capture both names and compare.
func isSelfPlus(s string) bool {
	m := stringSelfPlusRe.FindStringSubmatch(s)
	return len(m) == 4 && m[2] == m[3]
}

func init() { analyze.Register(StringConcatInLoop{}) }
