package performance

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// IndexOfNotIncludes flags the `.indexOf(x) !== -1` membership-test idiom in
// JS/TS, where `.includes(x)` is clearer and conveys intent directly. Only the
// comparison forms that mean "is present" are matched, so `.indexOf(x) > 5` and
// uses of the returned index are not touched.
type IndexOfNotIncludes struct{}

func (IndexOfNotIncludes) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "performance/indexof-not-includes",
		Category:   analyze.CategoryPerformance,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"js", "ts", "jsx", "tsx"},
		Summary:    "`.indexOf(x) !== -1` membership test; `.includes(x)` is clearer.",
		Fix:        "Replace with `.includes(x)`.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// Matches `.indexOf(...)` immediately followed by a present-membership
// comparison: `!== -1`, `!= -1`, `>= 0`, or `> -1`. The argument matcher allows
// one level of nested parentheses (e.g. `indexOf(fn(x))`) so a membership test
// on a function-call argument is not missed.
var indexOfMembershipRe = regexp.MustCompile(
	`\.indexOf\s*\([^()]*(?:\([^()]*\)[^()]*)*\)\s*(?:!==?\s*-\s*1|>=\s*0|>\s*-\s*1)`,
)

func (IndexOfNotIncludes) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	const ruleID = "performance/indexof-not-includes"
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		stripped := stripLineComment(sc.Text())
		if indexOfMembershipRe.MatchString(stripped) {
			out = append(out, analyze.Finding{
				RuleID:       ruleID,
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryPerformance,
				Message:      "indexOf(...) membership test; use .includes(...) instead",
				Fix:          "Replace `.indexOf(x) !== -1` with `.includes(x)`.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(IndexOfNotIncludes{}) }
