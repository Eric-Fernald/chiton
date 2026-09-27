package performance

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// IndexOfNotIncludes flags the JS/TS `.indexOf(x) !== -1` membership idiom; only "is present"
// comparisons match, so `.indexOf(x) > 5` and uses of the returned index are left alone.
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

// `.indexOf(...)` followed by `!== -1`, `!= -1`, `>= 0`, or `> -1`; the argument may contain one
// level of nested parens (`indexOf(fn(x))`).
var indexOfMembershipRe = regexp.MustCompile(
	`\.indexOf\s*\([^()]*(?:\([^()]*\)[^()]*)*\)\s*(?:!==?\s*-\s*1|>=\s*0|>\s*-\s*1)`,
)

func (IndexOfNotIncludes) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	const ruleID = "performance/indexof-not-includes"
	var out []analyze.Finding
	for line, raw := range analyze.FileLines(t) {
		stripped := analyze.StripLineComment(raw, true)
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
