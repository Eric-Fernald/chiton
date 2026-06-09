package quality

import (
	"context"
	"fmt"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
)

// HighComplexity flags a function whose approximate cyclomatic complexity
// exceeds MaxComplexity. Complexity is estimated as 1 plus the number of
// branch-introducing constructs in the body (if/for/while/case/catch) and
// short-circuit operators (&&, ||). Function boundaries come from the
// extractor's symbol spans — the same source LongFunction relies on.
type HighComplexity struct{}

// MaxComplexity is the threshold; functions above it fire.
const MaxComplexity = 15

func (HighComplexity) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "quality/high-complexity",
		Category:   analyze.CategoryQuality,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "rs", "java"},
		Summary:    fmt.Sprintf("Function with cyclomatic complexity over %d.", MaxComplexity),
		Fix:        "Decompose the function; extract branches into helpers.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// branchKeywordRe matches branch-introducing keywords as whole words. `case`
// covers switch/match arms across languages; `catch`/`except` cover error arms.
var branchKeywordRe = regexp.MustCompile(`\b(?:if|elif|for|while|case|catch|except)\b`)

// logicalOpRe matches short-circuit boolean operators, which each add a path.
var logicalOpRe = regexp.MustCompile(`&&|\|\|`)

func (HighComplexity) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for _, fb := range qFuncBodiesMasked(t) {
		c := complexity(fb.lines)
		if c <= MaxComplexity {
			continue
		}
		out = append(out, analyze.Finding{
			RuleID:       "quality/high-complexity",
			Severity:     analyze.SeverityMedium,
			Category:     analyze.CategoryQuality,
			Message:      fmt.Sprintf("function %q has cyclomatic complexity ~%d (limit %d)", fb.node.Label, c, MaxComplexity),
			Fix:          "Decompose the function; extract branches into helpers.",
			TargetURI:    t.URI,
			TargetNodeID: fb.node.ID,
			Span:         fb.node.Span,
		})
	}
	return out
}

// complexity estimates cyclomatic complexity from body source lines: a base of
// 1, plus each branch keyword and each short-circuit operator. Lines arrive
// pre-masked (qFuncBodiesMasked), so keywords/operators inside strings,
// comments, and Python docstrings are filler bytes and are not counted.
func complexity(lines []string) int {
	c := 1
	for _, ln := range lines {
		c += len(branchKeywordRe.FindAllString(ln, -1))
		c += len(logicalOpRe.FindAllString(ln, -1))
	}
	return c
}

func init() { analyze.Register(HighComplexity{}) }
