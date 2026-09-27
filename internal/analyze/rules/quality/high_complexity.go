package quality

import (
	"context"
	"fmt"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
)

// HighComplexity flags a function whose approximate cyclomatic complexity (1 + branch keywords + `&&`/`||`)
// exceeds MaxComplexity.
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

// complexity estimates cyclomatic complexity from pre-masked body lines: 1 plus each branch keyword and
// short-circuit operator.
func complexity(lines [][]byte) int {
	c := 1
	for _, ln := range lines {
		c += len(branchKeywordRe.FindAllIndex(ln, -1))
		c += len(logicalOpRe.FindAllIndex(ln, -1))
	}
	return c
}

func init() { analyze.Register(HighComplexity{}) }
