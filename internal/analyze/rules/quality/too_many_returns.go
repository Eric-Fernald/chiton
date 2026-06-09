package quality

import (
	"context"
	"fmt"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
)

// TooManyReturns flags a function with more than MaxReturns return statements.
// Many exit points make a function harder to follow and to reason about for
// resource cleanup. Function boundaries come from the extractor's symbol spans,
// shared with the long-function and high-complexity rules.
type TooManyReturns struct{}

// MaxReturns is the threshold; functions with more return statements fire.
const MaxReturns = 6

func (TooManyReturns) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "quality/too-many-returns",
		Category:   analyze.CategoryQuality,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "rs", "java"},
		Summary:    fmt.Sprintf("Function with more than %d return statements.", MaxReturns),
		Fix:        "Consolidate exit points or split the function.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// returnStmtRe matches a return statement as a whole word. It is applied to
// pre-masked body lines, so `return` inside a comment or string is filler and
// is not counted.
var returnStmtRe = regexp.MustCompile(`\breturn\b`)

func (TooManyReturns) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for _, fb := range qFuncBodiesMasked(t) {
		n := countReturns(fb.lines)
		if n <= MaxReturns {
			continue
		}
		out = append(out, analyze.Finding{
			RuleID:       "quality/too-many-returns",
			Severity:     analyze.SeverityLow,
			Category:     analyze.CategoryQuality,
			Message:      fmt.Sprintf("function %q has %d return statements (limit %d)", fb.node.Label, n, MaxReturns),
			Fix:          "Consolidate exit points or split the function.",
			TargetURI:    t.URI,
			TargetNodeID: fb.node.ID,
			Span:         fb.node.Span,
		})
	}
	return out
}

// countReturns counts return statements across the body's (pre-masked) source
// lines.
func countReturns(lines []string) int {
	n := 0
	for _, ln := range lines {
		n += len(returnStmtRe.FindAllString(ln, -1))
	}
	return n
}

func init() { analyze.Register(TooManyReturns{}) }
