package quality

import (
	"context"
	"fmt"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
)

// TooManyReturns flags a function with more than MaxReturns return statements. Many exit points make it
// harder to follow and to reason about resource cleanup.
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

// returnStmtRe matches `return` as a whole word. Masking keeps comments and strings from counting.
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
func countReturns(lines [][]byte) int {
	n := 0
	for _, ln := range lines {
		n += len(returnStmtRe.FindAllIndex(ln, -1))
	}
	return n
}

func init() { analyze.Register(TooManyReturns{}) }
