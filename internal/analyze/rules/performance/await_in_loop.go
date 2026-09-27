package performance

import (
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// AwaitInLoop flags an `await` expression inside a for/while loop body, which serializes the
// iterations; the `for await (... of ...)` header itself is not flagged.
type AwaitInLoop struct{}

func (AwaitInLoop) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "performance/await-in-loop",
		Category:   analyze.CategoryPerformance,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"js", "ts", "jsx", "tsx"},
		Summary:    "await inside a loop runs requests sequentially.",
		Fix:        "Collect promises and await Promise.all(...) when iterations are independent.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var (
	// `await` (not a member name) followed by an operand, so a bare prose "await" that survives
	// masking does not match.
	awaitExprRe = regexp.MustCompile(`(?:^|[^.\w])await\s+[\w$([{]`)
	forAwaitRe  = regexp.MustCompile(`^\s*for\s+await\b`)
)

func (AwaitInLoop) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	const ruleID = "performance/await-in-loop"
	// Exact prefilter: masking writes only 'x', so a masked `await` is a raw one too.
	if !bytes.Contains(t.Content, []byte("await")) {
		return nil
	}
	var out []analyze.Finding
	var b braceLoopScanner
	// Mask strings and comments file-wide (multi-line /* */ included) so an await inside them is ignored.
	for line, stripped := range analyze.Lines(analyze.MaskedCLiterals(t, false)) {
		isLoop := braceLoopHeaderRe.MatchString(stripped)
		inside := b.feed(stripped, isLoop)
		// Skip the `for await (... of ...)` async-iteration header itself.
		if forAwaitRe.MatchString(stripped) {
			continue
		}
		if inside && awaitExprRe.MatchString(stripped) {
			out = append(out, analyze.Finding{
				RuleID:       ruleID,
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryPerformance,
				Message:      "await inside loop body runs sequentially",
				Fix:          "Use Promise.all for independent iterations.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(AwaitInLoop{}) }
