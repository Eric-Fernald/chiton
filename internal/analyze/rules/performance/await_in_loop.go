package performance

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// AwaitInLoop flags an `await` expression used directly inside a for/while loop
// body. Sequential awaits in a loop run one request at a time; when the
// iterations are independent they should be parallelized with Promise.all.
// The legal `for await (... of ...)` async-iteration header is NOT flagged.
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
	// An await EXPRESSION: the `await` keyword (not preceded by a word char or
	// dot, so member names ending in `await` are excluded) followed by an
	// operand — an identifier/$, or an open paren/bracket/brace. Requiring an
	// operand distinguishes real `await x` / `await fn(...)` from the bare
	// prose word "await" (e.g. in a sentence) that masking can't reach when it
	// is genuine code text rather than a literal.
	awaitExprRe = regexp.MustCompile(`(?:^|[^.\w])await\s+[\w$([{]`)
	forAwaitRe  = regexp.MustCompile(`^\s*for\s+await\b`)
)

func (AwaitInLoop) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	const ruleID = "performance/await-in-loop"
	var out []analyze.Finding
	// Mask string and comment contents (including multi-line block comments)
	// across the whole file before scanning, so an `await` mentioned in a log
	// string or a /* ... */ block comment inside a loop is not flagged.
	masked := maskPerfLiteralsComments(t.Content)
	sc := bufio.NewScanner(bytes.NewReader(masked))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	var b braceLoopScanner
	for sc.Scan() {
		line++
		stripped := sc.Text()
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
