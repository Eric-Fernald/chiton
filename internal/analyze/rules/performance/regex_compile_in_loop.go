package performance

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// RegexCompileInLoop flags a regex compiled inside a loop body (Go regexp.Compile/MustCompile,
// Python re.compile, JS/TS new RegExp, Java Pattern.compile), where the result is loop-invariant.
type RegexCompileInLoop struct{}

func (RegexCompileInLoop) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "performance/regex-compile-in-loop",
		Category:   analyze.CategoryPerformance,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "java"},
		Summary:    "Regex compiled inside a loop recompiles every iteration.",
		Fix:        "Compile the pattern once outside the loop and reuse it.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var regexCompileRe = regexp.MustCompile(
	// `\bre\.compile` keeps identifiers ending in `re` (core.compile, genre.compile) from matching.
	`(?:regexp\.(?:MustCompile|Compile)|\bre\.compile|new\s+RegExp|Pattern\.compile)\s*\(`,
)

func (RegexCompileInLoop) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	const ruleID = "performance/regex-compile-in-loop"
	finder := func(line int) analyze.Finding {
		return analyze.Finding{
			RuleID:       ruleID,
			Severity:     analyze.SeverityMedium,
			Category:     analyze.CategoryPerformance,
			Message:      "regex compiled inside loop body recompiles each iteration",
			Fix:          "Hoist the compiled pattern out of the loop.",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		}
	}
	return scanLoopBody(t.Content, analyze.FileExt(t.Path) == "py", regexCompileRe, finder)
}

// scanLoopBody emits mk(line) for each line inside a loop body whose comment-stripped text matches
// hit; loop bodies are found by indentation for Python and by brace depth otherwise.
func scanLoopBody(content []byte, python bool, hit *regexp.Regexp, mk func(line int) analyze.Finding) []analyze.Finding {
	var out []analyze.Finding

	if python {
		// Stack of indentation levels at which a loop header opened.
		var loopIndents []int
		for line, raw := range analyze.Lines(content) {
			stripped := analyze.StripLineComment(raw, true)
			if strings.TrimSpace(stripped) == "" {
				continue
			}
			indent := leadingSpaces(raw)
			// Exit any loop bodies we've dedented out of.
			for len(loopIndents) > 0 && indent <= loopIndents[len(loopIndents)-1] {
				loopIndents = loopIndents[:len(loopIndents)-1]
			}
			inLoop := len(loopIndents) > 0
			if inLoop && hit.MatchString(stripped) {
				out = append(out, mk(line))
			}
			if pyLoopHeaderRe.MatchString(stripped) {
				loopIndents = append(loopIndents, indent)
			}
		}
		return out
	}

	var b braceLoopScanner
	for line, raw := range analyze.Lines(content) {
		stripped := analyze.StripLineComment(raw, true)
		isLoop := braceLoopHeaderRe.MatchString(stripped)
		inside := b.feed(stripped, isLoop)
		if inside && hit.MatchString(stripped) {
			out = append(out, mk(line))
		}
	}
	return out
}

var (
	pyLoopHeaderRe = regexp.MustCompile(`^\s*(?:for\b|while\b).*:\s*$`)
	// for/while not preceded by a word char or dot: matches mid-line headers (`if (c) { for (...) {`)
	// but not members like `foo.for`; braceLoopScanner finds the body extent.
	braceLoopHeaderRe = regexp.MustCompile(`(?:^|[^.\w])(?:for|while)\b`)
)

func init() { analyze.Register(RegexCompileInLoop{}) }
