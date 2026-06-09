package performance

import (
	"bufio"
	"bytes"
	"context"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// RegexCompileInLoop flags a regular expression that is compiled inside a loop
// body. Compilation is expensive and the result is almost always invariant
// across iterations, so it should be hoisted out of the loop (or, in Go, to a
// package-level var). Covers Go (regexp.Compile/MustCompile), Python
// (re.compile), JS/TS (new RegExp(...)) and Java (Pattern.compile).
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
	// `\bre\.compile` requires a word boundary before the standalone `re`
	// module so identifiers ending in `re` (core.compile, genre.compile) do
	// not match.
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
	return scanLoopBody(t.Content, isPython(t.Path), regexCompileRe, finder)
}

// isPython reports whether the file is a Python source file by extension.
func isPython(path string) bool {
	return strings.EqualFold(filepath.Ext(path), ".py")
}

// scanLoopBody scans content line by line and emits a finding (built by mk) for
// every line inside a loop body whose stripped text matches hit. For Python it
// uses indentation to determine the loop body; otherwise it uses brace depth.
func scanLoopBody(content []byte, python bool, hit *regexp.Regexp, mk func(line int) analyze.Finding) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0

	if python {
		// Stack of indentation levels at which a loop header opened.
		var loopIndents []int
		for sc.Scan() {
			line++
			raw := sc.Text()
			stripped := stripLineComment(raw)
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
	for sc.Scan() {
		line++
		stripped := stripLineComment(sc.Text())
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
	// A C-style loop header contains the for or while keyword not preceded by a
	// word character or a dot, so a header sharing a line with a preceding
	// token (`if (cond) { for (...) {`) is still recognized while a method or
	// field named `for`/`while` (`foo.for`, `x_for`) is not. The line may or
	// may not include the opening brace; brace counting in braceLoopScanner
	// determines the body extent. `do { }` while loops are intentionally
	// skipped (rare, low value).
	braceLoopHeaderRe = regexp.MustCompile(`(?:^|[^.\w])(?:for|while)\b`)
)

func init() { analyze.Register(RegexCompileInLoop{}) }
