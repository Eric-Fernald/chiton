// Package bugs contains deterministic rules for the bugs category.
package bugs

import (
	"bufio"
	"bytes"
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DeferInLoop flags `defer` statements that appear inside a for/range loop in
// Go. defers stack until function exit, which leaks resources when used in
// long-running loops (file handles, mutexes, etc.).
type DeferInLoop struct{}

func (DeferInLoop) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "bugs/defer-in-loop",
		Category:   analyze.CategoryBugs,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go"},
		Summary:    "defer inside a loop accumulates until function return; resources leak.",
		Fix:        "Move the deferred call into a helper func, or close explicitly inside the loop.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var (
	forLineRe   = regexp.MustCompile(`^\s*for\b.*\{?\s*$`)
	deferLineRe = regexp.MustCompile(`(^|\s)defer\s`)
)

func (DeferInLoop) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	type loopFrame struct {
		startLine int
		depth     int // brace depth at the for-line's opening brace
	}
	var stack []loopFrame
	depth := 0
	line := 0

	for sc.Scan() {
		line++
		text := sc.Text()
		stripped := stripGoComment(text)

		// Open new loop frame if this line begins a for-statement.
		if forLineRe.MatchString(stripped) {
			open := strings.Count(stripped, "{")
			depth += open - strings.Count(stripped, "}")
			if open > 0 {
				stack = append(stack, loopFrame{startLine: line, depth: depth})
			}
			continue
		}

		// Track depth without re-processing for-lines we already handled.
		depth += strings.Count(stripped, "{") - strings.Count(stripped, "}")

		// Pop loop frames whose depth has been exited.
		for len(stack) > 0 && depth < stack[len(stack)-1].depth {
			stack = stack[:len(stack)-1]
		}

		if len(stack) == 0 {
			continue
		}
		if deferLineRe.MatchString(stripped) {
			out = append(out, analyze.Finding{
				RuleID:       "bugs/defer-in-loop",
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryBugs,
				Message:      "defer inside loop accumulates until function exit",
				Fix:          "Wrap the body in a func() { defer ... }() helper.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

// stripGoComment is a small approximation of countLineBraces — drops the
// content of // line comments and blanks the content of string literals so
// keywords (and braces) inside strings never match. Block comments are not
// tracked across lines, matching the existing extractor's behaviour.
func stripGoComment(s string) string {
	b := []byte(s)
	var inStr byte
	for i := 0; i < len(b); i++ {
		c := b[i]
		if inStr != 0 {
			if c == '\\' && i+1 < len(b) {
				b[i], b[i+1] = ' ', ' '
				i++
				continue
			}
			if c == inStr {
				inStr = 0
				continue
			}
			b[i] = ' '
			continue
		}
		switch c {
		case '"', '\'', '`':
			inStr = c
		case '/':
			if i+1 < len(b) && b[i+1] == '/' {
				return string(b[:i])
			}
		}
	}
	return string(b)
}

func init() { analyze.Register(DeferInLoop{}) }
