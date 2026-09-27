// Package bugs contains deterministic rules for the bugs category.
package bugs

import (
	"bytes"
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DeferInLoop flags Go `defer` statements inside a for/range loop: defers stack until function exit,
// which leaks resources (file handles, locks) in long-running loops.
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
	if !bytes.Contains(t.Content, []byte("defer")) {
		return nil
	}
	var out []analyze.Finding
	var loopDepths []int // brace depth at each open for-line's opening brace
	depth := 0

	for line, text := range analyze.Lines(t.Content) {
		stripped := stripGoComment(text)

		// Open new loop frame if this line begins a for-statement.
		if forLineRe.MatchString(stripped) {
			open := strings.Count(stripped, "{")
			depth += open - strings.Count(stripped, "}")
			if open > 0 {
				loopDepths = append(loopDepths, depth)
			}
			continue
		}

		// Track depth without re-processing for-lines we already handled.
		depth += strings.Count(stripped, "{") - strings.Count(stripped, "}")

		// Pop loop frames whose depth has been exited.
		for len(loopDepths) > 0 && depth < loopDepths[len(loopDepths)-1] {
			loopDepths = loopDepths[:len(loopDepths)-1]
		}

		if len(loopDepths) == 0 {
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

// stripGoComment truncates a // line comment and blanks string-literal contents, so braces and keywords
// inside strings never match. Block comments are not tracked.
func stripGoComment(s string) string {
	if strings.IndexAny(s, "\"'`") < 0 {
		if i := strings.Index(s, "//"); i >= 0 {
			return s[:i]
		}
		return s
	}
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
