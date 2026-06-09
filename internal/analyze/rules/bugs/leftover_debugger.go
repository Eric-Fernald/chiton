package bugs

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// LeftoverDebugger flags debugger breakpoints accidentally left in source:
// `debugger;` in JS/TS, and `pdb.set_trace()` / `breakpoint()` in Python.
// These halt execution and must never reach production.
type LeftoverDebugger struct{}

func (LeftoverDebugger) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "bugs/leftover-debugger",
		Category:   analyze.CategoryBugs,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"js", "ts", "jsx", "tsx", "py"},
		Summary:    "Debugger breakpoint left in source (debugger; / pdb.set_trace() / breakpoint()).",
		Fix:        "Remove the breakpoint before committing.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var (
	jsDebuggerRe = regexp.MustCompile(`(^|[^.\w])debugger\s*;?\s*$`)
	pyPdbRe      = regexp.MustCompile(`(^|[^.\w])(pdb\.set_trace\s*\(\s*\)|breakpoint\s*\(\s*\))`)
)

func (LeftoverDebugger) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		// Strip both // (js) and # (py) line comments; harmless to apply both.
		text := stripPyComment(stripJSComment(sc.Text()))
		if jsDebuggerRe.MatchString(text) || pyPdbRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "bugs/leftover-debugger",
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryBugs,
				Message:      "debugger breakpoint left in source",
				Fix:          "Remove the breakpoint before committing.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(LeftoverDebugger{}) }
