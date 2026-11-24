package bugs

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// LeftoverDebugger flags breakpoints left in source: `debugger;` in JS/TS, and `pdb.set_trace()` or
// `breakpoint()` in Python. They halt execution and must never reach production.
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
	for line, raw := range analyze.FileLines(t) {
		// Strip both // (js) and # (py) line comments; harmless to apply both.
		text := analyze.StripPyComment(analyze.StripLineComment(raw, false))
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
