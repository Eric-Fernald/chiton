package bugs

import (
	"bytes"
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// ConsoleLog flags console.log statements in JavaScript/TypeScript source.
// Test files (`.test.` / `.spec.` in the path) are skipped.
type ConsoleLog struct{}

func (ConsoleLog) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "bugs/console-log",
		Category:   analyze.CategoryBugs,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"js", "ts", "jsx", "tsx"},
		Summary:    "console.log left in source.",
		Fix:        "Remove the call or replace with a real logger.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var consoleRe = regexp.MustCompile(`(^|[^a-zA-Z0-9_])console\.log\s*\(`)

func (ConsoleLog) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	low := strings.ToLower(t.Path)
	if strings.Contains(low, ".test.") || strings.Contains(low, ".spec.") {
		return nil
	}
	if !bytes.Contains(t.Content, []byte("console.log")) {
		return nil
	}
	var out []analyze.Finding
	for line, raw := range analyze.Lines(t.Content) {
		if consoleRe.MatchString(analyze.StripLineComment(raw, false)) {
			out = append(out, analyze.Finding{
				RuleID:       "bugs/console-log",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryBugs,
				Message:      "console.log call left in source",
				Fix:          "Remove or replace with a project logger.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(ConsoleLog{}) }
