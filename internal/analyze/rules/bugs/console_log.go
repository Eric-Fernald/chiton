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

// ConsoleLog flags console.log statements in JavaScript/TypeScript source.
// Test files and explicit logger wrappers are skipped.
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
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripJSComment(sc.Text())
		if consoleRe.MatchString(text) {
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

func stripJSComment(s string) string {
	var inStr byte
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inStr != 0 {
			if c == '\\' && i+1 < len(s) {
				i++
				continue
			}
			if c == inStr {
				inStr = 0
			}
			continue
		}
		switch c {
		case '"', '\'', '`':
			inStr = c
		case '/':
			if i+1 < len(s) && s[i+1] == '/' {
				return s[:i]
			}
		}
	}
	return s
}

func init() { analyze.Register(ConsoleLog{}) }
