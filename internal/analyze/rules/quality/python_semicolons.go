package quality

import (
	"context"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// PythonSemicolons flags a Python line that separates statements with a semicolon (`a = 1; b = 2`).
// Trailing semicolons and semicolons inside strings/comments are ignored.
type PythonSemicolons struct{}

func (PythonSemicolons) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "quality/python-semicolons",
		Category:   analyze.CategoryQuality,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"py"},
		Summary:    "Semicolon separating statements on one line (non-idiomatic Python).",
		Fix:        "Put each statement on its own line.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (PythonSemicolons) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, raw := range analyze.FileLines(t) {
		if separatingSemicolon(raw) {
			out = append(out, analyze.Finding{
				RuleID:       "quality/python-semicolons",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryQuality,
				Message:      "semicolon separates statements — use one statement per line",
				Fix:          "Split the statements onto separate lines.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

// separatingSemicolon reports whether a Python line has a ';' outside strings and comments that is
// followed by more code.
func separatingSemicolon(raw string) bool {
	var inStr byte
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if inStr != 0 {
			if c == '\\' && i+1 < len(raw) {
				i++
				continue
			}
			if c == inStr {
				inStr = 0
			}
			continue
		}
		switch c {
		case '"', '\'':
			inStr = c
		case '#':
			return false // rest of line is a comment
		case ';':
			rest := strings.TrimSpace(raw[i+1:])
			// A trailing ';' (optionally followed only by a comment) is not a
			// statement separator.
			if rest == "" || strings.HasPrefix(rest, "#") {
				return false
			}
			return true
		}
	}
	return false
}

func init() { analyze.Register(PythonSemicolons{}) }
