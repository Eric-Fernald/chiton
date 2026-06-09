package quality

import (
	"bufio"
	"bytes"
	"context"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// PythonSemicolons flags a Python line that uses a semicolon to separate
// statements (`a = 1; b = 2`). Idiomatic Python puts one statement per line;
// statement-separating semicolons are a code smell. Trailing semicolons and
// semicolons inside strings/comments are ignored.
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
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		if separatingSemicolon(sc.Text()) {
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

// separatingSemicolon reports whether the Python source line contains a
// semicolon that separates two statements — i.e. a ';' outside of any string or
// comment that is followed by further non-whitespace, non-comment code.
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
