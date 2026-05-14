package security

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// EvalUsage flags Python eval()/exec() calls. Their use in production code
// is almost always a code-injection waiting room.
type EvalUsage struct{}

func (EvalUsage) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/eval-usage",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.FileScope,
		Languages:  []string{"py"},
		Summary:    "Use of eval() / exec() allows arbitrary code execution.",
		Fix:        "Replace with ast.literal_eval, a dispatch table, or json.loads.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// Match eval( / exec( as a standalone call (not as part of a longer identifier
// like myeval()).
var evalRe = regexp.MustCompile(`(^|[^a-zA-Z0-9_])(eval|exec)\s*\(`)

func (EvalUsage) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
		trimmed := stripPyComment(text)
		if evalRe.MatchString(trimmed) {
			out = append(out, analyze.Finding{
				RuleID:       "security/eval-usage",
				Severity:     analyze.SeverityHigh,
				Category:     analyze.CategorySecurity,
				Message:      "eval()/exec() permits arbitrary code execution",
				Fix:          "Use ast.literal_eval or a dispatch table instead.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

// stripPyComment removes everything from the first '#' that isn't inside a
// quoted string. Keeps the rule from firing on lines that only mention eval()
// in a comment.
func stripPyComment(s string) string {
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
		case '"', '\'':
			inStr = c
		case '#':
			return s[:i]
		}
	}
	return s
}

func init() { analyze.Register(EvalUsage{}) }
