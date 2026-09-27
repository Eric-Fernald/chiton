package security

import (
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
	for line, text := range analyze.FileLines(t) {
		if containsAny(text, "eval", "exec") && evalRe.MatchString(analyze.StripPyComment(text)) {
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

func init() { analyze.Register(EvalUsage{}) }
