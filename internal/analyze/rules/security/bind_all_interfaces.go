package security

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// BindAllInterfaces flags a service bound to 0.0.0.0 (every interface). Low severity because it is
// often intentional in containers, but it is also a common accidental exposure.
type BindAllInterfaces struct{}

func (BindAllInterfaces) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/bind-all-interfaces",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "java"},
		Summary:    "Service bound to 0.0.0.0 (all network interfaces).",
		Fix:        "Bind to a specific interface (e.g. 127.0.0.1) unless external exposure is intended.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// bindAllRe matches 0.0.0.0[:port] only as a whole quoted string, so version strings and comments
// that merely mention the address don't count.
var bindAllRe = regexp.MustCompile(`["']0\.0\.0\.0(:\d+)?["']`)

func (BindAllInterfaces) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	if isTestFile(t.Path) {
		return nil
	}
	var out []analyze.Finding
	for line, text := range analyze.Lines(t.Content) {
		if !strings.Contains(text, "0.0.0.0") {
			continue
		}
		text = stripSlashCommentKeepURL(analyze.StripPyComment(text))
		if bindAllRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "security/bind-all-interfaces",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategorySecurity,
				Message:      "service bound to 0.0.0.0 (all interfaces)",
				Fix:          "Bind to a specific interface unless external exposure is intended.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(BindAllInterfaces{}) }
