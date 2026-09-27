package security

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// YAMLUnsafeLoad flags PyYAML's yaml.load() unless the line names a safe loader: the default
// loader constructs arbitrary Python objects, so untrusted YAML is an RCE vector.
type YAMLUnsafeLoad struct{}

func (YAMLUnsafeLoad) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/yaml-unsafe-load",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.FileScope,
		Languages:  []string{"py"},
		Summary:    "yaml.load() without a safe loader can execute arbitrary objects.",
		Fix:        "Use yaml.safe_load() or pass Loader=yaml.SafeLoader.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var (
	yamlLoadRe = regexp.MustCompile(`\byaml\.load\s*\(`)
	yamlSafeRe = regexp.MustCompile(`(?i)SafeLoader|CSafeLoader|safe_load`)
)

func (YAMLUnsafeLoad) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, text := range analyze.Lines(t.Content) {
		if !strings.Contains(text, "yaml.load") {
			continue
		}
		text = analyze.StripPyComment(text)
		if !yamlLoadRe.MatchString(text) {
			continue
		}
		if yamlSafeRe.MatchString(text) {
			continue // explicit safe loader on the same line
		}
		out = append(out, analyze.Finding{
			RuleID:       "security/yaml-unsafe-load",
			Severity:     analyze.SeverityHigh,
			Category:     analyze.CategorySecurity,
			Message:      "yaml.load() without a safe loader",
			Fix:          "Use yaml.safe_load() or Loader=yaml.SafeLoader.",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		})
	}
	return out
}

func init() { analyze.Register(YAMLUnsafeLoad{}) }
