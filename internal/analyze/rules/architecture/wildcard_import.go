package architecture

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// WildcardImport flags Python `from X import *` wildcard imports. They pollute
// the namespace, obscure where names come from, and defeat static analysis.
type WildcardImport struct{}

func (WildcardImport) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "architecture/wildcard-import",
		Category:   analyze.CategoryArchitecture,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"py"},
		Summary:    "Wildcard import (`from X import *`) pollutes the namespace.",
		Fix:        "Import only the names you use explicitly.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// from <module> import *  — module may be dotted or relative (leading dots).
var pyWildcardRe = regexp.MustCompile(`^from\s+[.\w]+\s+import\s+\*\s*$`)

func (WildcardImport) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, raw := range analyze.FileLines(t) {
		text := strings.TrimSpace(stripArchLineComment("py", raw))
		if pyWildcardRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "architecture/wildcard-import",
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryArchitecture,
				Message:      "wildcard import pollutes the namespace",
				Fix:          "Import only the names you use explicitly.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(WildcardImport{}) }
