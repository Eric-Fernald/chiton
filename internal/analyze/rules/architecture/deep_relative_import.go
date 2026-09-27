package architecture

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DeepRelativeImport flags JS/TS import paths that climb three or more parent directories
// (`../../../`), a sign of a module reaching across layer boundaries.
type DeepRelativeImport struct{}

func (DeepRelativeImport) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "architecture/deep-relative-import",
		Category:   analyze.CategoryArchitecture,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"js", "ts", "jsx", "tsx"},
		Summary:    "Import path climbs 3+ parent directories (../../../).",
		Fix:        "Use a path alias or restructure so modules don't reach across layers.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// A quoted module specifier beginning with three or more `../` segments. Matches
// inside ES `import ... from "..."`, bare `import "..."`, and `require("...")`.
var deepRelativeRe = regexp.MustCompile(`['"](\.\./){3,}`)

func (DeepRelativeImport) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, raw := range analyze.FileLines(t) {
		// Only consider lines that are plausibly imports to avoid matching deep
		// relative paths inside string data unrelated to module resolution.
		text := stripArchLineComment("js", raw)
		if !jsImportRe.MatchString(strings.TrimLeft(text, " \t")) &&
			!jsRequireRe.MatchString(text) {
			continue
		}
		if deepRelativeRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "architecture/deep-relative-import",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryArchitecture,
				Message:      "import climbs 3+ parent directories",
				Fix:          "Use a path alias or restructure module boundaries.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(DeepRelativeImport{}) }
