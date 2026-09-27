package performance

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// JSONDeepClone flags the JS/TS `JSON.parse(JSON.stringify(x))` deep clone, which is slow, drops
// functions/undefined/Dates/Maps, and throws on cyclic data.
type JSONDeepClone struct{}

func (JSONDeepClone) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "performance/json-deep-clone",
		Category:   analyze.CategoryPerformance,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"js", "ts", "jsx", "tsx"},
		Summary:    "`JSON.parse(JSON.stringify(x))` deep clone is slow and lossy.",
		Fix:        "Use structuredClone(x) or a dedicated deep-clone utility.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var jsonDeepCloneRe = regexp.MustCompile(`JSON\.parse\s*\(\s*JSON\.stringify\s*\(`)

func (JSONDeepClone) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	const ruleID = "performance/json-deep-clone"
	var out []analyze.Finding
	for line, raw := range analyze.FileLines(t) {
		stripped := analyze.StripLineComment(raw, true)
		if jsonDeepCloneRe.MatchString(stripped) {
			out = append(out, analyze.Finding{
				RuleID:       ruleID,
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryPerformance,
				Message:      "JSON.parse(JSON.stringify(x)) deep clone is slow and lossy",
				Fix:          "Use structuredClone(x) or a dedicated deep-clone utility.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(JSONDeepClone{}) }
