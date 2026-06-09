package performance

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// JSONDeepClone flags the `JSON.parse(JSON.stringify(x))` deep-clone idiom in
// JS/TS. It is slow, drops functions/undefined/Dates/Maps, and throws on cyclic
// data; structuredClone (or a purpose-built clone) is the modern replacement.
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
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		stripped := stripLineComment(sc.Text())
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
