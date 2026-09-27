package quality

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// EmptyBlock flags a single-line empty control-flow block (`if/for/while/switch ... {}`), which usually means
// dead code, a forgotten body, or an inverted condition.
type EmptyBlock struct{}

func (EmptyBlock) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "quality/empty-block",
		Category:   analyze.CategoryQuality,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "js", "ts", "jsx", "tsx", "java"},
		Summary:    "Empty control-flow block (e.g. `if (...) {}`).",
		Fix:        "Remove the empty block or fill in the intended body.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// The keyword requirement excludes function bodies and struct/object literals; the end-of-line `{}` skips
// `if (x) {} else {...}` chains, and the brace-free header keeps a match from spanning a nested block.
var emptyBlockRe = regexp.MustCompile(`\b(if|for|while|switch)\b[^{}]*\{\s*\}\s*$`)

func (EmptyBlock) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, raw := range analyze.Lines(t.Content) {
		if strings.IndexByte(raw, '}') < 0 {
			continue
		}
		m := emptyBlockRe.FindStringSubmatch(analyze.StripLineComment(raw, true))
		if m == nil {
			continue
		}
		out = append(out, analyze.Finding{
			RuleID:       "quality/empty-block",
			Severity:     analyze.SeverityLow,
			Category:     analyze.CategoryQuality,
			Message:      "empty " + m[1] + " block",
			Fix:          "Remove the empty block or implement its body.",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		})
	}
	return out
}

func init() { analyze.Register(EmptyBlock{}) }
