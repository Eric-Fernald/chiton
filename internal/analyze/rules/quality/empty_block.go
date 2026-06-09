package quality

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// EmptyBlock flags an empty control-flow block: `if/for/while/switch (...) {}`
// with nothing between the braces. An empty branch usually means dead code, a
// forgotten body, or an inverted condition. Function bodies, struct/interface
// literals, and `{}` object/map literals are excluded because the rule only
// matches when a control-flow keyword and a parenthesised condition precede the
// braces — and requires the `{}` to be empty on a single line.
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

// emptyBlockRe matches a control-flow keyword, then a header/condition (a
// parenthesised condition in C-family languages, or Go's parenless header),
// then an empty `{}` (allowing inner whitespace) at end of line. Anchoring
// `{}` to end-of-line keeps it from matching `if (x) {} else {...}` chains and
// from matching object/map literals (which are not preceded by a control-flow
// keyword). Requiring the keyword excludes empty function bodies and struct
// literals. `switch` is included; its empty body is just as suspicious. The
// header is restricted to non-brace characters so the match cannot span a
// nested block.
var emptyBlockRe = regexp.MustCompile(`\b(if|for|while|switch)\b[^{}]*\{\s*\}\s*$`)

func (EmptyBlock) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		code := qStripComment(sc.Text())
		m := emptyBlockRe.FindStringSubmatch(code)
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
