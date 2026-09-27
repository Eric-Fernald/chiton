package quality

import (
	"context"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// NestedTernary flags a nested ternary expression (`a ? b ? c : d : e`) on one line of JS/TS source,
// since nested ternaries are hard to read. Sibling ternaries on the same line do not fire.
type NestedTernary struct{}

func (NestedTernary) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "quality/nested-ternary",
		Category:   analyze.CategoryQuality,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"js", "ts", "jsx", "tsx"},
		Summary:    "Nested ternary expression — hard to read.",
		Fix:        "Refactor into if/else or extract intermediate variables.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (NestedTernary) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, raw := range analyze.FileLines(t) {
		if hasNestedTernary(raw) {
			out = append(out, analyze.Finding{
				RuleID:       "quality/nested-ternary",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryQuality,
				Message:      "nested ternary expression — hard to read",
				Fix:          "Refactor into if/else or extract intermediate variables.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

// hasNestedTernary reports whether a second ternary `?` opens before the first one's `:`, ignoring strings,
// `//` comments, `?.`, and `??`. A `;` or top-level `,` resets the depth so siblings start fresh.
func hasNestedTernary(raw string) bool {
	var inStr byte
	depth := 0   // currently-open ternaries (each `?` not yet closed by a `:`)
	bracket := 0 // (), [], {} nesting, to know what "top level" means for `,`
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if inStr != 0 {
			if c == '\\' && i+1 < len(raw) {
				i++
				continue
			}
			if c == inStr {
				inStr = 0
			}
			continue
		}
		switch c {
		case '"', '\'', '`':
			inStr = c
		case '/':
			if i+1 < len(raw) && raw[i+1] == '/' {
				return false // rest is a line comment
			}
		case '(', '[', '{':
			bracket++
		case ')', ']', '}':
			if bracket > 0 {
				bracket--
			}
		case ';':
			// A statement boundary ends any ternary; siblings start fresh.
			depth = 0
		case ',':
			// Only a top-level comma separates sibling expressions. A comma inside brackets keeps the depth.
			if bracket == 0 {
				depth = 0
			}
		case '?':
			// Skip `??` (nullish coalescing) and `?.` (optional chaining).
			if i+1 < len(raw) && (raw[i+1] == '?' || raw[i+1] == '.') {
				i++ // consume the second char of the operator
				continue
			}
			// Skip the second '?' of a `??` we did not pre-consume.
			if i > 0 && raw[i-1] == '?' {
				continue
			}
			depth++
			if depth >= 2 {
				return true // a second ternary opened before the first closed
			}
		case ':':
			if depth > 0 {
				depth--
			}
		}
	}
	return false
}

func init() { analyze.Register(NestedTernary{}) }
