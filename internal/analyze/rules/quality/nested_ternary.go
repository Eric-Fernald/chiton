package quality

import (
	"bufio"
	"bytes"
	"context"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// NestedTernary flags a nested ternary expression (`a ? b ? c : d : e`) in
// JS/TS sources. Nested ternaries are notoriously hard to read. Detection is
// conservative: it counts ternary `?` operators on a line (ignoring `?.`, `??`,
// strings, template literals, and comments) and fires only when two or more
// genuine ternary conditions and at least two `:` branch separators appear,
// which signals nesting rather than a single ternary.
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
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		if hasNestedTernary(sc.Text()) {
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

// hasNestedTernary reports whether a line contains an ACTUALLY nested ternary:
// a second ternary `?` opening before the matching `:` of the first one
// (`c1 ? c2 ? a : b : d`). It skips strings, template literals, comments, the
// optional-chaining `?.` operator, and the nullish-coalescing `??` operator.
//
// Independent sibling ternaries on one physical line (`a?b:c; d?e:f` or
// `f(a?b:c, d?e:f)`) are NOT nested: the running ternary depth is reset at a
// top-level `;` and at a top-level `,`, so each sibling starts fresh and never
// reaches depth 2.
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
			// A top-level comma separates sibling expressions (e.g. call
			// arguments at this bracket level); reset so siblings are not
			// treated as nested. A comma inside deeper brackets keeps depth.
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
