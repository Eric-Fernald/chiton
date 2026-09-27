package quality

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// TooManyParams flags a function/method declaration with more than MaxParams parameters; a long list
// usually signals a missing struct/options object.
type TooManyParams struct{}

// MaxParams is the threshold; lists with more than this many parameters fire.
const MaxParams = 5

func (TooManyParams) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "quality/too-many-params",
		Category:   analyze.CategoryQuality,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "rs", "java"},
		Summary:    fmt.Sprintf("Function declared with more than %d parameters.", MaxParams),
		Fix:        "Group related parameters into a struct/options object.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// funcDeclRe matches a keyword-led declaration up to its parameter list's open paren, so call expressions
// never count. It allows a Go receiver and `[...]`/`<...>` type parameters before the list.
var funcDeclRe = regexp.MustCompile(
	`(?:\bfunc\b|\bdef\b|\bfn\b|\bfunction\b)\s*` + // go/py/rust/js keyword form
		`(?:\([^)]*\)\s*)?` + // optional Go method receiver
		`[A-Za-z_][A-Za-z0-9_]*` + // function name (required for this form)
		`\s*(?:\[[^\]]*\]|<[^>]*>)?` + // optional generic/type params ([..] or <..>)
		`\s*\(`)

// methodDeclRe matches a Java/TS-style method declaration: a modifier keyword, then `name(`; requiring the
// modifier keeps arbitrary `foo(` call sites out.
var methodDeclRe = regexp.MustCompile(
	`(?:\bpublic\b|\bprivate\b|\bprotected\b|\bstatic\b|\bfinal\b|\babstract\b)` +
		`[^;={}()]*\b[A-Za-z_][A-Za-z0-9_]*\s*\(`)

func (TooManyParams) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	isPy := analyze.FileExt(t.Path) == "py"
	for line, raw := range analyze.FileLines(t) {
		if strings.IndexByte(raw, '(') < 0 {
			continue
		}
		text := analyze.StripLineComment(raw, true)
		loc := funcDeclRe.FindStringIndex(text)
		if loc == nil {
			loc = methodDeclRe.FindStringIndex(text)
		}
		if loc == nil {
			continue
		}
		// Both regexes end at the parameter list's opening paren.
		params, ok := paramList(text[loc[1]-1:])
		if !ok {
			// Not closed on this line: skip rather than guess across continuations (keeps FP low).
			continue
		}
		n := countParams(params)
		// Python's implicit receiver (`self`/`cls`) is not a parameter from the caller's perspective.
		if isPy && n > 0 && hasLeadingSelf(params) {
			n--
		}
		if n <= MaxParams {
			continue
		}
		out = append(out, analyze.Finding{
			RuleID:       "quality/too-many-params",
			Severity:     analyze.SeverityLow,
			Category:     analyze.CategoryQuality,
			Message:      fmt.Sprintf("function declared with %d parameters (limit %d)", n, MaxParams),
			Fix:          "Group related parameters into a struct/options object.",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		})
	}
	return out
}

// hasLeadingSelf reports whether the first parameter in a Python parameter list
// is the implicit receiver `self` or `cls` (optionally type-annotated/defaulted).
func hasLeadingSelf(params string) bool {
	first, _, _ := strings.Cut(params, ",")
	// Strip any annotation/default so `self: Foo` / `cls = X` still match.
	first, _, _ = strings.Cut(first, ":")
	first, _, _ = strings.Cut(first, "=")
	first = strings.TrimSpace(first)
	return first == "self" || first == "cls"
}

// paramList returns the text between the '(' that s starts with and its matching ')'; ok is false when the
// list is not balanced on this line (e.g. a multi-line signature).
func paramList(s string) (string, bool) {
	if len(s) == 0 || s[0] != '(' {
		return "", false
	}
	depth := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '(', '[', '{':
			depth++
		case ')', ']', '}':
			depth--
			if depth == 0 {
				return s[1:i], true
			}
		}
	}
	return "", false
}

// countParams counts top-level comma-separated parameters, skipping commas nested in brackets or generics;
// an empty or blank list counts as zero.
func countParams(s string) int {
	if strings.TrimSpace(s) == "" {
		return 0
	}
	depth := 0
	count := 1
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '(', '[', '{', '<':
			depth++
		case ')', ']', '}', '>':
			if depth > 0 {
				depth--
			}
		case ',':
			if depth == 0 {
				count++
			}
		}
	}
	return count
}

func init() { analyze.Register(TooManyParams{}) }
