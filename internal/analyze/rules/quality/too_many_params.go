package quality

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// TooManyParams flags a function/method declaration with more than
// MaxParams parameters. A long parameter list usually signals a missing
// struct/options object and hurts call-site readability.
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

// funcDeclRe matches the start of a function/method declaration across the
// supported languages and captures the open paren that begins the parameter
// list. It is deliberately anchored to declaration keywords / patterns so that
// ordinary call expressions are not counted.
//
//   - An optional Go method receiver `(r T)` may appear between `func` and the
//     name, so receiver methods (`func (s *S) Do(...)`) are matched.
//   - Generic/type-parameter lists may be `[...]` (Go/Rust) OR `<...>` (TS/Rust),
//     so generic signatures are matched.
var funcDeclRe = regexp.MustCompile(
	`(?:\bfunc\b|\bdef\b|\bfn\b|\bfunction\b)\s*` + // go/py/rust/js keyword form
		`(?:\([^)]*\)\s*)?` + // optional Go method receiver
		`[A-Za-z_][A-Za-z0-9_]*` + // function name (required for this form)
		`\s*(?:\[[^\]]*\]|<[^>]*>)?` + // optional generic/type params ([..] or <..>)
		`\s*\(`)

// methodDeclRe matches a Java/TS-style method declaration: an access or
// modifier keyword, then `name(`. Requires a leading modifier so we don't
// match arbitrary `foo(` call sites.
var methodDeclRe = regexp.MustCompile(
	`(?:\bpublic\b|\bprivate\b|\bprotected\b|\bstatic\b|\bfinal\b|\babstract\b)` +
		`[^;={}()]*\b[A-Za-z_][A-Za-z0-9_]*\s*\(`)

func (TooManyParams) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := qStripComment(sc.Text())
		loc := funcDeclRe.FindStringIndex(text)
		if loc == nil {
			loc = methodDeclRe.FindStringIndex(text)
		}
		if loc == nil {
			continue
		}
		// Parameter list starts at the last '(' inside the matched declaration.
		open := strings.LastIndexByte(text[:loc[1]], '(')
		if open < 0 {
			continue
		}
		params, ok := paramList(text[open:])
		if !ok {
			// Parameter list not closed on this single line — skip to avoid
			// guessing across line continuations (keeps FP low).
			continue
		}
		n := countParams(params)
		// In Python the implicit receiver (`self`/`cls`) is not a real
		// parameter from the caller's perspective; do not count it.
		if isPyPath(t.Path) && n > 0 && hasLeadingSelf(params) {
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

// isPyPath reports whether path names a Python source file.
func isPyPath(path string) bool {
	return strings.EqualFold(filepath.Ext(path), ".py")
}

// hasLeadingSelf reports whether the first parameter in a Python parameter list
// is the implicit receiver `self` or `cls` (optionally type-annotated/defaulted).
func hasLeadingSelf(params string) bool {
	first := params
	if i := strings.IndexByte(first, ','); i >= 0 {
		first = first[:i]
	}
	first = strings.TrimSpace(first)
	// Strip any annotation/default so `self: Foo` / `cls = X` still match.
	for _, sep := range []string{":", "="} {
		if i := strings.Index(first, sep); i >= 0 {
			first = strings.TrimSpace(first[:i])
		}
	}
	return first == "self" || first == "cls"
}

// paramList takes a string beginning at the opening '(' of a parameter list and
// returns the inner text up to the matching ')'. ok is false if the list is not
// balanced on this line (e.g. a multi-line signature).
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

// countParams counts top-level comma-separated parameters in a parameter list,
// ignoring commas nested in generics/tuples/defaults. An empty (or whitespace)
// list counts as zero.
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
