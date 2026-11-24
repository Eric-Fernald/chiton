package architecture

import (
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// InitSideEffect flags a Go package-level `func init()` only when its body does real work: env access,
// file I/O, network or DB setup, or global mutation. It skips idiomatic registration-only inits.
type InitSideEffect struct{}

func (InitSideEffect) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "architecture/init-side-effect",
		Category:   analyze.CategoryArchitecture,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go"},
		Summary:    "Package-level func init() runs hidden side effects on import.",
		Fix:        "Prefer explicit initialization the caller invokes over an implicit init().",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// Column-0 `func init(` — package-level only. gofmt never emits a space before
// the paren, so requiring its absence keeps the match tight.
var goInitFuncRe = regexp.MustCompile(`(?m)^func init\(\s*\)`)

// sideEffectRe matches the "real work" signatures in an init body: env reads and writes, file I/O,
// and network or DB setup.
var sideEffectRe = regexp.MustCompile(
	`\bos\.(Getenv|Setenv|LookupEnv|Open|OpenFile|Create|ReadFile|WriteFile|Mkdir|MkdirAll|Remove|RemoveAll)\b` +
		`|\bioutil\.(ReadFile|WriteFile|ReadDir)\b` +
		`|\bsql\.Open\b` +
		`|\bhttp\.\w+\b` +
		`|\.Dial\b`)

// globalMutationRe approximates a package-global mutation as a line-leading identifier (dotted, optionally
// indexed) assigned with plain `=`, so `:=`, `var`, `==`, and `+=` never match.
var globalMutationRe = regexp.MustCompile(`(?m)^\s*[A-Za-z_][\w.]*(?:\[[^\]]*\])?\s*=[^=]`)

func (InitSideEffect) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	// The prefilter is exact because masking writes only 'x', so a masked `func init(` is a raw one too.
	if !bytes.Contains(t.Content, []byte("func init(")) {
		return nil
	}
	// Mask comments and string literals so a `func init(` inside them never matches and their
	// braces cannot corrupt body matching.
	masked := analyze.MaskedCLiterals(t, true)

	var out []analyze.Finding
	for _, m := range goInitFuncRe.FindAllIndex(masked, -1) {
		// Locate the body: from the first '{' at/after the match to its matching
		// '}', using the masked content for brace counting.
		bodyStart := bytes.IndexByte(masked[m[1]:], '{')
		if bodyStart < 0 {
			continue
		}
		bodyStart += m[1]
		bodyEnd := matchingBrace(masked, bodyStart)
		if bodyEnd < 0 {
			continue
		}
		// The body opens at an unmasked '{', so this slice is exactly the body masked on its own.
		if !bodyHasSideEffect(masked[bodyStart+1 : bodyEnd]) {
			continue
		}
		line := 1 + bytes.Count(masked[:m[0]], []byte{'\n'})
		out = append(out, analyze.Finding{
			RuleID:       "architecture/init-side-effect",
			Severity:     analyze.SeverityLow,
			Category:     analyze.CategoryArchitecture,
			Message:      "package-level func init() runs side effects (env/IO/network/DB or global mutation) on import",
			Fix:          "Prefer explicit initialization the caller invokes.",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		})
	}
	return out
}

// bodyHasSideEffect reports whether a masked init body calls a side-effect API or assigns a
// package global. Registration calls like analyze.Register(...) have no top-level `=`.
func bodyHasSideEffect(masked []byte) bool {
	return sideEffectRe.Match(masked) || globalMutationRe.Match(masked)
}

// matchingBrace returns the index of the '}' that closes the '{' at open, or -1. s must be masked.
func matchingBrace(s []byte, open int) int {
	depth := 0
	for i := open; i < len(s); i++ {
		switch s[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

func init() { analyze.Register(InitSideEffect{}) }
