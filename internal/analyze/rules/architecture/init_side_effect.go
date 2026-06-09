package architecture

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// InitSideEffect flags a package-level `func init()` in Go ONLY when its body
// does real side-effect work — reading/writing the environment, file I/O,
// network or DB setup, or mutating package globals. A bare init that merely
// performs registration (the idiomatic `analyze.Register(...)` pattern this very
// rule uses) is NOT flagged: registration-only inits are conventional Go and
// flagging all 80+ of them is pure noise.
//
// Anchored at column 0 so only top-level declarations match — `init` cannot name
// a func literal, so a column-0 `func init(` is always the special init func.
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

// sideEffectRe matches the side-effect signatures we treat as "real work" inside
// an init body: env reads/writes, file I/O, network/DB setup, and reflection-y
// global wiring. Word boundaries keep these from matching mid-identifier.
var sideEffectRe = regexp.MustCompile(
	`\bos\.(Getenv|Setenv|LookupEnv|Open|OpenFile|Create|ReadFile|WriteFile|Mkdir|MkdirAll|Remove|RemoveAll)\b` +
		`|\bioutil\.(ReadFile|WriteFile|ReadDir)\b` +
		`|\bsql\.Open\b` +
		`|\bhttp\.\w+\b` +
		`|\.Dial\b`)

// globalMutationRe matches an assignment to a package-global-looking target on a
// line that is NOT a short-var declaration (`:=`) and NOT a local `var`. We
// approximate a package global as a bare/dotted identifier assigned with `=`
// (not `==`, `:=`, `<=`, `>=`, `!=`, `+=`, etc.). This is a heuristic, kept
// conservative so locals declared with `:=` never trip it.
var globalMutationRe = regexp.MustCompile(`(?m)^\s*[A-Za-z_][\w.]*(?:\[[^\]]*\])?\s*=[^=]`)

func (InitSideEffect) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	// Per-file masking pass: blank out the bodies of // line comments,
	// /* */ block comments, and "", '', `` string literals (newlines kept so
	// line numbers stay accurate). This stops a `func init(` inside a block
	// comment or backtick raw string from matching, and keeps braces that live
	// inside strings/comments from corrupting body-brace matching.
	masked := maskArchLiteralsAndComments(t.Content)

	var out []analyze.Finding
	for _, m := range goInitFuncRe.FindAllIndex(masked, -1) {
		// Locate the body: from the first '{' at/after the match to its matching
		// '}', using the masked content for brace counting.
		bodyStart := indexByteFrom(masked, '{', m[1])
		if bodyStart < 0 {
			continue
		}
		bodyEnd := matchingBrace(masked, bodyStart)
		if bodyEnd < 0 {
			continue
		}
		// Examine the ORIGINAL (unmasked) body so signatures inside the code are
		// visible, but the brace span came from the masked content.
		body := t.Content[bodyStart+1 : bodyEnd]
		if !bodyHasSideEffect(body) {
			continue
		}
		line := 1 + countNewlines(masked[:m[0]])
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

// bodyHasSideEffect reports whether an init body does real side-effect work.
// It masks the body's own strings/comments first (so an env name in a string
// literal, or a `// os.Getenv` comment, does not count), then looks for the
// side-effect signatures or a package-global mutation — while ignoring lines
// whose only "work" is a registration-style call.
func bodyHasSideEffect(body []byte) bool {
	mb := maskArchLiteralsAndComments(body)
	if sideEffectRe.Match(mb) {
		return true
	}
	// A package-global mutation is real work. Registration calls
	// (analyze.Register(...), sql.Register(...), RegisterFoo(...)) use no top-level
	// `=`, so a bare `=` assignment to a global never matches one of them.
	return globalMutationRe.Match(mb)
}

// indexByteFrom returns the index of the first b at or after start, or -1.
func indexByteFrom(s []byte, b byte, start int) int {
	for i := start; i < len(s); i++ {
		if s[i] == b {
			return i
		}
	}
	return -1
}

// matchingBrace returns the index of the '}' that closes the '{' at open, or -1
// if unbalanced. s must already be masked so braces inside strings/comments are
// gone.
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

// countNewlines counts '\n' bytes in s.
func countNewlines(s []byte) int {
	n := 0
	for _, b := range s {
		if b == '\n' {
			n++
		}
	}
	return n
}

// maskArchLiteralsAndComments returns a copy of src with the bodies of line
// comments, block comments, and double-quote, single-quote, and backtick string
// literals replaced by the filler byte 'x' (newlines preserved so line numbers
// stay accurate). The delimiters themselves are overwritten too so they cannot
// be re-interpreted by the rule regex. Block comments and backtick raw strings
// are tracked across line boundaries (this is a per-file pass).
func maskArchLiteralsAndComments(src []byte) []byte {
	out := make([]byte, len(src))
	copy(out, src)
	const filler = 'x'
	inLine := false
	inBlock := false
	inStr := byte(0) // 0, '"', '\'', or '`'
	for i := 0; i < len(out); i++ {
		c := out[i]
		switch {
		case inLine:
			if c == '\n' {
				inLine = false
			} else {
				out[i] = filler
			}
		case inBlock:
			if c == '*' && i+1 < len(out) && out[i+1] == '/' {
				out[i] = filler
				out[i+1] = filler
				i++
				inBlock = false
			} else if c != '\n' {
				out[i] = filler
			}
		case inStr != 0:
			// Backtick raw strings have no escapes; "" and '' do.
			if inStr != '`' && c == '\\' {
				if i+1 < len(out) && out[i+1] != '\n' {
					out[i] = filler
					out[i+1] = filler
				}
				i++ // skip escaped char
				continue
			}
			if c == inStr {
				inStr = 0
				out[i] = filler
			} else if c != '\n' {
				out[i] = filler
			}
		default:
			switch {
			case c == '/' && i+1 < len(out) && out[i+1] == '/':
				inLine = true
				out[i] = filler
			case c == '/' && i+1 < len(out) && out[i+1] == '*':
				inBlock = true
				out[i] = filler
			case c == '"' || c == '\'' || c == '`':
				inStr = c
				out[i] = filler
			}
		}
	}
	return out
}

func init() { analyze.Register(InitSideEffect{}) }
