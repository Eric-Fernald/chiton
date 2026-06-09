package reliability

import (
	"bufio"
	"bytes"
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

const rustMaskFiller = 'x'

// RustUnwrap flags `.unwrap()` and `.expect(...)` in Rust source. Both panic at
// runtime when called on an Err/None, turning a recoverable error into a process
// abort. Idiomatic Rust propagates with `?` or matches explicitly.
type RustUnwrap struct{}

func (RustUnwrap) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "reliability/rust-unwrap",
		Category:   analyze.CategoryReliability,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"rs"},
		Summary:    ".unwrap()/.expect() panics on Err/None instead of propagating.",
		Fix:        "Use the `?` operator, match the Result/Option, or unwrap_or*.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// `.unwrap()` and `.expect(` — anchored on the leading dot so it only matches
// method calls. `.unwrap_or(...)`, `.unwrap_or_else(...)`, `.unwrap_or_default()`
// are intentionally NOT matched (the `\(` right after `unwrap` excludes them).
var rustUnwrapRe = regexp.MustCompile(`\.(unwrap\s*\(|expect\s*\()`)

func (RustUnwrap) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	// Skip integration tests (conventionally under a tests/ directory) and unit
	// test files, where panicking on unwrap is the accepted assertion style.
	lp := strings.ToLower(filepathSlashed(t.Path))
	if strings.Contains(lp, "/tests/") || strings.HasPrefix(lp, "tests/") {
		return nil
	}

	// Mask Rust string literals (regular, raw r"..."/r#"..."#, byte and char
	// literals) and //-line plus /* */-block comments as a per-file pass, so a
	// `.unwrap()` written inside a string literal, a format!/println! macro
	// argument, or a (possibly multi-line) block comment no longer matches.
	masked := maskRustLiteralsAndComments(t.Content)

	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(masked))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
		if rustUnwrapRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "reliability/rust-unwrap",
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryReliability,
				Message:      ".unwrap()/.expect() panics on Err/None",
				Fix:          "Propagate with `?`, match the Result/Option, or use unwrap_or*.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

// filepathSlashed normalises Windows backslashes to forward slashes so path
// checks are OS-independent. Package-unique helper name.
func filepathSlashed(p string) string {
	return strings.ReplaceAll(p, "\\", "/")
}

// maskRustLiteralsAndComments returns a copy of src with the bodies of Rust
// string literals, char/byte literals, and // and /* */ comments replaced by a
// filler byte (newlines preserved so line numbers stay accurate). It handles
// Rust raw strings (`r"..."`, `r#"..."#`, `br"..."`, etc., where the number of
// `#` hashes around the quotes must balance) and nested block comments (Rust
// permits `/* /* */ */`). Block comments and raw/regular strings are tracked
// across line boundaries, so a `.unwrap()` inside any of them never matches.
func maskRustLiteralsAndComments(src []byte) []byte {
	out := make([]byte, len(src))
	copy(out, src)
	const filler = rustMaskFiller
	blockDepth := 0 // nested /* */ depth
	for i := 0; i < len(out); {
		// Inside a (possibly nested) block comment.
		if blockDepth > 0 {
			switch {
			case out[i] == '/' && i+1 < len(out) && out[i+1] == '*':
				out[i], out[i+1] = filler, filler
				blockDepth++
				i += 2
			case out[i] == '*' && i+1 < len(out) && out[i+1] == '/':
				out[i], out[i+1] = filler, filler
				blockDepth--
				i += 2
			default:
				if out[i] != '\n' {
					out[i] = filler
				}
				i++
			}
			continue
		}

		c := out[i]
		switch {
		case c == '/' && i+1 < len(out) && out[i+1] == '/':
			// Line comment: blank to end of line.
			out[i] = filler
			i++
			for i < len(out) && out[i] != '\n' {
				out[i] = filler
				i++
			}
		case c == '/' && i+1 < len(out) && out[i+1] == '*':
			out[i], out[i+1] = filler, filler
			blockDepth = 1
			i += 2
		case c == '"':
			i = maskRustRegularString(out, i, filler)
		case (c == 'r' || c == 'b') && rustRawStringStart(out, i):
			i = maskRustRawString(out, i, filler)
		case c == '\'':
			// Distinguish a char/byte literal from a lifetime (`'a`). A char
			// literal is `'x'`, `'\n'`, etc.; a lifetime has no closing quote
			// near the identifier. Only mask when we find a closing quote.
			if end, ok := rustCharLiteralEnd(out, i); ok {
				for j := i; j <= end; j++ {
					if out[j] != '\n' {
						out[j] = filler
					}
				}
				i = end + 1
			} else {
				i++
			}
		default:
			i++
		}
	}
	return out
}

// maskRustRegularString masks a normal "..." string starting at openIdx and
// returns the index just past the closing quote (or len when unterminated).
func maskRustRegularString(out []byte, openIdx int, filler byte) int {
	out[openIdx] = filler
	for i := openIdx + 1; i < len(out); i++ {
		if out[i] == '\\' {
			out[i] = filler
			if i+1 < len(out) && out[i+1] != '\n' {
				out[i+1] = filler
			}
			i++
			continue
		}
		if out[i] == '"' {
			out[i] = filler
			return i + 1
		}
		if out[i] != '\n' {
			out[i] = filler
		}
	}
	return len(out)
}

// rustRawStringStart reports whether a raw-string prefix begins at i: an `r`
// (optionally preceded by a `b` for byte strings) followed by zero or more `#`
// and then a `"`. i points at the `r` or the `b`.
func rustRawStringStart(out []byte, i int) bool {
	j := i
	if out[j] == 'b' {
		j++
	}
	if j >= len(out) || out[j] != 'r' {
		return false
	}
	j++
	for j < len(out) && out[j] == '#' {
		j++
	}
	return j < len(out) && out[j] == '"'
}

// maskRustRawString masks a raw string `r"..."` / `r#"..."#` / `br#"..."#`
// starting at openIdx and returns the index just past the terminator. The
// terminator is a `"` followed by exactly the same number of `#` as the opener.
func maskRustRawString(out []byte, openIdx int, filler byte) int {
	j := openIdx
	if out[j] == 'b' {
		out[j] = filler
		j++
	}
	out[j] = filler // 'r'
	j++
	hashes := 0
	for j < len(out) && out[j] == '#' {
		out[j] = filler
		hashes++
		j++
	}
	// j now points at the opening quote.
	out[j] = filler
	j++
	for j < len(out) {
		if out[j] == '"' {
			// Check for the required number of trailing '#'.
			k := j + 1
			cnt := 0
			for k < len(out) && cnt < hashes && out[k] == '#' {
				cnt++
				k++
			}
			if cnt == hashes {
				out[j] = filler
				for m := j + 1; m < k; m++ {
					out[m] = filler
				}
				return k
			}
		}
		if out[j] != '\n' {
			out[j] = filler
		}
		j++
	}
	return len(out)
}

// rustCharLiteralEnd returns the index of the closing quote of a char/byte
// literal starting at the opening quote at i, and whether a valid one was found.
// It scans a bounded window so a lifetime annotation (`'a`) does not consume the
// rest of the file.
func rustCharLiteralEnd(out []byte, i int) (int, bool) {
	// Empty or runaway: bail. A char literal is at most a few bytes
	// (`'\u{1F600}'`), so cap the search window.
	const maxChar = 12
	j := i + 1
	if j < len(out) && out[j] == '\\' {
		j++ // escape: the next byte is part of the literal
	}
	limit := i + maxChar
	if limit > len(out) {
		limit = len(out)
	}
	for ; j < limit; j++ {
		if out[j] == '\n' {
			return 0, false
		}
		if out[j] == '\'' {
			return j, true
		}
	}
	return 0, false
}

func init() { analyze.Register(RustUnwrap{}) }
