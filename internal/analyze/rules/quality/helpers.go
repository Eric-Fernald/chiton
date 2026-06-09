package quality

import (
	"bytes"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// qStripComment drops the content of a // or # line comment from s while
// respecting simple string/char/backtick literals so a `//` inside a string is
// not treated as the start of a comment. Block comments are not tracked across
// lines, matching the extractor's per-line behaviour. The leading-`#` form
// covers Python/Ruby/shell; the `//` form covers C-family languages.
func qStripComment(s string) string {
	var inStr byte
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inStr != 0 {
			if c == '\\' && i+1 < len(s) {
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
			if i+1 < len(s) && s[i+1] == '/' {
				return s[:i]
			}
		case '#':
			return s[:i]
		}
	}
	return s
}

// qMaskFiller is the neutral byte that replaces literal/comment content so the
// rule regexes can no longer match keywords/operators that live inside them.
const qMaskFiller = 'x'

// qMaskLiteralsAndComments returns a copy of src with the BODIES of string,
// char, and backtick literals, // and # line comments, /* */ block comments,
// and Python triple-quoted strings replaced by the filler byte 'x'. Newlines
// are preserved so reported line numbers stay accurate. Block comments and
// triple-quoted strings are tracked across lines, so callers must mask the
// WHOLE file content once (not line-by-line). A lone single quote that is not a
// valid char literal (e.g. a Rust lifetime `&'a`) is NOT treated as a string
// opener, so code after it still reaches the rule regex.
func qMaskLiteralsAndComments(src []byte) []byte {
	out := make([]byte, len(src))
	copy(out, src)
	inLine := false
	inBlock := false
	var triple []byte // open triple delimiter (""" or ''') when inside one
	inStr := byte(0)  // 0, '"', '\'', or '`' for a single-line/ordinary string
	for i := 0; i < len(out); i++ {
		c := out[i]
		switch {
		case inLine:
			if c == '\n' {
				inLine = false
			} else {
				out[i] = qMaskFiller
			}
		case inBlock:
			if c == '*' && i+1 < len(out) && out[i+1] == '/' {
				out[i] = qMaskFiller
				out[i+1] = qMaskFiller
				i++
				inBlock = false
			} else if c != '\n' {
				out[i] = qMaskFiller
			}
		case triple != nil:
			if c == '\\' {
				if i+1 < len(out) && out[i+1] != '\n' {
					out[i] = qMaskFiller
					out[i+1] = qMaskFiller
				}
				i++
				continue
			}
			if c == triple[0] && qHasTripleAt(out, i, triple[0]) {
				out[i] = qMaskFiller
				out[i+1] = qMaskFiller
				out[i+2] = qMaskFiller
				i += 2
				triple = nil
			} else if c != '\n' {
				out[i] = qMaskFiller
			}
		case inStr != 0:
			if c == '\\' {
				if i+1 < len(out) && out[i+1] != '\n' {
					out[i] = qMaskFiller
					out[i+1] = qMaskFiller
				}
				i++ // skip escaped char
				continue
			}
			if c == inStr {
				inStr = 0
				out[i] = qMaskFiller
			} else if c == '\n' {
				inStr = 0 // unterminated; reset at line end
			} else {
				out[i] = qMaskFiller
			}
		default:
			switch {
			case c == '/' && i+1 < len(out) && out[i+1] == '/':
				inLine = true
				out[i] = qMaskFiller
			case c == '/' && i+1 < len(out) && out[i+1] == '*':
				inBlock = true
				out[i] = qMaskFiller
			case c == '#':
				inLine = true
				out[i] = qMaskFiller
			case (c == '"' || c == '\'') && qHasTripleAt(out, i, c):
				triple = []byte{c}
				out[i] = qMaskFiller
				out[i+1] = qMaskFiller
				out[i+2] = qMaskFiller
				i += 2
			case c == '"' || c == '`':
				inStr = c
				out[i] = qMaskFiller
			case c == '\'':
				// Distinguish a char literal (`'c'`, `'\n'`) from a Rust
				// lifetime/label (`'a`) which has no closing quote. Only treat
				// it as a string opener when a valid char literal closes nearby;
				// otherwise leave it (and the following ident) untouched so code
				// after a lifetime still reaches the rule regex.
				if end := qCharLitEnd(out, i); end > i {
					for j := i; j <= end; j++ {
						out[j] = qMaskFiller
					}
					i = end
				}
				// else: lone single quote (lifetime) — not a string.
			}
		}
	}
	return out
}

// qCharLitEnd reports the index of the closing quote of a char literal starting
// at out[i] (which must be '\”), or i itself (no advance) when out[i:] is not a
// valid char literal — e.g. a Rust lifetime `'a` with no closing quote. Handles
// the escaped form `'\n'`.
func qCharLitEnd(out []byte, i int) int {
	// `'\x'` — escaped char literal.
	if i+3 < len(out) && out[i+1] == '\\' && out[i+3] == '\'' {
		return i + 3
	}
	// `'c'` — simple char literal (the inner char must not itself be a quote).
	if i+2 < len(out) && out[i+1] != '\'' && out[i+2] == '\'' {
		return i + 2
	}
	return i
}

// qHasTripleAt reports whether out[i:i+3] is three consecutive q bytes.
func qHasTripleAt(out []byte, i int, q byte) bool {
	return i+2 < len(out) && out[i] == q && out[i+1] == q && out[i+2] == q
}

// qFuncBody describes a single function-like symbol and the source lines that
// make up its body, so per-function rules (complexity, return-count) can share
// one boundary-detection path modelled on the long-function rule: both read the
// extractor's symbol spans rather than re-parsing function signatures.
type qFuncBody struct {
	node  model.Node
	lines []string // body source lines, 1:1 with the span (comments NOT stripped)
}

// qFuncBodiesMasked returns one qFuncBody per function-like symbol in the
// target that has a valid span within the file content, with the body lines'
// string, char, comment, and triple-quoted-string contents replaced by filler
// bytes (a per-FILE masking pass). It mirrors LongFunction's reliance on
// extractor-provided spans, so it inherits the same notion of "function".
// Per-function rules that scan for keywords/operators (complexity, return-count)
// use this so tokens inside literals/comments/docstrings — and Rust lifetimes —
// are not counted.
func qFuncBodiesMasked(t analyze.FileTarget) []qFuncBody {
	return funcBodiesFrom(t, qMaskLiteralsAndComments(t.Content))
}

// funcBodiesFrom is the shared implementation: it slices content into the body
// lines of each function-like symbol using the extractor's spans.
func funcBodiesFrom(t analyze.FileTarget, content []byte) []qFuncBody {
	fileLines := bytes.Split(content, []byte("\n"))
	var out []qFuncBody
	for _, n := range t.Nodes {
		if n.Kind != model.KindSymbol || n.Span == nil {
			continue
		}
		lk, _ := n.Attrs["lang_kind"].(string)
		if !isFuncLike(lk) {
			continue
		}
		start, end := n.Span.StartLine, n.Span.EndLine
		if start < 1 || end < start || end > len(fileLines) {
			continue
		}
		body := make([]string, 0, end-start+1)
		for i := start - 1; i < end; i++ {
			body = append(body, string(fileLines[i]))
		}
		out = append(out, qFuncBody{node: n, lines: body})
	}
	return out
}
