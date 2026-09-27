package quality

import (
	"bytes"
	"slices"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// qMaskLiteralsAndComments masks string, char, backtick, and triple-quoted literals and //, #, /* */
// comments with 'x', keeping newlines. It tracks state across lines, so mask the whole file once.
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
				out[i] = analyze.MaskFiller
			}
		case inBlock:
			if c == '*' && i+1 < len(out) && out[i+1] == '/' {
				out[i] = analyze.MaskFiller
				out[i+1] = analyze.MaskFiller
				i++
				inBlock = false
			} else if c != '\n' {
				out[i] = analyze.MaskFiller
			}
		case triple != nil:
			if c == '\\' {
				if i+1 < len(out) && out[i+1] != '\n' {
					out[i] = analyze.MaskFiller
					out[i+1] = analyze.MaskFiller
				}
				i++
				continue
			}
			if c == triple[0] && analyze.HasTripleAt(out, i, triple[0]) {
				out[i] = analyze.MaskFiller
				out[i+1] = analyze.MaskFiller
				out[i+2] = analyze.MaskFiller
				i += 2
				triple = nil
			} else if c != '\n' {
				out[i] = analyze.MaskFiller
			}
		case inStr != 0:
			if c == '\\' {
				if i+1 < len(out) && out[i+1] != '\n' {
					out[i] = analyze.MaskFiller
					out[i+1] = analyze.MaskFiller
				}
				i++ // skip escaped char
				continue
			}
			if c == inStr {
				inStr = 0
				out[i] = analyze.MaskFiller
			} else if c == '\n' {
				inStr = 0 // unterminated; reset at line end
			} else {
				out[i] = analyze.MaskFiller
			}
		default:
			switch {
			case c == '/' && i+1 < len(out) && out[i+1] == '/':
				inLine = true
				out[i] = analyze.MaskFiller
			case c == '/' && i+1 < len(out) && out[i+1] == '*':
				inBlock = true
				out[i] = analyze.MaskFiller
			case c == '#':
				inLine = true
				out[i] = analyze.MaskFiller
			case (c == '"' || c == '\'') && analyze.HasTripleAt(out, i, c):
				triple = []byte{c}
				out[i] = analyze.MaskFiller
				out[i+1] = analyze.MaskFiller
				out[i+2] = analyze.MaskFiller
				i += 2
			case c == '"' || c == '`':
				inStr = c
				out[i] = analyze.MaskFiller
			case c == '\'':
				// Mask only a closed char literal, since a Rust lifetime (`'a`) must not swallow the
				// code after it.
				if end := qCharLitEnd(out, i); end > i {
					for j := i; j <= end; j++ {
						out[j] = analyze.MaskFiller
					}
					i = end
				}
				// else: lone single quote (lifetime) — not a string.
			}
		}
	}
	return out
}

// qCharLitEnd returns the index of the closing quote of a char literal (`'c'` or `'\n'`) opening at out[i],
// or i itself when there is none, as with a Rust lifetime `'a`.
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

// qFuncBody is a function-like symbol and its body lines, located by the extractor's symbol spans (like
// LongFunction) rather than by re-parsing signatures.
type qFuncBody struct {
	node  model.Node
	lines [][]byte // body lines, one per span line, with literals and comments masked
}

// qFuncBodiesMasked returns the masked body of every function-like symbol whose span fits the file, so
// keywords and operators inside literals, comments, and docstrings are not counted.
func qFuncBodiesMasked(t analyze.FileTarget) []qFuncBody {
	// Skip the masking when no function has a usable span (extractors that set no EndLine).
	if !slices.ContainsFunc(t.Nodes, qHasBodySpan) {
		return nil
	}
	fileLines := analyze.Cached(t, "quality/masked-lines", func() [][]byte {
		return bytes.Split(qMaskLiteralsAndComments(t.Content), []byte("\n"))
	})
	var out []qFuncBody
	for _, n := range t.Nodes {
		if !qHasBodySpan(n) || n.Span.EndLine > len(fileLines) {
			continue
		}
		out = append(out, qFuncBody{node: n, lines: fileLines[n.Span.StartLine-1 : n.Span.EndLine]})
	}
	return out
}

// qHasBodySpan reports whether n is function-like with a non-empty 1-based line span.
func qHasBodySpan(n model.Node) bool {
	return isFuncNode(n) && n.Span.StartLine >= 1 && n.Span.EndLine >= n.Span.StartLine
}
