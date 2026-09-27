package bugs

import (
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
)

// The maskers below replace literal and comment bodies with 'x', keeping newlines so line numbers hold.
// They track block comments and multi-line strings across lines, so mask the whole file once.

// maskJSLiteralsAndComments masks JS/TS string, template, and regex literal bodies plus comments. A `/`
// opens a regex only where regexCanStart says an operand cannot end, so division in `a / b` is left alone.
func maskJSLiteralsAndComments(src []byte) []byte {
	out := make([]byte, len(src))
	copy(out, src)
	inLine := false
	inBlock := false
	inStr := byte(0)   // 0, '"', '\'', or '`'
	inRegex := false   // currently inside a /.../ literal body
	inClass := false   // inside a [...] char class within a regex
	lastSig := byte(0) // last significant (non-space, non-masked) source byte
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
			} else if c != '\n' {
				out[i] = analyze.MaskFiller
			}
		case inRegex:
			if c == '\\' {
				if i+1 < len(out) && out[i+1] != '\n' {
					out[i] = analyze.MaskFiller
					out[i+1] = analyze.MaskFiller
				}
				i++
				continue
			}
			switch {
			case c == '\n':
				// Unterminated regex on a line; bail out to be safe.
				inRegex = false
				inClass = false
			case c == '[':
				inClass = true
				out[i] = analyze.MaskFiller
			case c == ']':
				inClass = false
				out[i] = analyze.MaskFiller
			case c == '/' && !inClass:
				inRegex = false
				out[i] = analyze.MaskFiller
			default:
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
			case c == '/' && regexCanStart(lastSig):
				inRegex = true
				out[i] = analyze.MaskFiller
				lastSig = analyze.MaskFiller
				continue
			case c == '"' || c == '\'' || c == '`':
				inStr = c
				out[i] = analyze.MaskFiller
				lastSig = analyze.MaskFiller
				continue
			}
			if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
				lastSig = c
			}
		}
	}
	return out
}

// regexCanStart reports whether a `/` after the significant byte prev opens a regex literal rather than
// division: prev is absent (start of input) or cannot end an expression operand.
func regexCanStart(prev byte) bool {
	switch prev {
	case 0,
		'(', '[', '{', ',', ';', ':', '?',
		'=', '!', '&', '|', '+', '-', '*', '/', '%', '^', '~',
		'<', '>':
		return true
	}
	return false
}

// maskPyStringsAndComments masks Python string bodies (including multi-line triple-quoted) and # comments.
// Quote delimiters stay intact so a real `x is "foo"` still shows the rule regex its `is "` signal.
func maskPyStringsAndComments(src []byte) []byte {
	out := make([]byte, len(src))
	copy(out, src)
	var triple []byte // open triple delimiter (""" or ''') when inside one
	inStr := byte(0)  // single-line ' or " when inside one
	for i := 0; i < len(out); i++ {
		c := out[i]
		switch {
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
				// Leave the closing delimiter intact.
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
				i++
				continue
			}
			if c == inStr {
				inStr = 0 // leave the closing delimiter intact
			} else if c == '\n' {
				inStr = 0 // unterminated; reset at line end
			} else {
				out[i] = analyze.MaskFiller
			}
		default:
			switch {
			case (c == '"' || c == '\'') && analyze.HasTripleAt(out, i, c):
				triple = []byte{c} // leave the opening delimiter intact
				i += 2
			case c == '"' || c == '\'':
				inStr = c // leave the opening delimiter intact
			case c == '#':
				// Mask the rest of the line as a comment.
				for j := i; j < len(out) && out[j] != '\n'; j++ {
					out[j] = analyze.MaskFiller
				}
			}
		}
	}
	return out
}

// stripInlineBlockComment removes same-line /* ... */ comments outside string literals, so
// `x = x /* note */` keeps its end-of-line anchor. An unterminated `/*` truncates the line.
func stripInlineBlockComment(s string) string {
	if !strings.Contains(s, "/*") {
		return s
	}
	var inStr byte
	b := []byte(s)
	for i := 0; i < len(b); i++ {
		c := b[i]
		if inStr != 0 {
			if c == '\\' && i+1 < len(b) {
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
			if i+1 < len(b) && b[i+1] == '*' {
				end := -1
				for j := i + 2; j+1 < len(b); j++ {
					if b[j] == '*' && b[j+1] == '/' {
						end = j + 2
						break
					}
				}
				if end == -1 {
					return string(b[:i]) // unterminated; drop the rest
				}
				// Splice out the comment span [i, end) and re-scan from i.
				b = append(b[:i], b[end:]...)
				i--
			}
		}
	}
	return string(b)
}
