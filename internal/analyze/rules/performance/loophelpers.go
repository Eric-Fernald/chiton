package performance

import "strings"

// braceLoopScanner tracks whether a line lies inside a `{ }` loop body, using a stack so nested
// loops and inner non-loop blocks work, and tolerating an opening brace on the next line (Allman).
type braceLoopScanner struct {
	depth        int
	loopStack    []int // brace depths at which a loop body opened
	pendingBrace bool  // a loop header was seen, awaiting its opening `{`
}

// popExited drops the loop frames whose bodies the current depth has left.
func (b *braceLoopScanner) popExited() {
	for len(b.loopStack) > 0 && b.depth < b.loopStack[len(b.loopStack)-1] {
		b.loopStack = b.loopStack[:len(b.loopStack)-1]
	}
}

// feed advances by one line (isLoopLine: it holds a for/while header) and reports whether the
// line, after its own brace accounting, is inside a loop body.
func (b *braceLoopScanner) feed(stripped string, isLoopLine bool) (insideLoop bool) {
	open := strings.Count(stripped, "{")
	closeN := strings.Count(stripped, "}")

	if isLoopLine {
		if open > 0 {
			// The loop body content sits one brace-level deeper than the
			// depth just before this line, so the frame records preDepth+1.
			b.loopStack = append(b.loopStack, b.depth+1)
			b.depth += open - closeN
			// Pop now so a self-closing `for (...) { stmt; }` (net delta 0) doesn't leave
			// later lines at this depth treated as inside the loop.
			b.popExited()
		} else {
			b.depth += open - closeN
			// Brace likely on a following line.
			b.pendingBrace = true
		}
		// The for/while header line itself is not "inside" the body.
		return false
	}

	// A pending loop header consumes the next opening brace.
	if b.pendingBrace && open > 0 {
		// The loop body opens at this brace, so its content is one level
		// deeper than the depth before this line.
		b.loopStack = append(b.loopStack, b.depth+1)
		b.depth += open - closeN
		b.pendingBrace = false
		// Pop again in case the body also closes on this same line
		// (`{ stmt; }` all on the brace line).
		b.popExited()
		return false
	}
	b.pendingBrace = false

	b.depth += open - closeN
	b.popExited()
	return len(b.loopStack) > 0
}

// maskPyStrings returns a copy of src with Python strings (triple-quoted ones tracked across lines)
// and # comments, delimiters included, overwritten by 'x', keeping newlines for line numbers.
func maskPyStrings(src []byte) []byte {
	out := make([]byte, len(src))
	copy(out, src)
	const filler = 'x'
	inLine := false
	triple := byte(0) // quote char while inside a triple-quoted string
	inStr := byte(0)  // 0, '"', or '\'' for single-line strings
	for i := 0; i < len(out); i++ {
		c := out[i]
		switch {
		case inLine:
			if c == '\n' {
				inLine = false
			} else {
				out[i] = filler
			}
		case triple != 0:
			if c == triple && i+2 < len(out) && out[i+1] == c && out[i+2] == c {
				out[i] = filler
				out[i+1] = filler
				out[i+2] = filler
				i += 2
				triple = 0
			} else if c != '\n' {
				out[i] = filler
			}
		case inStr != 0:
			if c == '\\' {
				if i+1 < len(out) && out[i+1] != '\n' {
					out[i] = filler
					out[i+1] = filler
				}
				i++
				continue
			}
			if c == inStr || c == '\n' {
				if c != '\n' {
					out[i] = filler
				}
				inStr = 0
			} else {
				out[i] = filler
			}
		default:
			switch {
			case c == '#':
				inLine = true
				out[i] = filler
			case (c == '"' || c == '\'') && i+2 < len(out) && out[i+1] == c && out[i+2] == c:
				triple = c
				out[i] = filler
				out[i+1] = filler
				out[i+2] = filler
				i += 2
			case c == '"' || c == '\'':
				inStr = c
				out[i] = filler
			}
		}
	}
	return out
}
