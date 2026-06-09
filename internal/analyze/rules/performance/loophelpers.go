package performance

import "strings"

// stripLineComment drops the content of a trailing line comment (`//` for
// C-style languages, `#` for Python) while respecting string/char literals so
// a `//` or `#` inside a string is not treated as a comment start. Block
// comments are not tracked across lines, matching the extractor's behaviour.
// Named uniquely within the package to avoid colliding with helpers in other
// rule packages.
func stripLineComment(s string) string {
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

// braceLoopScanner tracks whether the current line lies inside a C-style loop
// body delimited by `{ }`. It mirrors the depth/loopDepth bookkeeping in
// string_concat_in_loop.go but supports nested loops via a stack so an inner
// non-loop block does not clear the enclosing loop state. It also tolerates a
// loop header whose opening brace sits on the following line (Java/Allman
// style).
type braceLoopScanner struct {
	depth        int
	loopStack    []int // brace depths at which a loop body opened
	pendingBrace bool  // a loop header was seen, awaiting its opening `{`
}

// feed advances the scanner by one line. isLoopLine reports whether the
// stripped line opens a for/while loop header. It returns whether the line
// (after this line's own brace accounting) is inside a loop body.
func (b *braceLoopScanner) feed(stripped string, isLoopLine bool) (insideLoop bool) {
	open := strings.Count(stripped, "{")
	closeN := strings.Count(stripped, "}")

	if isLoopLine {
		if open > 0 {
			// The loop body content sits one brace-level deeper than the
			// depth just before this line, so the frame records preDepth+1.
			b.loopStack = append(b.loopStack, b.depth+1)
			b.depth += open - closeN
			// A self-closing single-line loop (`for (...) { stmt; }`) has a
			// net brace delta of zero, so the frame just pushed must be popped
			// on this same line. Without this, every later line at the same
			// brace depth is wrongly treated as inside this loop. Run the
			// standard pop logic so the self-closing frame is removed.
			for len(b.loopStack) > 0 && b.depth < b.loopStack[len(b.loopStack)-1] {
				b.loopStack = b.loopStack[:len(b.loopStack)-1]
			}
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
		for len(b.loopStack) > 0 && b.depth < b.loopStack[len(b.loopStack)-1] {
			b.loopStack = b.loopStack[:len(b.loopStack)-1]
		}
		return false
	}
	b.pendingBrace = false

	b.depth += open - closeN
	// Pop loop frames we have exited.
	for len(b.loopStack) > 0 && b.depth < b.loopStack[len(b.loopStack)-1] {
		b.loopStack = b.loopStack[:len(b.loopStack)-1]
	}
	return len(b.loopStack) > 0
}

// maskPerfLiteralsComments returns a copy of src with the bodies (and
// delimiters) of double/single-quoted and backtick string literals plus // line
// and /* */ block comments replaced by the filler byte 'x'. Newlines are
// preserved so line numbers stay accurate. Block comments are tracked across
// lines (a per-file pass), so a keyword inside a multi-line /* */ comment is
// masked. Named uniquely within the package; modeled on reliability's
// maskLiteralsAndComments.
func maskPerfLiteralsComments(src []byte) []byte {
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
			if c == '\\' {
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

// maskPyStrings returns a copy of src with the contents (and delimiters) of
// Python string literals — triple-quoted ("""/”' docstrings, tracked across
// lines) and single-line '...'/"..." strings — plus # line comments replaced by
// the filler byte 'x'. Newlines are preserved so line numbers stay accurate.
// This is a per-file pass so a pattern inside a multi-line docstring is masked.
func maskPyStrings(src []byte) []byte {
	out := make([]byte, len(src))
	copy(out, src)
	const filler = 'x'
	inLine := false
	var triple string // "", `"""`, or `'''`
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
		case triple != "":
			if c == triple[0] && string(srcRun(out, i, 3)) == triple {
				out[i] = filler
				out[i+1] = filler
				out[i+2] = filler
				i += 2
				triple = ""
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
				triple = string([]byte{c, c, c})
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

// srcRun returns up to n bytes of b starting at i (fewer if near the end).
func srcRun(b []byte, i, n int) []byte {
	if i+n > len(b) {
		n = len(b) - i
	}
	return b[i : i+n]
}
