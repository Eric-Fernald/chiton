package bugs

// Per-file masking helpers for the bugs rules. The older strip* helpers in this
// package only TRUNCATE a line at the first comment marker outside a string;
// they leave string/template/regex bodies in place, so an operator or keyword
// that lives INSIDE a literal still reaches the rule regex (a false positive).
//
// The functions below replace the BODIES of string/template/regex literals and
// comments with a neutral filler byte ('x'), preserving newlines so line
// numbers stay accurate. Block comments and multi-line strings are tracked
// across lines, so callers must mask the whole file once (not line-by-line).

const bugsMaskFiller = 'x'

// maskJSLiteralsAndComments masks the bodies of JS/TS string ("..."/'...'),
// template (`...`), and regex (/.../ ) literals plus // line and /* */ block
// comments. Regex detection is heuristic: a `/` only opens a regex when the
// previous non-space byte is one that cannot terminate an expression (operator,
// open bracket, comma, etc.) or is absent (start of input). This avoids
// mistaking the division operator `a / b` for a regex while still masking
// genuine literals like `/a==b/`.
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
				out[i] = bugsMaskFiller
			}
		case inBlock:
			if c == '*' && i+1 < len(out) && out[i+1] == '/' {
				out[i] = bugsMaskFiller
				out[i+1] = bugsMaskFiller
				i++
				inBlock = false
			} else if c != '\n' {
				out[i] = bugsMaskFiller
			}
		case inStr != 0:
			if c == '\\' {
				if i+1 < len(out) && out[i+1] != '\n' {
					out[i] = bugsMaskFiller
					out[i+1] = bugsMaskFiller
				}
				i++ // skip escaped char
				continue
			}
			if c == inStr {
				inStr = 0
				out[i] = bugsMaskFiller
			} else if c != '\n' {
				out[i] = bugsMaskFiller
			}
		case inRegex:
			if c == '\\' {
				if i+1 < len(out) && out[i+1] != '\n' {
					out[i] = bugsMaskFiller
					out[i+1] = bugsMaskFiller
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
				out[i] = bugsMaskFiller
			case c == ']':
				inClass = false
				out[i] = bugsMaskFiller
			case c == '/' && !inClass:
				inRegex = false
				out[i] = bugsMaskFiller
			default:
				out[i] = bugsMaskFiller
			}
		default:
			switch {
			case c == '/' && i+1 < len(out) && out[i+1] == '/':
				inLine = true
				out[i] = bugsMaskFiller
			case c == '/' && i+1 < len(out) && out[i+1] == '*':
				inBlock = true
				out[i] = bugsMaskFiller
			case c == '/' && regexCanStart(lastSig):
				inRegex = true
				out[i] = bugsMaskFiller
				lastSig = bugsMaskFiller
				continue
			case c == '"' || c == '\'' || c == '`':
				inStr = c
				out[i] = bugsMaskFiller
				lastSig = bugsMaskFiller
				continue
			}
			if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
				lastSig = c
			}
		}
	}
	return out
}

// regexCanStart reports whether a `/` following the given previous significant
// byte should be treated as the start of a regex literal rather than division.
// A regex can begin at the start of input (prev == 0) or after a byte that
// cannot end an expression operand (operators, open brackets, separators).
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

// maskPyStringsAndComments masks the BODIES of Python string literals — single
// and double quoted ('...'/"...") and triple-quoted (”'...”'/"""...""") — plus
// `#` line comments, replacing them with the filler byte. The opening and
// closing quote DELIMITERS are intentionally left intact so that a genuine
// identity-vs-literal comparison such as `x is "foo"` still presents an `is "`
// signal to the rule regex, while text INSIDE the string (e.g. the words
// "...is 42...") is neutralised. Triple-quoted strings are tracked across lines,
// so a multi-line docstring is masked in full; the single-line triple-quoted
// case (a one-line docstring) is therefore also handled. Newlines are preserved.
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
					out[i] = bugsMaskFiller
					out[i+1] = bugsMaskFiller
				}
				i++
				continue
			}
			if c == triple[0] && hasTripleAt(out, i, triple[0]) {
				// Leave the closing delimiter intact.
				i += 2
				triple = nil
			} else if c != '\n' {
				out[i] = bugsMaskFiller
			}
		case inStr != 0:
			if c == '\\' {
				if i+1 < len(out) && out[i+1] != '\n' {
					out[i] = bugsMaskFiller
					out[i+1] = bugsMaskFiller
				}
				i++
				continue
			}
			if c == inStr {
				inStr = 0 // leave the closing delimiter intact
			} else if c == '\n' {
				inStr = 0 // unterminated; reset at line end
			} else {
				out[i] = bugsMaskFiller
			}
		default:
			switch {
			case (c == '"' || c == '\'') && hasTripleAt(out, i, c):
				triple = []byte{c} // leave the opening delimiter intact
				i += 2
			case c == '"' || c == '\'':
				inStr = c // leave the opening delimiter intact
			case c == '#':
				// Mask the rest of the line as a comment.
				for j := i; j < len(out) && out[j] != '\n'; j++ {
					out[j] = bugsMaskFiller
				}
			}
		}
	}
	return out
}

// hasTripleAt reports whether out[i:i+3] is three consecutive q bytes.
func hasTripleAt(out []byte, i int, q byte) bool {
	return i+2 < len(out) && out[i] == q && out[i+1] == q && out[i+2] == q
}

// stripInlineBlockComment removes single-line `/* ... */` block comments from a
// line (the `/*` and matching `*/` both present on this line), ignoring markers
// inside string literals. Used by the self-assignment rule so a trailing block
// comment like `x = x /* note */` does not defeat its end-of-line anchor. Lines
// with an unterminated `/*` are returned truncated at the marker.
func stripInlineBlockComment(s string) string {
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
