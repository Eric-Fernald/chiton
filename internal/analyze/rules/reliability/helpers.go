package reliability

import "strings"

// maskGoLiteralsAndComments returns a copy of src with the bodies of Go string
// literals (interpreted "..." and raw `...`), rune literals, and // line plus
// /* */ block comments replaced by the filler byte 'x' (newlines preserved so
// line numbers stay accurate). Block comments and raw strings are tracked across
// line boundaries as a per-file pass, so code commented out inside a /* ... */
// block — or an os.Exit/panic/log.Fatal token sitting inside a string — is not
// flagged by the Go line-scan rules in this package. Package-unique name.
func maskGoLiteralsAndComments(src []byte) []byte {
	out := make([]byte, len(src))
	copy(out, src)
	const filler = 'x'
	inLine := false
	inBlock := false
	inStr := byte(0) // 0, '"', '`', or '\'' (rune)
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
			// Backtick raw strings have no escapes; "..." and '...' do.
			if inStr != '`' && c == '\\' {
				out[i] = filler
				if i+1 < len(out) && out[i+1] != '\n' {
					out[i+1] = filler
				}
				i++
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
			case c == '"' || c == '`' || c == '\'':
				inStr = c
				out[i] = filler
			}
		}
	}
	return out
}

// langExt returns the lowercased file extension (without the dot) for a path,
// or "" if there is none. Package-unique helper name.
func langExt(path string) string {
	for i := len(path) - 1; i >= 0; i-- {
		switch path[i] {
		case '.':
			return strings.ToLower(path[i+1:])
		case '/', '\\':
			return ""
		}
	}
	return ""
}
