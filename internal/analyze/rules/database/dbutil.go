package database

import (
	"path"
	"strings"
)

// stripDBComment removes the line-comment tail from a source line, respecting
// string literals so a comment marker inside a quoted string is preserved.
// It strips the comment styles relevant to the file's language:
//   - "//" for go/js/ts/java
//   - "#" for python
//   - "--" for SQL (recognised in every language, since these rules scan SQL
//     embedded in string literals — but only outside of quotes)
//
// It also blanks out single-line "/* ... */" block-comment spans (the marker
// opens and closes on the same line) so a keyword inside such a comment does not
// match. Block comments that span multiple lines are handled separately by
// maskDBBlocks, which the rules apply as a per-file pass before scanning.
func stripDBComment(s, uri string) string {
	ext := strings.ToLower(strings.TrimPrefix(path.Ext(uri), "."))
	hash := ext == "py"
	b := []byte(s)
	var inStr byte
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
			if i+1 < len(b) && b[i+1] == '/' {
				return string(b[:i])
			}
			if i+1 < len(b) && b[i+1] == '*' {
				// Single-line block comment: blank everything through "*/" if it
				// closes on this line, otherwise blank to end of line.
				end := strings.Index(string(b[i+2:]), "*/")
				if end < 0 {
					return string(b[:i])
				}
				for j := i; j < i+2+end+2; j++ {
					b[j] = 'x'
				}
				i = i + 2 + end + 1
			}
		case '-':
			if i+1 < len(b) && b[i+1] == '-' {
				return string(b[:i])
			}
		case '#':
			if hash {
				return string(b[:i])
			}
		}
	}
	return string(b)
}

// maskDBBlocks returns a copy of src with the bodies of multi-line constructs
// blanked to the filler byte 'x' (newlines preserved so line numbers stay
// accurate). This is a per-FILE pass that tracks open/close across lines:
//   - "/* ... */" block comments (all languages)
//   - Python triple-quoted strings (”'...”' and """...""") in .py files,
//     which are routinely used as docstrings and may mention SQL keywords.
//
// Single-quoted and double-quoted single-line string literals are left intact
// here — the rules need their contents to detect embedded SQL — they are
// handled per line by stripDBComment instead.
func maskDBBlocks(src []byte, uri string) []byte {
	ext := strings.ToLower(strings.TrimPrefix(path.Ext(uri), "."))
	py := ext == "py"
	out := make([]byte, len(src))
	copy(out, src)
	const filler = 'x'
	inBlock := false  // inside /* ... */
	var triple string // "" or `'''` or `"""` when inside a triple-quoted string
	var inStr byte    // 0, '"' or '\'' for single-line string (so /* inside a string is ignored)
	for i := 0; i < len(out); i++ {
		c := out[i]
		switch {
		case inBlock:
			if c == '*' && i+1 < len(out) && out[i+1] == '/' {
				out[i] = filler
				out[i+1] = filler
				i++
				inBlock = false
			} else if c != '\n' {
				out[i] = filler
			}
		case triple != "":
			if i+len(triple) <= len(out) && string(out[i:i+len(triple)]) == triple {
				for j := i; j < i+len(triple); j++ {
					out[j] = filler
				}
				i += len(triple) - 1
				triple = ""
			} else if c != '\n' {
				out[i] = filler
			}
		case inStr != 0:
			if c == '\\' && i+1 < len(out) {
				i++ // skip escaped char (do not blank — single-line strings stay intact)
				continue
			}
			if c == inStr || c == '\n' {
				inStr = 0
			}
		default:
			switch {
			case c == '/' && i+1 < len(out) && out[i+1] == '*':
				inBlock = true
				out[i] = filler
			case py && (c == '\'' || c == '"') && i+2 < len(out) && out[i+1] == c && out[i+2] == c:
				triple = string([]byte{c, c, c})
				for j := i; j < i+3; j++ {
					out[j] = filler
				}
				i += 2
			case c == '"' || c == '\'' || c == '`':
				inStr = c
			}
		}
	}
	return out
}
