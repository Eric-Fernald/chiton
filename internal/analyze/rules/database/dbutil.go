package database

import (
	"bytes"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
)

// stripDBComment cuts s at a line comment outside quotes (`//`, `--`, and `#` when py) and blanks
// single-line /* */ spans; string contents are never touched, since the SQL rules match inside them.
func stripDBComment(s string, py bool) string {
	var b []byte // copy of s, made only once a /* */ span needs blanking
	cut := func(i int) string {
		if b == nil {
			return s[:i]
		}
		return string(b[:i])
	}
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
				return cut(i)
			}
			if i+1 < len(s) && s[i+1] == '*' {
				// Single-line block comment: blank everything through "*/" if it
				// closes on this line, otherwise blank to end of line.
				end := strings.Index(s[i+2:], "*/")
				if end < 0 {
					return cut(i)
				}
				if b == nil {
					b = []byte(s)
				}
				for j := i; j < i+2+end+2; j++ {
					b[j] = 'x'
				}
				i = i + 2 + end + 1
			}
		case '-':
			if i+1 < len(s) && s[i+1] == '-' {
				return cut(i)
			}
		case '#':
			if py {
				return cut(i)
			}
		}
	}
	if b == nil {
		return s
	}
	return string(b)
}

// maskedDB is maskDBBlocks(t.Content, py) computed once per file, shared by every database rule.
func maskedDB(t analyze.FileTarget, py bool) []byte {
	key := "database/maskDBBlocks"
	if py {
		key += "/py"
	}
	return analyze.Cached(t, key, func() []byte { return maskDBBlocks(t.Content, py) })
}

// maskDBBlocks overwrites /* */ comments and, when py, triple-quoted strings with 'x' (newlines kept),
// leaving other strings intact for the SQL rules; it may return src itself, so never modify the result.
func maskDBBlocks(src []byte, py bool) []byte {
	if !bytes.Contains(src, []byte("/*")) &&
		(!py || !bytes.Contains(src, []byte(`"""`)) && !bytes.Contains(src, []byte("'''"))) {
		return src
	}
	out := bytes.Clone(src)
	const filler = 'x'
	inBlock := false  // inside /* ... */
	triple := byte(0) // quote char while inside a triple-quoted string
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
				triple = c
				out[i] = filler
				out[i+1] = filler
				out[i+2] = filler
				i += 2
			case c == '"' || c == '\'' || c == '`':
				inStr = c
			}
		}
	}
	return out
}
