package analyze

// MaskFiller is the neutral byte maskers write over literal and comment bodies.
const MaskFiller = 'x'

// HasTripleAt reports whether b[i:i+3] is three consecutive q bytes.
func HasTripleAt(b []byte, i int, q byte) bool {
	return i+2 < len(b) && b[i] == q && b[i+1] == q && b[i+2] == q
}

// StripPyComment drops a Python # line comment, ignoring # inside single-line string literals.
func StripPyComment(s string) string {
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
		case '"', '\'':
			inStr = c
		case '#':
			return s[:i]
		}
	}
	return s
}

// StripLineComment cuts s at the first `//` (and, when hash, `#`) outside a ", ' or ` quoted literal,
// keeping literal contents; block comments are not tracked.
func StripLineComment(s string, hash bool) string {
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
			if hash {
				return s[:i]
			}
		}
	}
	return s
}

// MaskCLiterals returns a copy of src with ", ' and ` quoted literals and //, /* */ comments (delimiters
// included, tracked across lines) overwritten by MaskFiller, keeping newlines; rawBackticks disables \ in `.
func MaskCLiterals(src []byte, rawBackticks bool) []byte {
	return maskC(src, rawBackticks, false)
}

// MaskGoLiterals is MaskCLiterals for Go (raw backticks) that also masks a lone `\` before a newline or EOF.
func MaskGoLiterals(src []byte) []byte {
	return maskC(src, true, true)
}

// MaskedCLiterals is MaskCLiterals(t.Content, rawBackticks) computed once per file; never modify it.
func MaskedCLiterals(t FileTarget, rawBackticks bool) []byte {
	key := "MaskCLiterals"
	if rawBackticks {
		key = "MaskCLiterals/raw"
	}
	return Cached(t, key, func() []byte { return MaskCLiterals(t.Content, rawBackticks) })
}

// MaskedGoLiterals is MaskGoLiterals(t.Content) computed once per file; never modify it.
func MaskedGoLiterals(t FileTarget) []byte {
	return Cached(t, "MaskGoLiterals", func() []byte { return MaskGoLiterals(t.Content) })
}

func maskC(src []byte, rawBackticks, maskLoneEscape bool) []byte {
	out := make([]byte, len(src))
	copy(out, src)
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
				out[i] = MaskFiller
			}
		case inBlock:
			if c == '*' && i+1 < len(out) && out[i+1] == '/' {
				out[i] = MaskFiller
				out[i+1] = MaskFiller
				i++
				inBlock = false
			} else if c != '\n' {
				out[i] = MaskFiller
			}
		case inStr != 0:
			if c == '\\' && (inStr != '`' || !rawBackticks) {
				if i+1 < len(out) && out[i+1] != '\n' {
					out[i] = MaskFiller
					out[i+1] = MaskFiller
				} else if maskLoneEscape {
					out[i] = MaskFiller
				}
				i++ // skip escaped char
				continue
			}
			if c == inStr {
				inStr = 0
				out[i] = MaskFiller
			} else if c != '\n' {
				out[i] = MaskFiller
			}
		default:
			switch {
			case c == '/' && i+1 < len(out) && out[i+1] == '/':
				inLine = true
				out[i] = MaskFiller
			case c == '/' && i+1 < len(out) && out[i+1] == '*':
				inBlock = true
				out[i] = MaskFiller
			case c == '"' || c == '\'' || c == '`':
				inStr = c
				out[i] = MaskFiller
			}
		}
	}
	return out
}
