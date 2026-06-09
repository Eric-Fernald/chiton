package bugs

// stripPyComment drops the content of a Python `#` line comment, ignoring `#`
// characters that appear inside string literals. Block strings spanning lines
// are not tracked (single-line approximation), matching the other strip helpers
// in this package. Unique to the bugs package's new Python rules.
func stripPyComment(s string) string {
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
