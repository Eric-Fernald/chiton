package install

// jsoncToJSON strips JSONC extensions — // and /* */ comments and trailing
// commas — so encoding/json can parse the comment-laden configs that OpenCode
// and VS Code document for their MCP files. Bytes inside string literals are
// left untouched, so URLs ("http://...") and comment-like text inside values
// survive intact.
func jsoncToJSON(b []byte) []byte {
	out := make([]byte, 0, len(b))
	inStr := false
	for i := 0; i < len(b); i++ {
		c := b[i]
		if inStr {
			out = append(out, c)
			switch {
			case c == '\\' && i+1 < len(b):
				out = append(out, b[i+1])
				i++
			case c == '"':
				inStr = false
			}
			continue
		}
		switch {
		case c == '"':
			inStr = true
			out = append(out, c)
		case c == '/' && i+1 < len(b) && b[i+1] == '/':
			// Line comment: skip to (but keep) the newline.
			for i+1 < len(b) && b[i+1] != '\n' {
				i++
			}
		case c == '/' && i+1 < len(b) && b[i+1] == '*':
			// Block comment: skip through the closing */.
			i += 2
			for i+1 < len(b) && !(b[i] == '*' && b[i+1] == '/') {
				i++
			}
			i++ // land on the closing '/'; the loop's i++ then moves past it
		default:
			out = append(out, c)
		}
	}
	return removeTrailingCommas(out)
}

// removeTrailingCommas drops a comma that is followed (ignoring whitespace) by a
// closing } or ], which encoding/json rejects but JSONC allows. String contents
// are skipped so a comma inside a string is never touched.
func removeTrailingCommas(b []byte) []byte {
	out := make([]byte, 0, len(b))
	inStr := false
	for i := 0; i < len(b); i++ {
		c := b[i]
		if inStr {
			out = append(out, c)
			switch {
			case c == '\\' && i+1 < len(b):
				out = append(out, b[i+1])
				i++
			case c == '"':
				inStr = false
			}
			continue
		}
		if c == '"' {
			inStr = true
			out = append(out, c)
			continue
		}
		if c == ',' {
			j := i + 1
			for j < len(b) && (b[j] == ' ' || b[j] == '\t' || b[j] == '\n' || b[j] == '\r') {
				j++
			}
			if j < len(b) && (b[j] == '}' || b[j] == ']') {
				continue // drop the trailing comma
			}
		}
		out = append(out, c)
	}
	return out
}
