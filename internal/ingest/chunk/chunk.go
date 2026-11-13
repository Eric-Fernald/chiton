// Package chunk splits ingested text into paragraph-packed sections for doc_section nodes.
package chunk

import (
	"strings"
	"unicode/utf8"
)

// Paragraphs packs "\n\n"-separated paragraphs into trimmed chunks of at most size (> 0) bytes, hard-splitting a
// longer paragraph, and returns text whole when it fits. With maxChunks > 0 it drops text past that many chunks
// and reports true.
func Paragraphs(text string, size, maxChunks int) ([]string, bool) {
	if len(text) <= size {
		return []string{text}, false
	}
	var chunks []string
	var cur strings.Builder
	flush := func() {
		if c := strings.TrimSpace(cur.String()); c != "" {
			chunks = append(chunks, c)
		}
		cur.Reset()
	}
	for _, p := range strings.Split(text, "\n\n") {
		if cur.Len() > 0 && cur.Len()+len(p)+2 > size {
			flush()
		}
		// Only the last piece of a split paragraph is short enough to pack with what follows.
		for len(p) > size {
			cut := splitPoint(p, size)
			cur.WriteString(p[:cut])
			flush()
			p = p[cut:]
		}
		if cur.Len() > 0 {
			cur.WriteString("\n\n")
		}
		cur.WriteString(p)
	}
	flush()
	if maxChunks > 0 && len(chunks) > maxChunks {
		return chunks[:maxChunks], true
	}
	return chunks, false
}

// splitPoint returns where to cut s (longer than size) so the head is at most size bytes: after the last
// newline in reach, else after the last space, else at a rune boundary. It is always > 0.
func splitPoint(s string, size int) int {
	if i := strings.LastIndexByte(s[:size], '\n'); i >= 0 {
		return i + 1
	}
	if i := strings.LastIndexByte(s[:size], ' '); i >= 0 {
		return i + 1
	}
	return runeBoundary(s, size)
}

// runeBoundary returns the largest index <= limit that does not split a UTF-8
// rune. It always returns > 0 so callers make progress.
func runeBoundary(s string, limit int) int {
	if limit >= len(s) {
		return len(s)
	}
	i := limit
	for i > 0 && !utf8.RuneStart(s[i]) {
		i--
	}
	if i == 0 {
		return limit // no boundary found (pathological), so force progress
	}
	return i
}
