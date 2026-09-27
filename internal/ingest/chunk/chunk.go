// Package chunk splits ingested text into paragraph-packed sections for doc_section nodes.
package chunk

import "strings"

// Paragraphs packs "\n\n"-separated paragraphs into trimmed chunks of at most size (> 0) bytes, returning text
// whole when it fits. A lone paragraph may exceed size. With maxChunks > 0 the last chunk absorbs the rest.
func Paragraphs(text string, size, maxChunks int) []string {
	if len(text) <= size {
		return []string{text}
	}
	var chunks []string
	var cur strings.Builder
	// atCap means all but the final chunk exist, so remaining paragraphs accumulate into the last one.
	atCap := func() bool { return maxChunks > 0 && len(chunks) >= maxChunks-1 }
	for _, p := range strings.Split(text, "\n\n") {
		if cur.Len() > 0 && cur.Len()+len(p)+2 > size && !atCap() {
			chunks = append(chunks, strings.TrimSpace(cur.String()))
			cur.Reset()
		}
		if cur.Len() > 0 {
			cur.WriteString("\n\n")
		}
		cur.WriteString(p)
	}
	if rest := strings.TrimSpace(cur.String()); rest != "" {
		chunks = append(chunks, rest)
	}
	if len(chunks) == 0 {
		return []string{text}
	}
	return chunks
}
