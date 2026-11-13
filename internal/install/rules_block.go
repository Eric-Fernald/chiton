package install

import (
	"fmt"
	"os"
	"strings"
)

// ChitonBlockStart and ChitonBlockEnd delimit the region of a rules file chiton owns. Install and
// uninstall preserve everything outside them verbatim.
const (
	ChitonBlockStart = "<!-- chiton:start -->"
	ChitonBlockEnd   = "<!-- chiton:end -->"
)

// UpsertRulesBlock writes content wrapped in chiton's markers into path, replacing an existing block in
// place or appending after a blank line. It reports whether the file was created and a block preexisted.
func UpsertRulesBlock(path, content string) (created, preexisted bool, err error) {
	if err := mkdirParent(path); err != nil {
		return false, false, err
	}
	existing, readErr := os.ReadFile(path)
	if readErr != nil && !os.IsNotExist(readErr) {
		return false, false, fmt.Errorf("read %s: %w", path, readErr)
	}

	block := ChitonBlockStart + "\n" + strings.TrimRight(content, "\n") + "\n" + ChitonBlockEnd

	text := string(existing)
	var next string
	switch {
	case text == "":
		next = block + "\n"
		created = true
	case strings.Contains(text, ChitonBlockStart):
		preexisted = true
		next = replaceBlock(text, block)
	default:
		next = strings.TrimRight(text, "\n") + "\n\n" + block + "\n"
	}

	if err := writeKeepingMode(path, []byte(next), 0o644); err != nil {
		return false, false, fmt.Errorf("write %s: %w", path, err)
	}
	return created, preexisted, nil
}

// RemoveRulesBlock removes chiton's block from path, deleting the file if only whitespace remains.
// It reports whether anything changed.
func RemoveRulesBlock(path string) (changed bool, err error) {
	existing, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("read %s: %w", path, err)
	}
	text := string(existing)
	if !strings.Contains(text, ChitonBlockStart) {
		return false, nil
	}
	next := stripStrayMarkers(stripAllBlocks(text))
	if strings.TrimSpace(next) == "" && !isSymlink(path) { // a symlinked rules file is emptied, not unlinked
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return true, fmt.Errorf("remove %s: %w", path, err)
		}
		return true, nil
	}
	if err := writeKeepingMode(path, []byte(next), 0o644); err != nil {
		return true, fmt.Errorf("write %s: %w", path, err)
	}
	return true, nil
}

// RulesFileHasChiton reports whether path contains a chiton block.
// Missing file → false, nil.
func RulesFileHasChiton(path string) (bool, error) {
	existing, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return strings.Contains(string(existing), ChitonBlockStart), nil
}

// blockSpan returns the byte span of the first well-formed chiton block: the first end marker that follows a
// start marker, paired with the nearest start before it. Text between an unpaired start and that block is the
// user's, so it stays outside the span.
func blockSpan(text string) (start, end int, ok bool) {
	first := strings.Index(text, ChitonBlockStart)
	if first < 0 {
		return 0, 0, false
	}
	rel := strings.Index(text[first:], ChitonBlockEnd)
	if rel < 0 {
		return 0, 0, false
	}
	end = first + rel
	start = strings.LastIndex(text[:end], ChitonBlockStart)
	return start, end + len(ChitonBlockEnd), true
}

// replaceBlock puts block where the first well-formed chiton block is, or appends it when there is none. Any
// other block and any unpaired marker goes, so one block remains however the file was hand-edited.
func replaceBlock(text, block string) string {
	start, end, ok := blockSpan(text)
	if !ok {
		return strings.TrimRight(stripStrayMarkers(text), "\n") + "\n\n" + block + "\n"
	}
	return stripStrayMarkers(text[:start]) + block + stripStrayMarkers(stripAllBlocks(text[end:]))
}

// stripStrayMarkers removes unpaired marker lines from text that holds no well-formed block, keeping the
// text around them: what follows an orphan start marker may be the user's.
func stripStrayMarkers(text string) string {
	for _, m := range []string{ChitonBlockStart, ChitonBlockEnd} {
		text = strings.ReplaceAll(text, m+"\n", "")
		text = strings.ReplaceAll(text, m, "")
	}
	return text
}

// stripAllBlocks removes every well-formed chiton block from text, leaving all
// surrounding user content intact.
func stripAllBlocks(text string) string {
	for {
		next := stripBlock(text)
		if next == text {
			return text // no well-formed block is left
		}
		text = next
	}
}

func stripBlock(text string) string {
	start, end, ok := blockSpan(text)
	if !ok {
		return text
	}
	// Eat one trailing newline so we don't leave a blank line behind.
	if end < len(text) && text[end] == '\n' {
		end++
	}
	// Eat one preceding blank-line separator if we added one on install.
	prefix := text[:start]
	if strings.HasSuffix(prefix, "\n\n") {
		prefix = prefix[:len(prefix)-1]
	}
	return prefix + text[end:]
}
