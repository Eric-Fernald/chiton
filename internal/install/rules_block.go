package install

import (
	"fmt"
	"os"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/fsutil"
)

// ChitonBlockStart and ChitonBlockEnd delimit the region of a rules file chiton owns; everything outside
// them is preserved verbatim across install and uninstall.
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

	if err := fsutil.WriteFileAtomic(path, []byte(next), 0o644); err != nil {
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
	next := stripBlock(text)
	if strings.TrimSpace(next) == "" {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return true, fmt.Errorf("remove %s: %w", path, err)
		}
		return true, nil
	}
	if err := fsutil.WriteFileAtomic(path, []byte(next), 0o644); err != nil {
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

// blockSpan returns the byte span of the first chiton block; ok is false if a marker is missing or the
// first end marker precedes the first start marker.
func blockSpan(text string) (start, end int, ok bool) {
	start = strings.Index(text, ChitonBlockStart)
	end = strings.Index(text, ChitonBlockEnd)
	if start < 0 || end < 0 || end < start {
		return 0, 0, false
	}
	return start, end + len(ChitonBlockEnd), true
}

func replaceBlock(text, block string) string {
	start, end, ok := blockSpan(text)
	if !ok {
		// Malformed (e.g. a stray end marker before any start): strip any partial
		// chiton regions and append a fresh block instead of clobbering user text.
		return strings.TrimRight(stripAllBlocks(text), "\n") + "\n\n" + block + "\n"
	}
	// Strip any further blocks from the tail so a duplicated block cannot survive the replacement.
	return text[:start] + block + stripAllBlocks(text[end:])
}

// stripAllBlocks removes every well-formed chiton block from text, leaving all
// surrounding user content intact.
func stripAllBlocks(text string) string {
	for {
		next := stripBlock(text)
		if next == text {
			return text // no block left, or malformed ordering
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
