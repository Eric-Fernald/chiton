package install

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ChitonBlockStart and ChitonBlockEnd delimit the region in any rules file
// (CLAUDE.md, AGENTS.md, copilot-instructions.md, etc.) that chiton owns.
// Anything outside these markers is preserved verbatim across install and
// uninstall, so users can edit the file freely without losing their changes.
const (
	ChitonBlockStart = "<!-- chiton:start -->"
	ChitonBlockEnd   = "<!-- chiton:end -->"
)

// UpsertRulesBlock writes content into path, wrapped in chiton's start/end
// markers. If the file already contains a chiton block, that block is replaced
// in place; otherwise the block is appended (separated by a blank line if the
// existing file is non-empty).
//
// Returns whether the file was created (true) or modified in place (false),
// plus whether a chiton block already existed.
func UpsertRulesBlock(path, content string) (created, preexisted bool, err error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return false, false, fmt.Errorf("mkdir %s: %w", filepath.Dir(path), err)
	}
	existing, readErr := os.ReadFile(path)
	if readErr != nil && !os.IsNotExist(readErr) {
		return false, false, fmt.Errorf("read %s: %w", path, readErr)
	}

	block := ChitonBlockStart + "\n" + strings.TrimRight(content, "\n") + "\n" + ChitonBlockEnd

	var next string
	switch {
	case len(existing) == 0:
		next = block + "\n"
		created = true
	case strings.Contains(string(existing), ChitonBlockStart):
		preexisted = true
		next = replaceBlock(string(existing), block)
	default:
		next = strings.TrimRight(string(existing), "\n") + "\n\n" + block + "\n"
	}

	if err := os.WriteFile(path, []byte(next), 0o644); err != nil {
		return false, false, fmt.Errorf("write %s: %w", path, err)
	}
	return created, preexisted, nil
}

// RemoveRulesBlock removes chiton's owned block from path. If removing the
// block leaves the file empty (or whitespace-only), the file is deleted.
//
// Returns whether anything was changed.
func RemoveRulesBlock(path string) (changed bool, err error) {
	existing, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("read %s: %w", path, err)
	}
	if !strings.Contains(string(existing), ChitonBlockStart) {
		return false, nil
	}
	next := stripBlock(string(existing))
	if strings.TrimSpace(next) == "" {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return true, fmt.Errorf("remove %s: %w", path, err)
		}
		return true, nil
	}
	if err := os.WriteFile(path, []byte(next), 0o644); err != nil {
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

func replaceBlock(text, block string) string {
	startIdx := strings.Index(text, ChitonBlockStart)
	endIdx := strings.Index(text, ChitonBlockEnd)
	if startIdx < 0 || endIdx < 0 || endIdx < startIdx {
		// Malformed: append a fresh block at the end instead of clobbering.
		return strings.TrimRight(text, "\n") + "\n\n" + block + "\n"
	}
	endIdx += len(ChitonBlockEnd)
	return text[:startIdx] + block + text[endIdx:]
}

func stripBlock(text string) string {
	startIdx := strings.Index(text, ChitonBlockStart)
	endIdx := strings.Index(text, ChitonBlockEnd)
	if startIdx < 0 || endIdx < 0 || endIdx < startIdx {
		return text
	}
	endIdx += len(ChitonBlockEnd)
	// Eat one trailing newline so we don't leave a blank line behind.
	if endIdx < len(text) && text[endIdx] == '\n' {
		endIdx++
	}
	// Eat one preceding blank-line separator if we added one on install.
	prefix := text[:startIdx]
	if strings.HasSuffix(prefix, "\n\n") {
		prefix = prefix[:len(prefix)-1]
	}
	return prefix + text[endIdx:]
}
