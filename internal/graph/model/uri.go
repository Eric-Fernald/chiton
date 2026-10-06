package model

import (
	"path/filepath"
	"strings"
)

const fileScheme = "file://"

// FileURI returns the canonical file:// source URI for path: absolute, forward slashes, and on Windows the on-disk
// case of every existing component with symlinked or junction directories resolved. A non-absolutable path is kept.
func FileURI(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		return fileScheme + filepath.ToSlash(path)
	}
	return fileScheme + filepath.ToSlash(canonicalPath(abs))
}

// PathFromFileURI inverts FileURI into an OS path. ok is false (and path "") when uri isn't a file:// URI.
func PathFromFileURI(uri string) (path string, ok bool) {
	rest, ok := strings.CutPrefix(uri, fileScheme)
	if !ok {
		return "", false
	}
	return filepath.FromSlash(rest), true
}

// Slug lowercases s and collapses each run of characters other than ASCII letters and digits to one hyphen,
// trimming hyphens at the ends, and returns fallback when nothing is left. Node IDs depend on its output.
func Slug(s, fallback string) string {
	var b strings.Builder
	lastHyphen := false
	for _, r := range strings.ToLower(s) {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'):
			b.WriteRune(r)
			lastHyphen = false
		default:
			if !lastHyphen {
				b.WriteByte('-')
				lastHyphen = true
			}
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return fallback
	}
	return out
}
