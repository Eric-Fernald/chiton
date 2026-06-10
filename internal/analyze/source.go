package analyze

import "strings"

// IsTestSource reports whether a source URI (or path) points at a test file:
// Go *_test.go, JS/TS *.test.* / *.spec.*, or anything under a tests/ dir.
// Shared by rule packages so "skip test fixtures" behaves consistently.
func IsTestSource(sourceURI string) bool {
	low := strings.ToLower(sourceURI)
	switch {
	case strings.HasSuffix(low, "_test.go"):
		return true
	case strings.Contains(low, ".test."), strings.Contains(low, ".spec."):
		return true
	case strings.Contains(low, "/tests/"), strings.Contains(low, "\\tests\\"):
		return true
	}
	return false
}
