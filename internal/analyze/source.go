package analyze

import (
	"iter"
	"path/filepath"
	"strings"
	"sync"
)

// SourceLangs is the Meta.Languages list for a rule that reads any source code: every extension with a code
// extractor, so not Markdown or images. Rules share it, so never mutate it.
var SourceLangs = []string{"go", "py", "js", "ts", "jsx", "tsx", "rs", "java"}

// IsTestSource reports whether a source URI (or path) points at a test file: Go *_test.go,
// JS/TS *.test.* or *.spec.*, or anything under a tests/ dir. Rule packages share it so they agree.
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

// FileExt returns path's lowercase extension without the leading dot, or "" if it has none.
func FileExt(path string) string {
	return strings.ToLower(strings.TrimPrefix(filepath.Ext(path), "."))
}

// PathBase returns the part of p after its last '/' or '\', or p itself when it has neither.
func PathBase(p string) string {
	if i := strings.LastIndexAny(p, "/\\"); i >= 0 {
		return p[i+1:]
	}
	return p
}

// IsFuncKind reports whether an extractor lang_kind names a function-like symbol.
func IsFuncKind(langKind string) bool {
	switch langKind {
	case "func", "method", "arrow":
		return true
	}
	return false
}

// maxLineBytes is the bufio.Scanner token limit line-based rules historically used.
const maxLineBytes = 1024 * 1024

// Lines yields each 1-based line of content exactly as a 1 MiB-limited bufio.ScanLines Scanner would (stopping
// at the first line that won't fit).
func Lines(content []byte) iter.Seq2[int, string] { return LinesOf(string(content)) }

// FileLines is Lines(t.Content) with the content converted to a string once per file and shared by every rule.
func FileLines(t FileTarget) iter.Seq2[int, string] {
	return LinesOf(Cached(t, "text", func() string { return string(t.Content) }))
}

// LinesOf is Lines over a string, so a caller already holding (or caching) one doesn't copy it again.
func LinesOf(s string) iter.Seq2[int, string] {
	return func(yield func(int, string) bool) {
		src := s // per-iteration cursor, so the sequence can be ranged more than once
		for n := 1; src != ""; n++ {
			line, rest, _ := strings.Cut(src, "\n")
			if len(line) >= maxLineBytes {
				return
			}
			if !yield(n, strings.TrimSuffix(line, "\r")) {
				return
			}
			src = rest
		}
	}
}

// fileMemo caches per-file derived data, such as masked content, for Cached.
type fileMemo struct {
	mu sync.Mutex
	m  map[string]any
}

// Cached returns fn() memoized under key for the file t was loaded from, so rules sharing a masker run it
// once per file. Never modify the shared result. Hand-built targets have no memo, so fn runs on every call.
func Cached[T any](t FileTarget, key string, fn func() T) T {
	if t.memo == nil {
		return fn()
	}
	t.memo.mu.Lock()
	v, ok := t.memo.m[key]
	t.memo.mu.Unlock()
	if ok {
		return v.(T)
	}
	r := fn() // runs without the lock so fn may itself call Cached
	t.memo.mu.Lock()
	t.memo.m[key] = r
	t.memo.mu.Unlock()
	return r
}
