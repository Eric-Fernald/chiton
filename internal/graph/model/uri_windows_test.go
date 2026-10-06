//go:build windows

package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestFileURICanonicalCase checks every case spelling of one file, and a missing file in the same directory,
// map to one URI with an upper-case drive letter.
func TestFileURICanonicalCase(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "Mixed.go")
	if err := os.WriteFile(p, []byte("package x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	want := FileURI(p)
	if d := want[len(fileScheme)]; d < 'A' || d > 'Z' {
		t.Fatalf("FileURI(%q) = %q, want an upper-case drive letter", p, want)
	}
	for _, variant := range []string{strings.ToLower(p), strings.ToUpper(p)} {
		if got := FileURI(variant); got != want {
			t.Errorf("FileURI(%q) = %q, want %q", variant, got, want)
		}
	}
	gone := FileURI(filepath.Join(strings.ToUpper(dir), "Gone.go"))
	if w := strings.TrimSuffix(want, "Mixed.go") + "Gone.go"; gone != w {
		t.Errorf("FileURI(missing file) = %q, want %q", gone, w)
	}
}
