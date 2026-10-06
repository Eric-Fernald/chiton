//go:build unix

package main

import (
	"os"
	"path/filepath"
	"testing"
)

// Anyone who can connect to the socket reads the whole graph, so it must be owner-only whatever the umask.
func TestListenUnixPrivateIsOwnerOnly(t *testing.T) {
	// A short directory keeps the path under macOS's 104-byte socket path limit.
	dir, err := os.MkdirTemp("", "chs")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	path := filepath.Join(dir, "s.sock")

	ln, err := listenUnixPrivate(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o600 {
		t.Errorf("socket mode = %#o, want 0600", got)
	}
}
