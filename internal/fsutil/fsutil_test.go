package fsutil

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"time"
)

// TestWriteFileAtomic checks content and that each write leaves the mode os.WriteFile gives a new file
// (perm minus the umask), even when replacing a file of another mode.
func TestWriteFileAtomic(t *testing.T) {
	dir, refDir := t.TempDir(), t.TempDir()
	path := filepath.Join(dir, "out.json")

	for i, tc := range []struct {
		data string
		perm os.FileMode
	}{{"first\n", 0o644}, {"second, longer\n", 0o600}, {"", 0o640}, {"last", 0o644}} {
		if err := WriteFileAtomic(path, []byte(tc.data), tc.perm); err != nil {
			t.Fatal(err)
		}
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != tc.data {
			t.Errorf("content = %q, want %q", got, tc.data)
		}
		ref := filepath.Join(refDir, strconv.Itoa(i))
		if err := os.WriteFile(ref, []byte(tc.data), tc.perm); err != nil {
			t.Fatal(err)
		}
		if gotMode, wantMode := mode(t, path), mode(t, ref); gotMode != wantMode {
			t.Errorf("perm %v: mode = %v, want %v (as os.WriteFile)", tc.perm, gotMode, wantMode)
		}
	}
	assertOnly(t, dir, "out.json")
}

func mode(t *testing.T, path string) os.FileMode {
	t.Helper()
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return fi.Mode().Perm()
}

func TestWriteFileAtomicMissingDir(t *testing.T) {
	dir := t.TempDir()
	if err := WriteFileAtomic(filepath.Join(dir, "no", "such", "f"), []byte("x"), 0o644); err == nil {
		t.Fatal("want an error for a missing parent directory")
	}
	assertOnly(t, dir)
}

// Concurrent writers each use their own temp file, so every write lands whole and none leaks a temp file.
func TestWriteFileAtomicConcurrent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cache.json")
	payloads := []string{"aaaa", "bbbbbbbb", "cc"}
	var wg sync.WaitGroup
	for i := range 30 {
		wg.Go(func() {
			if err := WriteFileAtomic(path, []byte(payloads[i%len(payloads)]), 0o644); err != nil {
				// Windows may refuse a rename while another writer's rename holds the target.
				if runtime.GOOS != "windows" {
					t.Error(err)
				}
			}
		})
	}
	wg.Wait()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	ok := false
	for _, p := range payloads {
		ok = ok || string(got) == p
	}
	if !ok {
		t.Errorf("torn content %q", got)
	}
	assertOnly(t, dir, "cache.json")
}

// assertOnly fails unless dir holds exactly names.
func assertOnly(t *testing.T, dir string, names ...string) {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, e := range ents {
		got = append(got, e.Name())
	}
	if len(got) != len(names) {
		t.Fatalf("dir holds %v, want %v", got, names)
	}
	for i := range names {
		if got[i] != names[i] {
			t.Fatalf("dir holds %v, want %v", got, names)
		}
	}
}

// On Windows, replacing a file another handle briefly holds open must succeed once the handle closes.
func TestWriteFileAtomicRetriesHeldFileOnWindows(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("rename-over-open-file only fails on Windows")
	}
	path := filepath.Join(t.TempDir(), "cluster.json")
	if err := os.WriteFile(path, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	held, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		time.Sleep(100 * time.Millisecond)
		held.Close()
	}()
	if err := WriteFileAtomic(path, []byte("new"), 0o644); err != nil {
		t.Fatalf("WriteFileAtomic over a briefly held file: %v", err)
	}
	if got, _ := os.ReadFile(path); string(got) != "new" {
		t.Fatalf("content = %q, want %q", got, "new")
	}
}

func TestResolveWithin(t *testing.T) {
	root, outside := t.TempDir(), t.TempDir()
	pkg := filepath.Join(root, "pkg")
	if err := os.Mkdir(pkg, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		in, want string // want "" means p must be refused
	}{
		{".", root},
		{"pkg", pkg},
		{pkg, pkg},
		{filepath.Join("pkg", ".."), root},
		{"..", ""},
		{filepath.Join("pkg", "..", ".."), ""},
		{outside, ""},
		{"missing", ""},
	} {
		got, err := ResolveWithin(root, tc.in)
		if tc.want == "" {
			if err == nil {
				t.Errorf("ResolveWithin(%q) = %q, want an error", tc.in, got)
			}
			continue
		}
		if err != nil || got != tc.want {
			t.Errorf("ResolveWithin(%q) = %q, %v; want %q", tc.in, got, err, tc.want)
		}
	}

	if err := os.Symlink(outside, filepath.Join(root, "escape")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if got, err := ResolveWithin(root, "escape"); err == nil {
		t.Errorf("ResolveWithin followed a symlink out of root to %q", got)
	}
}
