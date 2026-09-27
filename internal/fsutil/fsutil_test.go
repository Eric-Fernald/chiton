package fsutil

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"testing"
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
