package walk

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
)

// writeTree creates each slash-separated path under root with the given content.
func writeTree(t *testing.T, root string, files map[string]string) {
	t.Helper()
	for rel, body := range files {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// collect returns the root-relative, slash-separated files Files yields, sorted.
func collect(t *testing.T, root string, p *Policy) []string {
	t.Helper()
	var got []string
	if _, err := Files(context.Background(), root, p, func(path string) error {
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		got = append(got, filepath.ToSlash(rel))
		return nil
	}, nil); err != nil {
		t.Fatal(err)
	}
	slices.Sort(got)
	return got
}

func TestFilesAppliesDefaultsAndChitonignore(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, map[string]string{
		"main.go":               "",
		"node_modules/dep/x.js": "",
		".venv/lib/site.py":     "",
		"pkg/__pycache__/m.pyc": "",
		"target/debug/b.rs":     "",
		"vendor/v.go":           "",
		"gen/out.go":            "",
		"sub/gen/keep.go":       "",
		"docs/api/ref.md":       "",
		"docs/guide.md":         "",
		"a.pb.go":               "",
		"sub/b.pb.go":           "",
		IgnoreFile:              "# generated\n*.pb.go\n/gen/\ndocs/api\n!vendor\n\n[bad\n",
	})
	p, err := LoadPolicy(root)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{IgnoreFile, "docs/guide.md", "main.go", "sub/gen/keep.go", "vendor/v.go"}
	if got := collect(t, root, p); !slices.Equal(got, want) {
		t.Errorf("Files = %v, want %v", got, want)
	}
}

func TestPolicySkip(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, map[string]string{IgnoreFile: "build/\n"})
	p, err := LoadPolicy(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		rel   string
		isDir bool
		want  bool
	}{
		{"main.go", false, false},
		{"node_modules/x/index.js", false, true},
		{"pkg/.git", true, true},
		{"build", true, true},
		{"build", false, false}, // a trailing "/" matches directories only
		{"cmd/build/main.go", false, true},
		{".", true, false},
		{"../outside.go", false, false},
	} {
		if got := p.Skip(filepath.Join(root, filepath.FromSlash(tc.rel)), tc.isDir); got != tc.want {
			t.Errorf("Skip(%q, %v) = %v, want %v", tc.rel, tc.isDir, got, tc.want)
		}
	}
}

func TestFilesSkipsSymlinks(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, map[string]string{"notes.md": "# notes\n"})
	secret := filepath.Join(t.TempDir(), "id_rsa")
	if err := os.WriteFile(secret, []byte("PRIVATE KEY\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(secret, filepath.Join(root, "key.md")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if err := os.Symlink(filepath.Dir(secret), filepath.Join(root, "outside")); err != nil {
		t.Fatal(err)
	}
	p, err := LoadPolicy(root)
	if err != nil {
		t.Fatal(err)
	}
	if got := collect(t, root, p); !slices.Equal(got, []string{"notes.md"}) {
		t.Errorf("Files = %v, want only notes.md (symlinks skipped)", got)
	}
}

func TestFilesSkipsUnreadableDirs(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("chmod cannot make a directory unreadable on Windows")
	}
	if os.Geteuid() == 0 {
		t.Skip("root can read any directory")
	}
	root := t.TempDir()
	writeTree(t, root, map[string]string{"a.go": "", "locked/b.go": ""})
	locked := filepath.Join(root, "locked")
	if err := os.Chmod(locked, 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(locked, 0o755) })
	p, err := LoadPolicy(root)
	if err != nil {
		t.Fatal(err)
	}
	var got, failed []string
	st, err := Files(context.Background(), root, p, func(path string) error {
		got = append(got, filepath.Base(path))
		return nil
	}, func(path string, _ error) { failed = append(failed, path) })
	if err != nil {
		t.Fatalf("Files aborted on an unreadable dir: %v", err)
	}
	if st.Unreadable != 1 || !slices.Equal(failed, []string{locked}) {
		t.Errorf("Unreadable = %d, onErr paths = %v; want 1, [%s]", st.Unreadable, failed, locked)
	}
	if !slices.Equal(got, []string{"a.go"}) {
		t.Errorf("Files = %v, want [a.go]", got)
	}
}

func TestFilesStopsOnCancelAndMissingRoot(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, map[string]string{"a.go": ""})
	p, err := LoadPolicy(root)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Files(ctx, root, p, func(string) error { return nil }, nil); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled Files err = %v, want context.Canceled", err)
	}
	missing := filepath.Join(root, "nope")
	mp, err := LoadPolicy(missing)
	if err != nil {
		t.Fatalf("LoadPolicy(missing root) = %v, want the default policy", err)
	}
	if _, err := Files(context.Background(), missing, mp, func(string) error { return nil }, nil); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("missing root err = %v, want fs.ErrNotExist", err)
	}
}
