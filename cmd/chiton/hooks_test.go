package main

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	chitoncrypto "github.com/Eric-Fernald/chiton/internal/crypto"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
	"github.com/spf13/cobra"
)

// gitRepo creates a repository holding files, staged; commit also commits them.
func gitRepo(t *testing.T, files map[string]string, commit bool) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	git("init")
	for name, body := range files {
		p := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		git("add", filepath.ToSlash(name))
	}
	if commit {
		git("commit", "-m", "c")
	}
	return dir
}

// unavailableStore points config at a store a hook cannot open: held open (locked, as by a running
// `sync watch`) or beside a keyfile with no passphrase available.
func unavailableStore(t *testing.T, kind string) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("CHITON_PROJECT_DIR", "")
	t.Setenv(chitoncrypto.EnvVarPassphrase, "")
	graph := filepath.Join(t.TempDir(), "graph")
	t.Setenv("CHITON_GRAPH_PATH", graph)
	switch kind {
	case "locked":
		gs, err := badgerstore.Open(graph)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { gs.Close() })
	case "encrypted":
		if err := os.MkdirAll(graph, 0o755); err != nil {
			t.Fatal(err)
		}
		kf, err := chitoncrypto.DefaultKeyFile()
		if err != nil {
			t.Fatal(err)
		}
		if err := kf.Save(filepath.Join(filepath.Dir(graph), chitoncrypto.KeyfileName), false); err != nil {
			t.Fatal(err)
		}
	}
}

func execute(t *testing.T, cmd *cobra.Command, args ...string) (stderr string, err error) {
	t.Helper()
	var out, errOut bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&errOut)
	cmd.SetArgs(args)
	err = cmd.ExecuteContext(context.Background())
	return errOut.String(), err
}

// TestHooksRunFailsOpen checks an unusable store or policy file never blocks a commit, while a staged
// finding still does.
func TestHooksRunFailsOpen(t *testing.T) {
	for _, kind := range []string{"locked", "encrypted"} {
		t.Run(kind, func(t *testing.T) {
			unavailableStore(t, kind)
			repo := gitRepo(t, map[string]string{"a.go": "package a\n\nfunc A() {}\n", ".chiton/hooks.yaml": "pre-commit: ["}, false)
			stderr, err := execute(t, hooksRunCmd(), "--stage", "pre-commit", "--dir", repo)
			if err != nil {
				t.Fatalf("hooks run blocked a clean commit: %v\n%s", err, stderr)
			}
			for _, want := range []string{"using the default policy", "findings not recorded"} {
				if !strings.Contains(stderr, want) {
					t.Errorf("stderr lacks %q:\n%s", want, stderr)
				}
			}

			leaky := gitRepo(t, map[string]string{"k.go": "package a\n\nconst Key = \"AKIAIOSFODNN7EXAMPLE\"\n"}, false)
			if _, err := execute(t, hooksRunCmd(), "--stage", "pre-commit", "--dir", leaky); err == nil {
				t.Error("a staged critical finding should still block")
			}
		})
	}
}

// TestSyncGitHookFailsOpen checks the post-commit hook warns instead of failing when the store is locked.
func TestSyncGitHookFailsOpen(t *testing.T) {
	unavailableStore(t, "locked")
	repo := gitRepo(t, map[string]string{"a.go": "package a\n"}, true)
	stderr, err := execute(t, syncGitCmd(), "--stage", "post-commit", "--root", repo)
	if err != nil || !strings.Contains(stderr, "graph not updated") {
		t.Fatalf("sync git hook = %v, stderr %q; want a warning and exit 0", err, stderr)
	}
}
