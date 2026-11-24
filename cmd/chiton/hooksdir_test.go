package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func runGit(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

// samePath reports whether a and b name the same directory, creating both so short and long Windows
// spellings of one temp path compare equal.
func samePath(t *testing.T, a, b string) bool {
	t.Helper()
	for _, p := range []string{a, b} {
		if err := os.MkdirAll(p, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	ai, err := os.Stat(a)
	if err != nil {
		t.Fatal(err)
	}
	bi, err := os.Stat(b)
	if err != nil {
		t.Fatal(err)
	}
	return os.SameFile(ai, bi)
}

// TestResolveGitHooksDirAsksGit checks hooks go where git reads them: the shared hooks directory for a linked
// worktree, and core.hooksPath when a hook manager set one inside the repository.
func TestResolveGitHooksDirAsksGit(t *testing.T) {
	repo := gitRepo(t, map[string]string{"a.go": "package a\n"}, true)

	got, err := resolveGitHooksDir(repo)
	if err != nil || !samePath(t, got, filepath.Join(repo, ".git", "hooks")) {
		t.Errorf("plain repo: hooks dir = %q, %v; want .git/hooks", got, err)
	}

	wt := filepath.Join(t.TempDir(), "wt")
	runGit(t, repo, "worktree", "add", "-q", wt)
	got, err = resolveGitHooksDir(wt)
	if err != nil || !samePath(t, got, filepath.Join(repo, ".git", "hooks")) {
		t.Errorf("linked worktree: hooks dir = %q, %v; want the main repository's .git/hooks, which git runs", got, err)
	}

	runGit(t, repo, "config", "core.hooksPath", ".husky")
	got, err = resolveGitHooksDir(repo)
	if err != nil || !samePath(t, got, filepath.Join(repo, ".husky")) {
		t.Errorf("core.hooksPath inside the repo: hooks dir = %q, %v; want .husky", got, err)
	}
}

// TestResolveGitHooksDirRefusesSharedHooksPath checks chiton won't write hooks into a core.hooksPath outside
// the repository, where they would run for every repository that shares it.
func TestResolveGitHooksDirRefusesSharedHooksPath(t *testing.T) {
	repo := gitRepo(t, map[string]string{"a.go": "package a\n"}, true)
	shared := t.TempDir()
	runGit(t, repo, "config", "core.hooksPath", filepath.ToSlash(shared))
	got, err := resolveGitHooksDir(repo)
	if err == nil || !strings.Contains(err.Error(), "core.hooksPath") {
		t.Errorf("hooks dir = %q, %v; want an error naming core.hooksPath", got, err)
	}
}

func TestResolveGitHooksDirNotARepo(t *testing.T) {
	if _, err := resolveGitHooksDir(t.TempDir()); err == nil {
		t.Error("a directory that is not a git repository resolved a hooks dir")
	}
}
