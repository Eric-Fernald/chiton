package git

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseNameStatus(t *testing.T) {
	root := t.TempDir() // absolute; abs() resolves root to absolute either way
	out := "M\tsrc/a.go\n" +
		"A\tnew.go\n" +
		"D\tgone.go\n" +
		"R100\told/x.go\tnew/y.go\n" +
		"C75\tsrc/a.go\tcopy.go\n" +
		"T\tlink.go\n" +
		"\n" // trailing blank line ignored

	changes := parseNameStatus(root, out)

	want := []struct {
		rel string
		op  Op
	}{
		{"src/a.go", OpUpsert},
		{"new.go", OpUpsert},
		{"gone.go", OpDelete},
		{"old/x.go", OpDelete}, // rename: delete old
		{"new/y.go", OpUpsert}, // rename: upsert new
		{"copy.go", OpUpsert},  // copy: upsert dest only
		{"link.go", OpUpsert},
	}
	if len(changes) != len(want) {
		t.Fatalf("got %d changes, want %d: %+v", len(changes), len(want), changes)
	}
	for i, w := range want {
		wantPath := filepath.Join(root, filepath.FromSlash(w.rel))
		if changes[i].Path != wantPath {
			t.Errorf("change %d path = %q, want %q", i, changes[i].Path, wantPath)
		}
		if changes[i].Op != w.op {
			t.Errorf("change %d op = %v, want %v", i, changes[i].Op, w.op)
		}
	}
}

// TestIntegration exercises Available / LastCommitChanges against a throwaway
// git repo. Skipped when git is not on PATH.
func TestIntegration(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	runGit := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}

	runGit("init")
	if !Available(root) {
		t.Fatal("Available should be true inside a fresh repo")
	}
	if err := os.WriteFile(filepath.Join(root, "a.go"), []byte("package a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit("add", "a.go")
	runGit("commit", "-m", "first")

	changes, err := LastCommitChanges(root)
	if err != nil {
		t.Fatalf("LastCommitChanges: %v", err)
	}
	if len(changes) != 1 || changes[0].Op != OpUpsert {
		t.Fatalf("expected 1 upsert for the initial commit, got %+v", changes)
	}
	if filepath.Base(changes[0].Path) != "a.go" {
		t.Errorf("expected a.go, got %s", changes[0].Path)
	}

	// Second commit: delete the file.
	runGit("rm", "a.go")
	runGit("commit", "-m", "remove")
	changes, err = LastCommitChanges(root)
	if err != nil {
		t.Fatalf("LastCommitChanges(2): %v", err)
	}
	if len(changes) != 1 || changes[0].Op != OpDelete {
		t.Fatalf("expected 1 delete, got %+v", changes)
	}

	// ChangedSince HEAD~1 should report the delete too.
	since, err := ChangedSince(root, "HEAD~1")
	if err != nil {
		t.Fatalf("ChangedSince: %v", err)
	}
	if len(since) != 1 || since[0].Op != OpDelete {
		t.Fatalf("ChangedSince expected 1 delete, got %+v", since)
	}
}

// TestMergeCommitChanges verifies LastCommitChanges reports the files a merge
// commit brings onto the branch. A plain `git diff-tree HEAD` is empty for a
// merge, which used to leave every merged-in file stale in the graph.
func TestMergeCommitChanges(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	runGit := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	write := func(name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	runGit("init")
	write("base.go", "package base\n")
	runGit("add", "base.go")
	runGit("commit", "-m", "base")

	// Capture the default branch name (main or master, depending on git config).
	out, err := exec.Command("git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD").Output()
	if err != nil {
		t.Fatalf("rev-parse branch: %v", err)
	}
	base := strings.TrimSpace(string(out))

	// Feature branch adds a file.
	runGit("checkout", "-b", "feature")
	write("feature.go", "package feature\n")
	runGit("add", "feature.go")
	runGit("commit", "-m", "feature work")

	// A divergent commit on base forces a real (non-fast-forward) merge.
	runGit("checkout", base)
	write("mainwork.go", "package mainwork\n")
	runGit("add", "mainwork.go")
	runGit("commit", "-m", "main work")

	// HEAD is now a 2-parent merge commit.
	runGit("merge", "--no-ff", "-m", "merge feature", "feature")

	changes, err := LastCommitChanges(root)
	if err != nil {
		t.Fatalf("LastCommitChanges: %v", err)
	}
	// The merge's net change vs its first parent is the feature file it brings in.
	var gotFeature bool
	for _, c := range changes {
		if filepath.Base(c.Path) == "feature.go" {
			if c.Op != OpUpsert {
				t.Errorf("feature.go op = %v, want upsert", c.Op)
			}
			gotFeature = true
		}
	}
	if !gotFeature {
		t.Fatalf("merge commit did not report feature.go; got %+v", changes)
	}
}

func TestAvailableFalseOutsideRepo(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	if Available(t.TempDir()) {
		t.Error("Available should be false outside a git work tree")
	}
}
