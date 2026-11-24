package git

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestStagedFiles checks StagedFiles returns index content, not the working copy, and skips deletions,
// untracked files, and paths keep rejects.
func TestStagedFiles(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	runGit := gitRunner(t, root)
	write := func(name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runGit("init")
	write("gone.go", "package p\n")
	runGit("add", "gone.go")
	runGit("commit", "-m", "base")

	write("a.go", "package p // staged\n")
	write("notes.txt", "staged text\n")
	runGit("add", "a.go", "notes.txt")
	runGit("rm", "-q", "gone.go")
	write("a.go", "package p // working copy\n")
	write("untracked.go", "package p\n")

	files, err := StagedFiles(context.Background(), root, func(p string) bool { return filepath.Ext(p) == ".go" })
	if err != nil {
		t.Fatalf("StagedFiles: %v", err)
	}
	if len(files) != 1 || filepath.Base(files[0].Path) != "a.go" {
		t.Fatalf("want only a.go, got %+v", files)
	}
	if got := string(files[0].Content); got != "package p // staged\n" {
		t.Errorf("content = %q, want the staged blob", got)
	}
}
