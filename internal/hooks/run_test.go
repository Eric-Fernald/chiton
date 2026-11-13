package hooks

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules" // register rules
)

const leak = "package a\n\nconst Key = \"AKIAIOSFODNN7EXAMPLE\"\n"

// gitIn runs git in dir with a fixed identity, failing t on error.
func gitIn(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
	cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestRunPreCommitGatesStagedContent checks the pre-commit gate judges the staged blobs: a staged secret
// blocks even after the working copy drops it, and unstaged or untracked secrets never block.
func TestRunPreCommitGatesStagedContent(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	gitIn(t, repo, "init")
	path := filepath.Join(repo, "leak.go")
	writeFile(t, path, leak)
	gitIn(t, repo, "add", "leak.go")
	writeFile(t, path, "package a\n")
	opts := RunOptions{Stage: StagePreCommit, Config: Default().PreCommit, Root: repo}

	res, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !res.Blocks() {
		t.Fatalf("staged secret should block: %+v", res)
	}

	gitIn(t, repo, "add", "leak.go")
	writeFile(t, path, leak)
	writeFile(t, filepath.Join(repo, "wip.go"), leak)
	if res, err = Run(context.Background(), opts); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Blocks() || res.FilesScanned != 1 {
		t.Errorf("clean index: FilesScanned=%d, blocking=%d; want 1 file, 0 blocking", res.FilesScanned, res.BlockingCount)
	}
}

// TestRunPreCommitSkipsIgnoredPaths checks staged files under an ignored directory are not gated, matching
// the tree walk pre-push and `chiton analyze` use.
func TestRunPreCommitSkipsIgnoredPaths(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	gitIn(t, repo, "init")
	if err := os.Mkdir(filepath.Join(repo, "vendor"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(repo, "vendor", "leak.go"), leak)
	writeFile(t, filepath.Join(repo, "ok.go"), "package a\n")
	gitIn(t, repo, "add", "vendor/leak.go", "ok.go")

	res, err := Run(context.Background(), RunOptions{Stage: StagePreCommit, Config: Default().PreCommit, Root: repo})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Blocks() || res.FilesScanned != 1 {
		t.Errorf("FilesScanned=%d, blocking=%d; want 1 file (vendor/ skipped), 0 blocking", res.FilesScanned, res.BlockingCount)
	}
}

// TestRunPreCommitReportsUnparsableStagedFile checks a staged file that cannot be parsed is named in the hook
// output with the reason, without blocking the commit.
func TestRunPreCommitReportsUnparsableStagedFile(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	gitIn(t, repo, "init")
	writeFile(t, filepath.Join(repo, "ok.go"), "package a\n")
	writeFile(t, filepath.Join(repo, "bad.go"), "package a\n\nfunc {\n")
	gitIn(t, repo, "add", "ok.go", "bad.go")

	res, err := Run(context.Background(), RunOptions{Stage: StagePreCommit, Config: Default().PreCommit, Root: repo})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Blocks() || res.FilesScanned != 1 || len(res.Skipped) != 1 {
		t.Fatalf("blocking=%d FilesScanned=%d Skipped=%v; want 0 blocking, ok.go scanned, bad.go skipped",
			res.BlockingCount, res.FilesScanned, res.Skipped)
	}
	var out strings.Builder
	FormatResult(&out, res)
	if got := out.String(); !strings.Contains(got, "warning") || !strings.Contains(got, "bad.go: ") || !strings.Contains(got, "expected") {
		t.Errorf("hook output does not warn about bad.go and why:\n%s", got)
	}
}

// TestFormatResultListsSkippedBesideBlockers checks the skipped warning is printed when the gate blocks too.
func TestFormatResultListsSkippedBesideBlockers(t *testing.T) {
	blocker := analyze.Finding{RuleID: "security/hardcoded-secret", Severity: analyze.SeverityCritical, Message: "leak", TargetURI: "file:///a.go"}
	var out strings.Builder
	FormatResult(&out, &Result{
		Stage: StagePreCommit, FilesScanned: 1, TotalFindings: 1, BlockingMin: analyze.SeverityHigh,
		BlockingCount: 1, BlockingDetails: []analyze.Finding{blocker},
		Skipped: []string{"bad.go: parse 3:6: expected 'IDENT'", "worse.go: parse 1:1: expected 'package'"},
	})
	got := out.String()
	for _, want := range []string{"2 path(s)", "bad.go: parse 3:6", "worse.go: parse 1:1", "blocked by findings", "security/hardcoded-secret"} {
		if !strings.Contains(got, want) {
			t.Errorf("output lacks %q:\n%s", want, got)
		}
	}
}

// TestRunHonoursContext checks a done context stops the scan, which is how timeout-seconds takes effect.
func TestRunHonoursContext(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "a.go"), leak)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Run(ctx, RunOptions{Stage: StagePrePush, Config: Default().PrePush, Root: root}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Run with a cancelled context = %v, want context.Canceled", err)
	}
}
