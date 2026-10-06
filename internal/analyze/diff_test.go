package analyze

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func TestDiff(t *testing.T) {
	base := []FindingView{
		{ID: "a", RuleID: "r1", Severity: "high", Message: "m1"},
		{ID: "b", RuleID: "r2", Severity: "low", Message: "m2"},
	}
	cur := []FindingView{
		{ID: "b", RuleID: "r2", Severity: "low", Message: "m2-changed"}, // changed message
		{ID: "c", RuleID: "r3", Severity: "medium", Message: "m3"},      // new
	}
	d := Diff(base, cur)
	if len(d.New) != 1 || d.New[0].ID != "c" {
		t.Fatalf("new: %+v", d.New)
	}
	if len(d.Resolved) != 1 || d.Resolved[0].ID != "a" {
		t.Fatalf("resolved: %+v", d.Resolved)
	}
	if len(d.Changed) != 1 || d.Changed[0].ID != "b" {
		t.Fatalf("changed: %+v", d.Changed)
	}
	if d.Changed[0].Before.Message != "m2" || d.Changed[0].After.Message != "m2-changed" {
		t.Fatalf("changed before/after: %+v", d.Changed[0])
	}
	if d.Empty() {
		t.Fatal("diff should not be empty")
	}
}

func TestDiffEmptyWhenIdentical(t *testing.T) {
	fs := []FindingView{{ID: "a", RuleID: "r1", Severity: "high", Message: "m"}}
	if !Diff(fs, fs).Empty() {
		t.Fatal("identical sets should diff empty")
	}
	// Against an empty baseline, every current finding is new.
	d := Diff(nil, fs)
	if len(d.New) != 1 || d.Empty() {
		t.Fatalf("expected 1 new vs empty baseline: %+v", d)
	}
}

func TestBaselineRoundTrip(t *testing.T) {
	dir := BaselineDir(t.TempDir())
	findings := []FindingView{{ID: "a", RuleID: "r1", Severity: "high", Message: "m"}}
	now := time.Now()

	path, err := SaveBaseline(dir, "abc123def456", findings, now)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(path) != "abc123def456.json" {
		t.Fatalf("unexpected baseline filename: %s", path)
	}

	got, err := LoadBaseline(dir, "abc123def456")
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.Commit != "abc123def456" || len(got.Findings) != 1 {
		t.Fatalf("loaded baseline: %+v", got)
	}

	latest, err := LatestBaseline(dir)
	if err != nil || latest == nil || latest.Commit != "abc123def456" {
		t.Fatalf("latest: %+v (err=%v)", latest, err)
	}

	// "latest" alias resolves to LatestBaseline.
	byAlias, err := LoadBaseline(dir, "latest")
	if err != nil || byAlias == nil || byAlias.Commit != "abc123def456" {
		t.Fatalf("latest alias: %+v (err=%v)", byAlias, err)
	}

	// Missing baseline → (nil, nil).
	miss, err := LoadBaseline(dir, "does-not-exist")
	if err != nil {
		t.Fatal(err)
	}
	if miss != nil {
		t.Fatal("expected nil for missing baseline")
	}
}

func TestBaselineEmptyDir(t *testing.T) {
	// LatestBaseline on a non-existent dir returns (nil, nil), not an error.
	b, err := LatestBaseline(filepath.Join(t.TempDir(), "nope"))
	if err != nil {
		t.Fatal(err)
	}
	if b != nil {
		t.Fatal("expected nil baseline for empty dir")
	}
}

// diff_findings resolves MCP-supplied refs through GitCommit, and an unguarded `git rev-parse --show-toplevel`
// prints the repo path, which the tool then echoes back.
func TestGitCommitRefusesOptionLikeRefs(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	for _, args := range [][]string{{"init", "-q"}, {"commit", "--allow-empty", "-q", "-m", "c"}} {
		cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	for _, ref := range []string{"--show-toplevel", "--git-dir", "--output=x"} {
		if got := GitCommit(root, ref); got != "" {
			t.Errorf("GitCommit(%q) = %q, want \"\"", ref, got)
		}
	}
	if sha := GitCommit(root, "HEAD"); len(sha) != 40 {
		t.Errorf("GitCommit(HEAD) = %q, want a 40-hex SHA", sha)
	}
}
