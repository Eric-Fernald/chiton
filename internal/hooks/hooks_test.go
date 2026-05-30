package hooks

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoad_DefaultsWhenMissing(t *testing.T) {
	cfg, err := Load(filepath.Join(t.TempDir(), "nonexistent.yaml"))
	if err != nil {
		t.Fatalf("Load missing: %v", err)
	}
	def := Default()
	if cfg.PreCommit.TimeoutSeconds != def.PreCommit.TimeoutSeconds {
		t.Errorf("expected default timeout, got %d", cfg.PreCommit.TimeoutSeconds)
	}
	if len(cfg.PreCommit.BlockOn) != len(def.PreCommit.BlockOn) {
		t.Errorf("expected default BlockOn, got %v", cfg.PreCommit.BlockOn)
	}
}

func TestLoad_PartialOverlay(t *testing.T) {
	path := filepath.Join(t.TempDir(), "hooks.yaml")
	if err := os.WriteFile(path, []byte("pre-commit:\n  block-on: [critical]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.PreCommit.BlockOn) != 1 || cfg.PreCommit.BlockOn[0] != "critical" {
		t.Errorf("override not applied: %v", cfg.PreCommit.BlockOn)
	}
	if cfg.PreCommit.TimeoutSeconds == 0 {
		t.Errorf("expected default timeout to fill in")
	}
}

func TestMinBlockingSeverity(t *testing.T) {
	cases := []struct {
		name string
		in   []string
		want string
	}{
		{"critical-only", []string{"critical"}, "critical"},
		{"critical-and-high", []string{"critical", "high"}, "high"},
		{"includes-low", []string{"low", "critical"}, "low"},
		{"empty", nil, ""},
		{"all-junk", []string{"banana"}, ""},
	}
	for _, tc := range cases {
		got := StageConfig{BlockOn: tc.in}.MinBlockingSeverity()
		if string(got) != tc.want {
			t.Errorf("%s: want %q got %q", tc.name, tc.want, got)
		}
	}
}

func TestInstall_WritesScriptsAndIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	rep, err := Install(dir, "/fake/chiton", Stages())
	if err != nil {
		t.Fatalf("Install: %v", err)
	}
	if len(rep.Wrote) != 2 {
		t.Errorf("expected 2 hooks written, got %d", len(rep.Wrote))
	}

	pre := filepath.Join(dir, "pre-commit")
	body, err := os.ReadFile(pre)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), HookMarker) {
		t.Errorf("hook missing chiton marker: %s", body)
	}
	if !strings.Contains(string(body), "hooks run --stage pre-commit") {
		t.Errorf("hook missing chiton invocation: %s", body)
	}

	// Re-run install — should refresh without complaints.
	if _, err := Install(dir, "/fake/chiton", Stages()); err != nil {
		t.Fatalf("second Install: %v", err)
	}
}

func TestInstall_SyncStages(t *testing.T) {
	dir := t.TempDir()
	if _, err := Install(dir, "/fake/chiton", SyncStages()); err != nil {
		t.Fatalf("Install sync stages: %v", err)
	}

	pc, err := os.ReadFile(filepath.Join(dir, "post-commit"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(pc), "sync git --stage post-commit") {
		t.Errorf("post-commit should call sync git: %s", pc)
	}
	if strings.Contains(string(pc), "hooks run") {
		t.Errorf("post-commit must not be a gating hook: %s", pc)
	}

	co, err := os.ReadFile(filepath.Join(dir, "post-checkout"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(co), "sync git --stage post-checkout") {
		t.Errorf("post-checkout should call sync git: %s", co)
	}
	if !strings.Contains(string(co), `"$@"`) {
		t.Errorf("post-checkout should forward git's positional args: %s", co)
	}
}

func TestSyncStageHelpers(t *testing.T) {
	if !IsSyncStage(StagePostCommit) || !IsSyncStage(StagePostCheckout) {
		t.Error("post-commit/post-checkout should be sync stages")
	}
	if IsSyncStage(StagePreCommit) || IsSyncStage(StagePrePush) {
		t.Error("gating stages should not be sync stages")
	}
	if len(AllStages()) != 4 {
		t.Errorf("AllStages should have 4 entries, got %d", len(AllStages()))
	}
}

func TestInstall_LeavesForeignHooksAlone(t *testing.T) {
	dir := t.TempDir()
	pre := filepath.Join(dir, "pre-commit")
	if err := os.WriteFile(pre, []byte("#!/bin/sh\necho user-script\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	rep, err := Install(dir, "/fake/chiton", []Stage{StagePreCommit})
	if err != nil {
		t.Fatalf("Install: %v", err)
	}
	if len(rep.Wrote) != 0 {
		t.Errorf("expected no writes when foreign hook present, got %v", rep.Wrote)
	}
	if len(rep.Notes) == 0 {
		t.Errorf("expected note about foreign hook")
	}
	body, _ := os.ReadFile(pre)
	if !strings.Contains(string(body), "user-script") {
		t.Errorf("foreign hook was clobbered: %s", body)
	}
}

func TestUninstall_RemovesOnlyChitonOwned(t *testing.T) {
	dir := t.TempDir()
	if _, err := Install(dir, "/fake/chiton", Stages()); err != nil {
		t.Fatal(err)
	}
	// Drop a foreign pre-push to make sure we don't remove it.
	prePush := filepath.Join(dir, "pre-push")
	if err := os.WriteFile(prePush, []byte("#!/bin/sh\necho mine\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := Uninstall(dir, Stages()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "pre-commit")); !os.IsNotExist(err) {
		t.Errorf("expected pre-commit removed, got err=%v", err)
	}
	if _, err := os.Stat(prePush); err != nil {
		t.Errorf("foreign pre-push was removed: %v", err)
	}
}

func TestStatus(t *testing.T) {
	dir := t.TempDir()
	if _, err := Install(dir, "/fake/chiton", []Stage{StagePreCommit}); err != nil {
		t.Fatal(err)
	}
	entries, err := Status(dir, Stages())
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if e.Stage == StagePreCommit && (!e.Exists || !e.ChitonOwned) {
			t.Errorf("pre-commit should be installed and chiton-owned: %+v", e)
		}
		if e.Stage == StagePrePush && e.Exists {
			t.Errorf("pre-push should not exist: %+v", e)
		}
	}
}
