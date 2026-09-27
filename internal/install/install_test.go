package install

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteMCPConfig_CreatesFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".mcp.json")

	created, pre, err := mcpServersTable.write(path, ServerEntry{Command: "chiton", Args: []string{"serve", "--stdio"}})
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	if !created {
		t.Errorf("expected created=true on first write")
	}
	if pre {
		t.Errorf("expected preexisted=false on first write")
	}

	doc := readJSON(t, path)
	servers, _ := doc["mcpServers"].(map[string]any)
	if _, ok := servers["chiton"]; !ok {
		t.Errorf("expected mcpServers.chiton, got %v", doc)
	}
}

func TestWriteMCPConfig_PreservesOtherServers(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".mcp.json")
	existing := `{
  "mcpServers": {
    "filesystem": {
      "command": "fs-mcp",
      "args": ["/tmp"]
    }
  },
  "otherKey": "preserved"
}`
	if err := os.WriteFile(path, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	if _, _, err := mcpServersTable.write(path, ServerEntry{Command: "chiton", Args: []string{"serve"}}); err != nil {
		t.Fatalf("write: %v", err)
	}

	doc := readJSON(t, path)
	servers, _ := doc["mcpServers"].(map[string]any)
	if _, ok := servers["filesystem"]; !ok {
		t.Errorf("filesystem entry was removed; got %v", servers)
	}
	if _, ok := servers["chiton"]; !ok {
		t.Errorf("chiton entry missing; got %v", servers)
	}
	if doc["otherKey"] != "preserved" {
		t.Errorf("otherKey not preserved; got %v", doc["otherKey"])
	}
}

func TestRemoveMCPConfigEntry_RemovesOnlyChiton(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".mcp.json")
	existing := `{"mcpServers":{"chiton":{"command":"x","args":[]},"other":{"command":"y","args":[]}}}`
	if err := os.WriteFile(path, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	changed, err := mcpServersTable.remove(path)
	if err != nil {
		t.Fatal(err)
	}
	if !changed {
		t.Errorf("expected changed=true")
	}
	doc := readJSON(t, path)
	servers, _ := doc["mcpServers"].(map[string]any)
	if _, ok := servers["chiton"]; ok {
		t.Errorf("chiton not removed")
	}
	if _, ok := servers["other"]; !ok {
		t.Errorf("other entry was removed")
	}
}

func TestRemoveMCPConfigEntry_DeletesEmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".mcp.json")
	existing := `{"mcpServers":{"chiton":{"command":"x","args":[]}}}`
	if err := os.WriteFile(path, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := mcpServersTable.remove(path); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("expected file to be removed; got err=%v", err)
	}
}

func TestUpsertRulesBlock_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "CLAUDE.md")

	if _, _, err := UpsertRulesBlock(path, "chiton block v1"); err != nil {
		t.Fatal(err)
	}
	body, _ := os.ReadFile(path)
	if !strings.Contains(string(body), "chiton block v1") {
		t.Errorf("block content missing: %s", body)
	}
	if !strings.Contains(string(body), ChitonBlockStart) {
		t.Errorf("start marker missing")
	}

	// Re-upsert replaces the block in place.
	if _, _, err := UpsertRulesBlock(path, "chiton block v2"); err != nil {
		t.Fatal(err)
	}
	body, _ = os.ReadFile(path)
	if strings.Contains(string(body), "chiton block v1") {
		t.Errorf("old block content not removed: %s", body)
	}
	if !strings.Contains(string(body), "chiton block v2") {
		t.Errorf("new block content missing: %s", body)
	}
	if strings.Count(string(body), ChitonBlockStart) != 1 {
		t.Errorf("expected exactly one start marker, got %d", strings.Count(string(body), ChitonBlockStart))
	}
}

func TestUpsertRulesBlock_PreservesExistingContent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "CLAUDE.md")
	if err := os.WriteFile(path, []byte("# Project rules\n\nFollow PEP 8.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := UpsertRulesBlock(path, "chiton instructions here"); err != nil {
		t.Fatal(err)
	}
	body, _ := os.ReadFile(path)
	if !strings.Contains(string(body), "Follow PEP 8.") {
		t.Errorf("user content lost: %s", body)
	}
	if !strings.Contains(string(body), "chiton instructions here") {
		t.Errorf("chiton block missing: %s", body)
	}
}

func TestRemoveRulesBlock_LeavesUserContent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "CLAUDE.md")
	if err := os.WriteFile(path, []byte("# Project rules\n\nFollow PEP 8.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := UpsertRulesBlock(path, "chiton instructions"); err != nil {
		t.Fatal(err)
	}
	if _, err := RemoveRulesBlock(path); err != nil {
		t.Fatal(err)
	}
	body, _ := os.ReadFile(path)
	if !strings.Contains(string(body), "Follow PEP 8.") {
		t.Errorf("user content removed: %s", body)
	}
	if strings.Contains(string(body), ChitonBlockStart) {
		t.Errorf("block marker still present: %s", body)
	}
}

func TestRemoveRulesBlock_DeletesEmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "CLAUDE.md")
	if _, _, err := UpsertRulesBlock(path, "chiton only"); err != nil {
		t.Fatal(err)
	}
	if _, err := RemoveRulesBlock(path); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("expected file to be removed; got err=%v", err)
	}
}

func TestClaudeCode_InstallUninstallRoundTrip(t *testing.T) {
	dir := t.TempDir()
	opts := Options{Dir: dir, Exe: "/fake/chiton"}

	rep, err := ClaudeCode{}.Install(opts)
	if err != nil {
		t.Fatalf("Install: %v", err)
	}
	if len(rep.Wrote) == 0 {
		t.Errorf("expected files written")
	}

	// Verify each expected artifact exists.
	want := []string{".mcp.json", "CLAUDE.md"}
	for _, skill := range claudeSkills {
		want = append(want, filepath.Join(".claude", "skills", skill.Dir, "SKILL.md"))
	}
	for _, rel := range want {
		if _, err := os.Stat(filepath.Join(dir, rel)); err != nil {
			t.Errorf("expected %s after install: %v", rel, err)
		}
	}

	// Every installed skill must carry YAML frontmatter with a description, or
	// the assistant will not load it as a skill at all.
	for _, skill := range claudeSkills {
		body, err := os.ReadFile(claudeSkillPath(dir, skill.Dir))
		if err != nil {
			t.Fatalf("read %s skill: %v", skill.Dir, err)
		}
		if !strings.HasPrefix(string(body), "---\n") {
			t.Errorf("%s skill does not start with YAML frontmatter", skill.Dir)
		}
		if !strings.Contains(string(body), "\ndescription:") {
			t.Errorf("%s skill frontmatter has no description", skill.Dir)
		}
		// The directory name supplies the skill name in both the standalone and
		// the plugin layout, so an explicit name: would break one of them.
		if strings.Contains(string(body), "\nname:") {
			t.Errorf("%s skill pins an explicit name:, which collides with the plugin namespace", skill.Dir)
		}
	}

	// Re-install should be idempotent (no second chiton block, etc.).
	if _, err := (ClaudeCode{}).Install(opts); err != nil {
		t.Fatalf("second Install: %v", err)
	}
	rules, _ := os.ReadFile(filepath.Join(dir, "CLAUDE.md"))
	if c := strings.Count(string(rules), ChitonBlockStart); c != 1 {
		t.Errorf("expected exactly one chiton block after re-install, got %d", c)
	}

	doctor, err := ClaudeCode{}.Doctor(opts)
	if err != nil {
		t.Fatalf("Doctor: %v", err)
	}
	if !doctor.Registered {
		t.Errorf("expected Registered=true after Install; issues=%v", doctor.Issues)
	}

	if _, err := (ClaudeCode{}).Uninstall(opts); err != nil {
		t.Fatalf("Uninstall: %v", err)
	}
	for _, rel := range []string{".mcp.json", "CLAUDE.md"} {
		if _, err := os.Stat(filepath.Join(dir, rel)); !os.IsNotExist(err) {
			t.Errorf("expected %s removed after uninstall (file is chiton-only); got err=%v", rel, err)
		}
	}
	for _, skill := range claudeSkills {
		if _, err := os.Stat(filepath.Join(dir, ".claude", "skills", skill.Dir)); !os.IsNotExist(err) {
			t.Errorf("expected %s skill dir removed after uninstall; got err=%v", skill.Dir, err)
		}
	}
}

func TestCopilotVSCode_InstallUninstallRoundTrip(t *testing.T) {
	dir := t.TempDir()
	opts := Options{Dir: dir, Exe: "/fake/chiton"}

	if _, err := (CopilotVSCode{}).Install(opts); err != nil {
		t.Fatalf("Install: %v", err)
	}
	for _, rel := range []string{
		filepath.Join(".vscode", "mcp.json"),
		filepath.Join(".github", "copilot-instructions.md"),
	} {
		if _, err := os.Stat(filepath.Join(dir, rel)); err != nil {
			t.Errorf("expected %s after install: %v", rel, err)
		}
	}

	if _, err := (CopilotVSCode{}).Uninstall(opts); err != nil {
		t.Fatalf("Uninstall: %v", err)
	}
}

func TestResolve_AllAndAuto(t *testing.T) {
	dir := t.TempDir()

	got, err := Resolve("all", dir)
	if err != nil {
		t.Fatalf("Resolve(all): %v", err)
	}
	if len(got) < 2 {
		t.Errorf("expected at least 2 installers from 'all', got %d", len(got))
	}

	if _, err := Resolve("auto", dir); err == nil {
		t.Errorf("expected error from 'auto' in empty dir")
	}

	// Drop a CLAUDE.md so Claude Code detects.
	if err := os.WriteFile(filepath.Join(dir, "CLAUDE.md"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err = Resolve("auto", dir)
	if err != nil {
		t.Fatalf("Resolve(auto) after marker: %v", err)
	}
	if len(got) != 1 || got[0].Name() != "claude-code" {
		t.Errorf("expected only claude-code, got %v", names(got))
	}
}

func TestResolve_ExplicitAndCSV(t *testing.T) {
	got, err := Resolve("claude-code,copilot-vscode", t.TempDir())
	if err != nil {
		t.Fatalf("Resolve(csv): %v", err)
	}
	if len(got) != 2 {
		t.Errorf("expected 2, got %d (%v)", len(got), names(got))
	}

	if _, err := Resolve("does-not-exist", t.TempDir()); err == nil {
		t.Errorf("expected error for unknown assistant")
	}
}

func names(is []Installer) []string {
	out := make([]string, len(is))
	for i, x := range is {
		out[i] = x.Name()
	}
	return out
}
