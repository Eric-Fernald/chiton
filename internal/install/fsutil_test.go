package install

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestConfigWritesKeepExistingMode checks install and uninstall keep a config's permissions, since
// ~/.codex/config.toml and friends often hold other MCP servers' API keys.
func TestConfigWritesKeepExistingMode(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix permission bits are not meaningful on Windows")
	}
	dir := t.TempDir()
	cfg := filepath.Join(dir, "mcp.json")
	if err := os.WriteFile(cfg, []byte(`{"mcpServers":{"other":{"command":"x","env":{"API_KEY":"k"}}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(cfg, 0o640); err != nil { // explicit chmod, so the umask can't mask the check
		t.Fatal(err)
	}
	if _, _, err := mcpServersTable.write(cfg, ChitonServerEntry("/abs/chiton", "")); err != nil {
		t.Fatal(err)
	}
	if got := permOf(t, cfg); got != 0o640 {
		t.Errorf("after install mode = %v, want 0640 kept", got)
	}
	if _, err := mcpServersTable.remove(cfg); err != nil {
		t.Fatal(err)
	}
	if got := permOf(t, cfg); got != 0o640 {
		t.Errorf("after uninstall mode = %v, want 0640 kept", got)
	}

	rules := filepath.Join(dir, "AGENTS.md")
	if err := os.WriteFile(rules, []byte("# mine\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(rules, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := UpsertRulesBlock(rules, "hello"); err != nil {
		t.Fatal(err)
	}
	if got := permOf(t, rules); got != 0o600 {
		t.Errorf("rules file mode = %v, want 0600 kept", got)
	}
}

// A new MCP config starts owner-only.
func TestNewMCPConfigIsOwnerOnly(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix permission bits are not meaningful on Windows")
	}
	cfg := filepath.Join(t.TempDir(), ".codex", "config.toml")
	if _, _, err := codexTable.write(cfg, chitonCodexEntry("/abs/chiton", "")); err != nil {
		t.Fatal(err)
	}
	if got := permOf(t, cfg); got&0o077 != 0 {
		t.Errorf("new config mode = %v, want no group/other bits", got)
	}
}

// TestConfigWritesFollowSymlinks checks a symlinked config (a dotfiles manager's layout) is updated in place:
// the link survives and the file it names gets the change, on install and on an emptying uninstall.
func TestConfigWritesFollowSymlinks(t *testing.T) {
	dir := t.TempDir()
	dotfiles := filepath.Join(dir, "dotfiles")
	if err := os.MkdirAll(dotfiles, 0o755); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(dotfiles, "config.toml")
	if err := os.WriteFile(target, []byte("model = \"o3\"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "config.toml")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("cannot create symlinks here: %v", err)
	}
	if _, _, err := codexTable.write(link, chitonCodexEntry("/abs/chiton", "")); err != nil {
		t.Fatal(err)
	}
	assertSymlink(t, link)
	if has, _ := codexTable.has(target); !has {
		t.Fatal("chiton entry not written to the symlink target")
	}
	if readTOML(t, target)["model"] != "o3" {
		t.Error("the target's other settings were lost")
	}

	// A chiton-only file behind a link: uninstall must empty the target, not delete the link and strand it.
	only := filepath.Join(dotfiles, "mcp.json")
	if err := os.WriteFile(only, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	onlyLink := filepath.Join(dir, "mcp.json")
	if err := os.Symlink(only, onlyLink); err != nil {
		t.Fatal(err)
	}
	if _, _, err := mcpServersTable.write(onlyLink, ChitonServerEntry("/abs/chiton", "")); err != nil {
		t.Fatal(err)
	}
	if _, err := mcpServersTable.remove(onlyLink); err != nil {
		t.Fatal(err)
	}
	assertSymlink(t, onlyLink)
	if has, _ := mcpServersTable.has(only); has {
		t.Error("uninstall left the chiton entry in the symlink target")
	}

	// CLAUDE.md -> AGENTS.md is a common layout; the rules block must land in AGENTS.md.
	agents := filepath.Join(dir, "AGENTS.md")
	if err := os.WriteFile(agents, []byte("# shared\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	claude := filepath.Join(dir, "CLAUDE.md")
	if err := os.Symlink(agents, claude); err != nil {
		t.Fatal(err)
	}
	if _, _, err := UpsertRulesBlock(claude, "hello"); err != nil {
		t.Fatal(err)
	}
	assertSymlink(t, claude)
	if has, _ := RulesFileHasChiton(agents); !has {
		t.Error("rules block not written through the CLAUDE.md symlink")
	}
}

func TestConfigWriteRejectsDanglingSymlink(t *testing.T) {
	dir := t.TempDir()
	link := filepath.Join(dir, "mcp.json")
	if err := os.Symlink(filepath.Join(dir, "missing.json"), link); err != nil {
		t.Skipf("cannot create symlinks here: %v", err)
	}
	_, _, err := mcpServersTable.write(link, ChitonServerEntry("/abs/chiton", ""))
	if err == nil || !strings.Contains(err.Error(), "symlink") {
		t.Fatalf("write through a dangling symlink: err = %v, want a symlink error", err)
	}
	assertSymlink(t, link)
}

func permOf(t *testing.T, path string) os.FileMode {
	t.Helper()
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return fi.Mode().Perm()
}

func assertSymlink(t *testing.T, path string) {
	t.Helper()
	fi, err := os.Lstat(path)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("%s was replaced by a regular file", path)
	}
}
