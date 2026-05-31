package install

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	toml "github.com/pelletier/go-toml/v2"
	"gopkg.in/yaml.v3"
)

// setHome redirects user-scope writes (Codex, Antigravity, Copilot CLI) to dir
// for the duration of the test and neutralises COPILOT_HOME.
func setHome(t *testing.T, dir string) {
	t.Helper()
	prev := homeDir
	homeDir = func() (string, error) { return dir, nil }
	t.Cleanup(func() { homeDir = prev })
	t.Setenv("COPILOT_HOME", "")
}

// TestAllInstallers_InstallDoctorUninstall exercises the core contract for
// every registered installer (the two pre-existing ones and the seven added in
// Phase 3.5): Install registers, Doctor confirms, re-Install is idempotent, and
// Uninstall fully deregisters.
func TestAllInstallers_InstallDoctorUninstall(t *testing.T) {
	for _, in := range Available() {
		t.Run(in.Name(), func(t *testing.T) {
			repo := t.TempDir()
			setHome(t, t.TempDir())
			opts := Options{Dir: repo, Exe: "/fake/chiton"}

			if _, err := in.Install(opts); err != nil {
				t.Fatalf("Install: %v", err)
			}
			doc, err := in.Doctor(opts)
			if err != nil {
				t.Fatalf("Doctor: %v", err)
			}
			if !doc.Registered {
				t.Fatalf("expected Registered after Install; issues=%v", doc.Issues)
			}

			// Re-install must be idempotent and stay registered.
			if _, err := in.Install(opts); err != nil {
				t.Fatalf("re-Install: %v", err)
			}
			if doc2, _ := in.Doctor(opts); !doc2.Registered {
				t.Fatalf("expected Registered after re-Install")
			}

			if _, err := in.Uninstall(opts); err != nil {
				t.Fatalf("Uninstall: %v", err)
			}
			doc3, err := in.Doctor(opts)
			if err != nil {
				t.Fatalf("Doctor after Uninstall: %v", err)
			}
			if doc3.Registered {
				t.Fatalf("expected not Registered after Uninstall; ok=%v", doc3.OK)
			}
		})
	}
}

func TestResolve_IncludesPhase35Installers(t *testing.T) {
	got, err := Resolve("all", t.TempDir())
	if err != nil {
		t.Fatalf("Resolve(all): %v", err)
	}
	have := names(got)
	for _, want := range []string{"cursor", "codex", "gemini-cli", "opencode", "aider", "antigravity", "copilot-cli"} {
		if !slices.Contains(have, want) {
			t.Errorf("Resolve(all) missing %q; got %v", want, have)
		}
		if _, err := Resolve(want, t.TempDir()); err != nil {
			t.Errorf("Resolve(%q): %v", want, err)
		}
	}
}

// --- Codex TOML helpers ---

func TestCodexConfig_PreservesAndRoundTrips(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.toml")
	existing := "model = \"o3\"\n\n[mcp_servers.other]\ncommand = \"x\"\nargs = [\"y\"]\n"
	if err := os.WriteFile(path, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	created, pre, err := WriteCodexConfig(path, "/abs/chiton", "./g/graph.badger")
	if err != nil {
		t.Fatal(err)
	}
	if created || pre {
		t.Errorf("created=%v pre=%v; want false,false", created, pre)
	}

	doc := readTOML(t, path)
	if doc["model"] != "o3" {
		t.Errorf("model not preserved: %v", doc["model"])
	}
	servers := doc["mcp_servers"].(map[string]any)
	if _, ok := servers["other"]; !ok {
		t.Errorf("other server dropped: %v", servers)
	}
	chiton := servers["chiton"].(map[string]any)
	if chiton["command"] != "/abs/chiton" {
		t.Errorf("command = %v", chiton["command"])
	}
	args := chiton["args"].([]any)
	if len(args) != 4 || args[0] != "serve" || args[3] != "./g/graph.badger" {
		t.Errorf("args = %v", args)
	}

	// Idempotent: second write reports preexisted.
	if _, pre2, _ := WriteCodexConfig(path, "/abs/chiton", ""); !pre2 {
		t.Errorf("expected preexisted=true on second write")
	}
	if has, _ := CodexConfigHasChiton(path); !has {
		t.Errorf("CodexConfigHasChiton=false after write")
	}

	changed, err := RemoveCodexEntry(path)
	if err != nil || !changed {
		t.Fatalf("RemoveCodexEntry: changed=%v err=%v", changed, err)
	}
	if has, _ := CodexConfigHasChiton(path); has {
		t.Errorf("chiton still present after remove")
	}
	doc = readTOML(t, path)
	if doc["model"] != "o3" {
		t.Errorf("model lost after remove: %v", doc["model"])
	}
	if _, ok := doc["mcp_servers"].(map[string]any)["other"]; !ok {
		t.Errorf("other server lost after remove")
	}
}

func TestCodexConfig_DeletesEmptyFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.toml")
	if _, _, err := WriteCodexConfig(path, "/abs/chiton", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := RemoveCodexEntry(path); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("expected chiton-only config.toml removed; err=%v", err)
	}
}

// --- OpenCode helpers ---

func TestOpenCodeConfig_PreservesAndRoundTrips(t *testing.T) {
	path := filepath.Join(t.TempDir(), "opencode.json")
	existing := `{"model":"anthropic/claude","mcp":{"other":{"type":"local","command":["a"],"enabled":true}}}`
	if err := os.WriteFile(path, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	if _, _, err := WriteOpenCodeConfig(path, "/abs/chiton", "./g"); err != nil {
		t.Fatal(err)
	}
	doc := readJSON(t, path)
	if doc["model"] != "anthropic/claude" {
		t.Errorf("model not preserved: %v", doc["model"])
	}
	if doc["$schema"] != openCodeSchema {
		t.Errorf("$schema = %v", doc["$schema"])
	}
	mcp := doc["mcp"].(map[string]any)
	if _, ok := mcp["other"]; !ok {
		t.Errorf("other server dropped")
	}
	chiton := mcp["chiton"].(map[string]any)
	if chiton["type"] != "local" || chiton["enabled"] != true {
		t.Errorf("type/enabled wrong: %v", chiton)
	}
	cmd := chiton["command"].([]any)
	want := []any{"/abs/chiton", "serve", "--stdio", "--graph", "./g"}
	if len(cmd) != len(want) {
		t.Fatalf("command = %v, want %v", cmd, want)
	}
	for i := range want {
		if cmd[i] != want[i] {
			t.Errorf("command[%d] = %v, want %v", i, cmd[i], want[i])
		}
	}

	if has, _ := OpenCodeConfigHasChiton(path); !has {
		t.Errorf("OpenCodeConfigHasChiton=false after write")
	}
	if _, err := RemoveOpenCodeEntry(path); err != nil {
		t.Fatal(err)
	}
	if has, _ := OpenCodeConfigHasChiton(path); has {
		t.Errorf("chiton still present after remove")
	}
	doc = readJSON(t, path)
	if doc["model"] != "anthropic/claude" {
		t.Errorf("model lost after remove")
	}
}

func TestOpenCodeConfig_DeletesSchemaOnlyFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "opencode.json")
	if _, _, err := WriteOpenCodeConfig(path, "/abs/chiton", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := RemoveOpenCodeEntry(path); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("expected chiton-only opencode.json removed; err=%v", err)
	}
}

// --- Aider .aider.conf.yml read-list helpers ---

func TestAiderRead_ScalarAndListMerge(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".aider.conf.yml")
	// `read` as a scalar string plus an unrelated setting.
	if err := os.WriteFile(path, []byte("auto-commits: false\nread: EXISTING.md\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	if _, pre, err := AddAiderRead(path, "CONVENTIONS.md"); err != nil || pre {
		t.Fatalf("AddAiderRead: pre=%v err=%v", pre, err)
	}
	doc := readYAML(t, path)
	if doc["auto-commits"] != false {
		t.Errorf("auto-commits not preserved: %v", doc["auto-commits"])
	}
	reads := normalizeStringList(doc["read"])
	if !slices.Contains(reads, "EXISTING.md") || !slices.Contains(reads, "CONVENTIONS.md") {
		t.Errorf("read list = %v, want both EXISTING.md and CONVENTIONS.md", reads)
	}

	// Idempotent.
	if _, pre, _ := AddAiderRead(path, "CONVENTIONS.md"); !pre {
		t.Errorf("expected preexisted=true on second AddAiderRead")
	}
	if reads := normalizeStringList(readYAML(t, path)["read"]); len(reads) != 2 {
		t.Errorf("expected 2 read entries after idempotent add, got %v", reads)
	}

	if has, _ := AiderConfHasRead(path, "CONVENTIONS.md"); !has {
		t.Errorf("AiderConfHasRead=false")
	}
	changed, err := RemoveAiderRead(path, "CONVENTIONS.md")
	if err != nil || !changed {
		t.Fatalf("RemoveAiderRead: changed=%v err=%v", changed, err)
	}
	reads = normalizeStringList(readYAML(t, path)["read"])
	if slices.Contains(reads, "CONVENTIONS.md") || !slices.Contains(reads, "EXISTING.md") {
		t.Errorf("after remove read = %v, want only EXISTING.md", reads)
	}
}

func TestAider_UserScopeUnsupported(t *testing.T) {
	if _, err := (Aider{}).Install(Options{Dir: t.TempDir(), UserScope: true}); err != ErrUserScopeUnsupported {
		t.Errorf("expected ErrUserScopeUnsupported, got %v", err)
	}
}

// TestSharedAGENTSFile verifies that two installers sharing AGENTS.md keep a
// single chiton block and that uninstalling one leaves the file usable.
func TestSharedAGENTSFile(t *testing.T) {
	repo := t.TempDir()
	setHome(t, t.TempDir())
	opts := Options{Dir: repo, Exe: "/fake/chiton"}

	if _, err := (OpenCode{}).Install(opts); err != nil {
		t.Fatal(err)
	}
	if _, err := (Codex{}).Install(opts); err != nil {
		t.Fatal(err)
	}
	agents := filepath.Join(repo, "AGENTS.md")
	body, _ := os.ReadFile(agents)
	if c := countOccurrences(string(body), ChitonBlockStart); c != 1 {
		t.Fatalf("expected exactly one chiton block in shared AGENTS.md, got %d", c)
	}
}

// TestSharedAGENTS_UninstallReferenceCounted verifies that uninstalling one
// AGENTS.md-using assistant keeps the shared chiton block for the others, and
// only the last uninstall removes it.
func TestSharedAGENTS_UninstallReferenceCounted(t *testing.T) {
	repo := t.TempDir()
	setHome(t, t.TempDir())
	opts := Options{Dir: repo, Exe: "/fake/chiton"}
	agents := filepath.Join(repo, "AGENTS.md")

	if _, err := (OpenCode{}).Install(opts); err != nil {
		t.Fatal(err)
	}
	if _, err := (Codex{}).Install(opts); err != nil {
		t.Fatal(err)
	}

	// Uninstalling Codex must not strip the block OpenCode still relies on.
	if _, err := (Codex{}).Uninstall(opts); err != nil {
		t.Fatal(err)
	}
	if has, _ := RulesFileHasChiton(agents); !has {
		t.Fatal("shared AGENTS.md block removed while OpenCode still installed")
	}
	if doc, _ := (OpenCode{}).Doctor(opts); !doc.Registered {
		t.Fatalf("OpenCode lost registration after Codex uninstall: %v", doc.Issues)
	}

	// Uninstalling the last AGENTS.md user removes the block and the file.
	if _, err := (OpenCode{}).Uninstall(opts); err != nil {
		t.Fatal(err)
	}
	if has, _ := RulesFileHasChiton(agents); has {
		t.Fatal("block should be gone after the last AGENTS.md user uninstalled")
	}
	if _, err := os.Stat(agents); !os.IsNotExist(err) {
		t.Errorf("expected chiton-only AGENTS.md removed; err=%v", err)
	}
}

func countOccurrences(s, sub string) int {
	n := 0
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			n++
		}
	}
	return n
}

// --- helpers ---

func readTOML(t *testing.T, path string) map[string]any {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	doc := map[string]any{}
	if err := toml.Unmarshal(b, &doc); err != nil {
		t.Fatalf("unmarshal toml %s: %v", path, err)
	}
	return doc
}

func readJSON(t *testing.T, path string) map[string]any {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("unmarshal json %s: %v", path, err)
	}
	return doc
}

func readYAML(t *testing.T, path string) map[string]any {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	doc := map[string]any{}
	if err := yaml.Unmarshal(b, &doc); err != nil {
		t.Fatalf("unmarshal yaml %s: %v", path, err)
	}
	return doc
}
