package chiton

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// These tests guard the Claude Code plugin layout, which otherwise only an actual Claude Code install
// would exercise.

var kebab = regexp.MustCompile(`^[a-z0-9]+(-[a-z0-9]+)*$`)

func readJSON(t *testing.T, path string, into any) {
	t.Helper()
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if len(body) >= 3 && body[0] == 0xEF && body[1] == 0xBB && body[2] == 0xBF {
		t.Fatalf("%s starts with a UTF-8 BOM; Claude Code cannot parse it", path)
	}
	if err := json.Unmarshal(body, into); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}

func TestPluginManifest(t *testing.T) {
	var m struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		Version     string `json:"version"`
		Author      struct {
			Name string `json:"name"`
		} `json:"author"`
	}
	readJSON(t, filepath.Join(".claude-plugin", "plugin.json"), &m)

	if !kebab.MatchString(m.Name) {
		t.Errorf("plugin name %q is not kebab-case; the claude.ai marketplace sync rejects other forms", m.Name)
	}
	if m.Description == "" {
		t.Error("plugin.json has no description; it is what users see when browsing the marketplace")
	}
	if m.Version == "" {
		t.Error("plugin.json has no version; users only receive updates when it is bumped")
	}
	if m.Author.Name == "" {
		t.Error("plugin.json has no author.name; `claude plugin validate --strict` warns on this")
	}
}

func TestMarketplaceManifest(t *testing.T) {
	var m struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		Owner       struct {
			Name string `json:"name"`
		} `json:"owner"`
		Plugins []struct {
			Name    string `json:"name"`
			Source  string `json:"source"`
			Version string `json:"version"`
		} `json:"plugins"`
	}
	readJSON(t, filepath.Join(".claude-plugin", "marketplace.json"), &m)

	if !kebab.MatchString(m.Name) {
		t.Errorf("marketplace name %q is not kebab-case", m.Name)
	}
	if m.Description == "" {
		t.Error("marketplace.json has no description; --strict turns that warning into a failure")
	}
	if m.Owner.Name == "" {
		t.Error("marketplace.json has no owner.name, which is required")
	}
	if len(m.Plugins) != 1 {
		t.Fatalf("expected exactly one plugin entry, got %d", len(m.Plugins))
	}

	entry := m.Plugins[0]
	// "./" and not "." — a bare "." only validates on Claude Code 2.1.221+.
	if entry.Source != "./" {
		t.Errorf("plugin entry source = %q, want \"./\" (the repo root is the plugin)", entry.Source)
	}
	// plugin.json wins at install time; declaring a version here too only
	// creates a mismatch warning that --strict fails on.
	if entry.Version != "" {
		t.Errorf("plugin entry declares version %q; leave it to plugin.json", entry.Version)
	}
}

func TestPluginMCPConfig(t *testing.T) {
	var m struct {
		MCPServers map[string]struct {
			Command string            `json:"command"`
			Args    []string          `json:"args"`
			Env     map[string]string `json:"env"`
		} `json:"mcpServers"`
	}
	readJSON(t, ".mcp.json", &m)

	srv, ok := m.MCPServers["chiton"]
	if !ok {
		t.Fatal(".mcp.json has no \"chiton\" server entry")
	}
	// Launching by bare name is deliberate: an absolute or extensionless path
	// is not portable across Windows and POSIX, so the binary must be on PATH.
	if srv.Command != "chiton" {
		t.Errorf("mcp command = %q, want \"chiton\" (resolved from PATH)", srv.Command)
	}
	if strings.Join(srv.Args, " ") != "serve --stdio" {
		t.Errorf("mcp args = %v, want [serve --stdio]", srv.Args)
	}
	// Without this the server resolves the graph against its own working
	// directory, which is not reliably the workspace root.
	if got := srv.Env["CHITON_PROJECT_DIR"]; got != "${CLAUDE_PROJECT_DIR}" {
		t.Errorf("env CHITON_PROJECT_DIR = %q, want ${CLAUDE_PROJECT_DIR}", got)
	}
	// A stdio server that could write would make add_url reachable from the
	// assistant without the user asking. Keep the default read-only.
	for _, a := range srv.Args {
		if a == "--writable" {
			t.Error("the plugin must not start a writable server by default")
		}
	}
}

func TestSkillsAreLoadable(t *testing.T) {
	skills := map[string]string{
		"graph":          GraphSkillMarkdown,
		"subagent-tasks": SubagentTasksSkillMarkdown,
		"setup":          SetupSkillMarkdown,
	}

	for dir, body := range skills {
		if !strings.HasPrefix(body, "---\n") {
			t.Errorf("skills/%s/SKILL.md does not open with YAML frontmatter", dir)
			continue
		}
		end := strings.Index(body[4:], "\n---\n")
		if end < 0 {
			t.Errorf("skills/%s/SKILL.md frontmatter is not terminated", dir)
			continue
		}
		front := body[4 : 4+end]

		if !strings.Contains(front, "description:") {
			t.Errorf("skills/%s/SKILL.md has no description; Claude cannot tell when to use it", dir)
		}
		// The directory name supplies the command in both the plugin and `chiton install` layouts.
		// An explicit `name:` would desynchronise the two.
		if strings.Contains(front, "name:") {
			t.Errorf("skills/%s/SKILL.md pins an explicit name:, which collides with the plugin namespace", dir)
		}
		if len(strings.TrimSpace(body[4+end+5:])) == 0 {
			t.Errorf("skills/%s/SKILL.md has frontmatter but no body", dir)
		}
	}

	// Every directory under skills/ must be embedded, or the plugin would ship
	// a skill the binary does not know about.
	entries, err := os.ReadDir("skills")
	if err != nil {
		t.Fatalf("read skills/: %v", err)
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if _, ok := skills[e.Name()]; !ok {
			t.Errorf("skills/%s is not embedded in embed.go", e.Name())
		}
	}
}

func TestPluginComponentsAtRoot(t *testing.T) {
	// Claude Code only reads these at the plugin root. The single most common
	// packaging mistake is nesting them inside .claude-plugin/.
	for _, p := range []string{
		".mcp.json",
		filepath.Join("skills", "graph", "SKILL.md"),
		filepath.Join("bin", "chiton"),
		filepath.Join("bin", "chiton.cmd"),
		filepath.Join("scripts", "install-chiton.sh"),
		filepath.Join("scripts", "install-chiton.ps1"),
	} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("expected %s at the plugin root: %v", p, err)
		}
	}

	// .claude-plugin/ carries manifests and nothing else.
	entries, err := os.ReadDir(".claude-plugin")
	if err != nil {
		t.Fatalf("read .claude-plugin/: %v", err)
	}
	for _, e := range entries {
		switch e.Name() {
		case "plugin.json", "marketplace.json":
		default:
			t.Errorf(".claude-plugin/%s does not belong there; components live at the plugin root", e.Name())
		}
	}
}

func TestPOSIXShimsUseLFAndAShebang(t *testing.T) {
	// Committed from Windows, these are the files most likely to arrive with
	// CRLF, which makes the shebang line unusable on Linux and macOS.
	for _, p := range []string{filepath.Join("bin", "chiton"), filepath.Join("scripts", "install-chiton.sh")} {
		body, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("read %s: %v", p, err)
		}
		if !strings.HasPrefix(string(body), "#!/bin/sh") {
			t.Errorf("%s does not start with a /bin/sh shebang", p)
		}
		if strings.Contains(string(body), "\r\n") {
			t.Errorf("%s contains CRLF line endings; it will not execute on POSIX hosts", p)
		}
	}
}
