package install

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/install/manifests"
)

var updateGolden = flag.Bool("update", false, "rewrite testdata/installers.golden")

const (
	goldenJSONServers = "{\n  // user servers\n  \"mcpServers\": {\"other\": {\"command\": \"x\", \"args\": []}},\n  \"extra\": true,\n}\n"
	goldenOpenCode    = `{"model":"m","mcp":{"other":{"type":"local","command":["a"],"enabled":true}}}` + "\n"
	goldenRules       = "# User rules\n\nKeep this.\n"
	goldenBadRules    = "<!-- chiton:end -->\nuser text\n<!-- chiton:start -->\nstale\n"
)

// goldenSeeded is user content every installer must merge around; keys are relative to the test root.
var goldenSeeded = map[string]string{
	"repo/.mcp.json":                       goldenJSONServers,
	"repo/.vscode/mcp.json":                goldenJSONServers,
	"repo/.cursor/mcp.json":                goldenJSONServers,
	"repo/.gemini/settings.json":           goldenJSONServers,
	"repo/opencode.json":                   goldenOpenCode,
	"repo/CLAUDE.md":                       goldenRules,
	"repo/AGENTS.md":                       goldenRules,
	"repo/GEMINI.md":                       goldenRules,
	"repo/CONVENTIONS.md":                  goldenRules,
	"repo/.github/copilot-instructions.md": goldenRules,
	"repo/.aider.conf.yml":                 "auto-commits: false\nread: EXISTING.md\n",
	"home/.codex/config.toml":              "model = \"o3\"\n\n[mcp_servers.other]\ncommand = \"x\"\nargs = [\"y\"]\n",
	"home/.gemini/config/mcp_config.json":  goldenJSONServers,
	"home/.copilot/mcp-config.json":        goldenJSONServers,
	"home/.cursor/mcp.json":                goldenJSONServers,
	"home/.gemini/settings.json":           goldenJSONServers,
	"home/.config/opencode/opencode.json":  goldenOpenCode,
}

// goldenMalformed puts unparseable content (and out-of-order block markers) at every path installers touch.
var goldenMalformed = map[string]string{
	"repo/.mcp.json":                       "{ not json",
	"repo/.vscode/mcp.json":                "{ not json",
	"repo/.cursor/mcp.json":                `{"mcpServers": "oops"}`,
	"repo/.gemini/settings.json":           "{ not json",
	"repo/opencode.json":                   `{"mcp": []}`,
	"repo/CLAUDE.md":                       goldenBadRules,
	"repo/AGENTS.md":                       goldenBadRules,
	"repo/GEMINI.md":                       goldenBadRules,
	"repo/CONVENTIONS.md":                  goldenBadRules,
	"repo/.github/copilot-instructions.md": goldenBadRules,
	"repo/.aider.conf.yml":                 "read: [unclosed\n",
	"home/.codex/config.toml":              "model = \n",
	"home/.gemini/config/mcp_config.json":  "{ not json",
	"home/.copilot/mcp-config.json":        `{"mcpServers": 3}`,
}

var goldenScenarios = []struct {
	name      string
	seed      map[string]string
	graph     string
	userScope bool
	steps     []string
}{
	{"fresh", nil, "", false, []string{"install", "install", "doctor", "uninstall", "doctor", "uninstall"}},
	{"seeded", goldenSeeded, "./g/graph.badger", false, []string{"install", "doctor", "uninstall", "doctor"}},
	{"user-scope", goldenSeeded, "", true, []string{"install", "doctor", "uninstall"}},
	{"malformed", goldenMalformed, "", false, []string{"doctor", "install", "uninstall"}},
}

// TestInstallersGolden pins every installer's files, reports, and errors byte-for-byte.
// Regenerate with: go test ./internal/install -run TestInstallersGolden -update
func TestInstallersGolden(t *testing.T) {
	var sb strings.Builder
	for _, in := range Available() {
		for _, sc := range goldenScenarios {
			fmt.Fprintf(&sb, "# %s / %s\n", in.Name(), sc.name)
			g := newGoldenRun(t, &sb, sc.seed)
			opts := Options{Dir: g.repo, Exe: "/fake/chiton", GraphPath: sc.graph, UserScope: sc.userScope}
			for _, step := range sc.steps {
				g.run(in, step, opts)
			}
		}
	}

	sb.WriteString("# shared AGENTS.md\n")
	g := newGoldenRun(t, &sb, nil)
	opts := Options{Dir: g.repo, Exe: "/fake/chiton"}
	for _, name := range []string{"opencode", "codex", "copilot-cli", "antigravity"} {
		g.run(Lookup(name), "install", opts)
	}
	for _, name := range []string{"codex", "opencode", "antigravity", "copilot-cli"} {
		g.run(Lookup(name), "uninstall", opts)
	}

	compareGolden(t, filepath.Join("testdata", "installers.golden"), sb.String())
}

type goldenRun struct {
	t                *testing.T
	sb               *strings.Builder
	root, repo, home string
	prev             map[string]string
}

func newGoldenRun(t *testing.T, sb *strings.Builder, seed map[string]string) *goldenRun {
	t.Helper()
	root := t.TempDir()
	g := &goldenRun{t: t, sb: sb, root: root, repo: filepath.Join(root, "repo"), home: filepath.Join(root, "home")}
	for _, d := range []string{g.repo, g.home} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for rel, body := range seed {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	setHome(t, g.home)
	g.prev = g.files()
	return g
}

func (g *goldenRun) run(in Installer, step string, opts Options) {
	var (
		rep any
		err error
	)
	switch step {
	case "install":
		var r Report
		r, err = in.Install(opts)
		r.Wrote, r.Skipped, r.Notes = g.normAll(r.Wrote), g.normAll(r.Skipped), g.normAll(r.Notes)
		rep = r
	case "uninstall":
		var r Report
		r, err = in.Uninstall(opts)
		r.Wrote, r.Skipped, r.Notes = g.normAll(r.Wrote), g.normAll(r.Skipped), g.normAll(r.Notes)
		rep = r
	case "doctor":
		var r DoctorReport
		r, err = in.Doctor(opts)
		r.OK, r.Issues = g.normAll(r.OK), g.normAll(r.Issues)
		rep = r
	}
	fmt.Fprintf(g.sb, "## %s %s\n", step, in.Name())
	if err != nil {
		fmt.Fprintf(g.sb, "err: %s\n", g.norm(err.Error()))
	}
	b, _ := json.MarshalIndent(rep, "", "  ")
	g.sb.Write(b)
	g.sb.WriteString("\n")

	cur := g.files()
	keys := make([]string, 0, len(cur)+len(g.prev))
	for k := range cur {
		keys = append(keys, k)
	}
	for k := range g.prev {
		if _, ok := cur[k]; !ok {
			keys = append(keys, k)
		}
	}
	slices.Sort(keys)
	for _, k := range keys {
		body, ok := cur[k]
		prev, had := g.prev[k]
		switch {
		case !ok:
			fmt.Fprintf(g.sb, "--- removed %s\n", k)
		case !had || body != prev:
			fmt.Fprintf(g.sb, "+++ %s\n%s\n", k, body)
		}
	}
	g.prev = cur
}

// files maps every file under the test root to its content, with embedded markdown replaced by placeholders.
func (g *goldenRun) files() map[string]string {
	out := map[string]string{}
	err := filepath.WalkDir(g.root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		s := string(b)
		s = strings.ReplaceAll(s, manifests.SkillMarkdown, "<graph skill markdown>")
		s = strings.ReplaceAll(s, manifests.SubagentTasksSkillMarkdown, "<subagent-tasks skill markdown>")
		s = strings.ReplaceAll(s, strings.TrimRight(manifests.RulesPrompt, "\n"), "<rules prompt>")
		out[g.norm(p)] = s
		return nil
	})
	if err != nil {
		g.t.Fatal(err)
	}
	return out
}

func (g *goldenRun) norm(s string) string {
	s = strings.ReplaceAll(s, g.home, "$HOME")
	s = strings.ReplaceAll(s, g.repo, "$REPO")
	return filepath.ToSlash(s)
}

func (g *goldenRun) normAll(ss []string) []string {
	out := make([]string, len(ss))
	for i, s := range ss {
		out[i] = g.norm(s)
	}
	return out
}

func compareGolden(t *testing.T, path, got string) {
	t.Helper()
	if *updateGolden {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden (rerun with -update to create it): %v", err)
	}
	want := strings.ReplaceAll(string(raw), "\r\n", "\n")
	if got == want {
		return
	}
	gl, wl := strings.Split(got, "\n"), strings.Split(want, "\n")
	for i := range max(len(gl), len(wl)) {
		var g, w string
		if i < len(gl) {
			g = gl[i]
		}
		if i < len(wl) {
			w = wl[i]
		}
		if g != w {
			t.Fatalf("%s differs at line %d:\n got: %q\nwant: %q", path, i+1, g, w)
		}
	}
}
