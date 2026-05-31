package install

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/install/manifests"
)

// GeminiCLI installs chiton for Google's Gemini CLI.
//
// Per-repo layout:
//
//	.gemini/settings.json            ← MCP server registration (mcpServers key)
//	GEMINI.md                        ← always-on context file (repo root)
//
// With --user-scope the MCP config is written to ~/.gemini/settings.json.
// Gemini CLI auto-loads GEMINI.md (not AGENTS.md) by default, so the rules go
// there.
type GeminiCLI struct{}

func (GeminiCLI) Name() string        { return "gemini-cli" }
func (GeminiCLI) Description() string { return "Google Gemini CLI" }

func (GeminiCLI) Detect(dir string) bool {
	for _, rel := range []string{
		filepath.Join(".gemini", "settings.json"),
		"GEMINI.md",
		".geminiignore",
		".gemini",
	} {
		if _, err := os.Stat(filepath.Join(dir, rel)); err == nil {
			return true
		}
	}
	return false
}

func (g GeminiCLI) mcpPath(opts Options) (string, error) {
	if opts.UserScope {
		return userPath(".gemini", "settings.json")
	}
	return filepath.Join(opts.Dir, ".gemini", "settings.json"), nil
}

func (g GeminiCLI) Install(opts Options) (Report, error) {
	rep := Report{Installer: g.Name()}

	exe, err := ResolveExe(opts.Exe)
	if err != nil {
		return rep, err
	}

	mcpPath, err := g.mcpPath(opts)
	if err != nil {
		return rep, err
	}
	created, preexisted, err := WriteMCPConfig(mcpPath, ChitonServerEntry(exe, opts.GraphPath))
	if err != nil {
		return rep, err
	}
	rep.Wrote = append(rep.Wrote, mcpPath)
	if !created && preexisted {
		rep.Notes = append(rep.Notes, fmt.Sprintf("%s: refreshed existing chiton entry", mcpPath))
	}

	rulesPath := filepath.Join(opts.Dir, "GEMINI.md")
	if _, _, err := UpsertRulesBlock(rulesPath, manifests.RulesPrompt); err != nil {
		return rep, err
	}
	rep.Wrote = append(rep.Wrote, rulesPath)

	rep.Notes = append(rep.Notes, "Restart Gemini CLI to pick up the new MCP server.")
	return rep, nil
}

func (g GeminiCLI) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: g.Name()}

	mcpPath, err := g.mcpPath(opts)
	if err != nil {
		return rep, err
	}
	changed, err := RemoveMCPConfigEntry(mcpPath)
	if err != nil {
		return rep, err
	}
	if changed {
		rep.Wrote = append(rep.Wrote, mcpPath)
	} else {
		rep.Skipped = append(rep.Skipped, mcpPath)
	}

	rulesPath := filepath.Join(opts.Dir, "GEMINI.md")
	changed, err = RemoveRulesBlock(rulesPath)
	if err != nil {
		return rep, err
	}
	if changed {
		rep.Wrote = append(rep.Wrote, rulesPath)
	} else {
		rep.Skipped = append(rep.Skipped, rulesPath)
	}
	return rep, nil
}

func (g GeminiCLI) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: g.Name(), Detected: g.Detect(opts.Dir)}

	mcpPath, err := g.mcpPath(opts)
	if err != nil {
		return rep, err
	}
	hasMCP, err := MCPConfigHasChiton(mcpPath)
	if err != nil {
		rep.Issues = append(rep.Issues, fmt.Sprintf("%s: %v", mcpPath, err))
	} else if hasMCP {
		rep.OK = append(rep.OK, mcpPath+": chiton server registered")
	} else {
		rep.Issues = append(rep.Issues, mcpPath+": no chiton entry (run `chiton install --assistant gemini-cli`)")
	}

	rulesPath := filepath.Join(opts.Dir, "GEMINI.md")
	hasRules, err := RulesFileHasChiton(rulesPath)
	if err != nil {
		rep.Issues = append(rep.Issues, fmt.Sprintf("%s: %v", rulesPath, err))
	} else if hasRules {
		rep.OK = append(rep.OK, rulesPath+": chiton rules block present")
	} else {
		rep.Issues = append(rep.Issues, rulesPath+": no chiton rules block")
	}

	rep.Registered = hasMCP && hasRules
	return rep, nil
}

func init() { Register(GeminiCLI{}) }
