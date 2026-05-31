package install

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/install/manifests"
)

// Antigravity installs chiton for Google's Antigravity IDE.
//
// Antigravity's MCP config is user/global only — there is no documented
// repo-relative MCP file — and lives in the unified ~/.gemini config dir shared
// with the Antigravity CLI and Gemini CLI (a different file from Gemini CLI's
// settings.json, so they don't collide):
//
//	~/.gemini/config/mcp_config.json ← MCP server registration (mcpServers key)
//	AGENTS.md                        ← always-on rules block (repo root)
//
// Rules go to AGENTS.md rather than GEMINI.md to avoid the global
// ~/.gemini/GEMINI.md collision with Gemini CLI.
type Antigravity struct{}

func (Antigravity) Name() string        { return "antigravity" }
func (Antigravity) Description() string { return "Google Antigravity IDE" }

func (Antigravity) Detect(dir string) bool {
	// .agent/rules is the strongest repo-local Antigravity signal (GEMINI.md /
	// AGENTS.md are shared with other tools, and the MCP config is user-global).
	_, err := os.Stat(filepath.Join(dir, ".agent", "rules"))
	return err == nil
}

func (a Antigravity) Install(opts Options) (Report, error) {
	rep := Report{Installer: a.Name()}

	exe, err := ResolveExe(opts.Exe)
	if err != nil {
		return rep, err
	}

	mcpPath, err := userPath(".gemini", "config", "mcp_config.json")
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

	rulesPath := filepath.Join(opts.Dir, "AGENTS.md")
	if _, _, err := UpsertRulesBlock(rulesPath, manifests.RulesPrompt); err != nil {
		return rep, err
	}
	rep.Wrote = append(rep.Wrote, rulesPath)

	rep.Notes = append(rep.Notes, "MCP config is user-scope (~/.gemini/config/mcp_config.json); it applies to Antigravity in every project.")
	return rep, nil
}

func (a Antigravity) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: a.Name()}

	mcpPath, err := userPath(".gemini", "config", "mcp_config.json")
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

	if err := uninstallSharedAgents(&rep, a.Name(), opts); err != nil {
		return rep, err
	}
	return rep, nil
}

func (a Antigravity) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: a.Name(), Detected: a.Detect(opts.Dir)}

	mcpPath, err := userPath(".gemini", "config", "mcp_config.json")
	if err != nil {
		return rep, err
	}
	hasMCP, err := MCPConfigHasChiton(mcpPath)
	if err != nil {
		rep.Issues = append(rep.Issues, fmt.Sprintf("%s: %v", mcpPath, err))
	} else if hasMCP {
		rep.OK = append(rep.OK, mcpPath+": chiton server registered")
	} else {
		rep.Issues = append(rep.Issues, mcpPath+": no chiton entry (run `chiton install --assistant antigravity`)")
	}

	rulesPath := filepath.Join(opts.Dir, "AGENTS.md")
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

func init() { Register(Antigravity{}) }
