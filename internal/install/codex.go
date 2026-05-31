package install

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/install/manifests"
)

// Codex installs chiton for the OpenAI Codex CLI.
//
// Codex keeps all settings in a single user-scope TOML file; MCP servers are
// [mcp_servers.<name>] tables there. The project-scope .codex/config.toml is
// loaded only for trusted projects, so the installer targets the always-honored
// user file and writes repo rules to AGENTS.md.
//
//	~/.codex/config.toml             ← [mcp_servers.chiton] table
//	AGENTS.md                        ← always-on rules block (repo root)
type Codex struct{}

func (Codex) Name() string        { return "codex" }
func (Codex) Description() string { return "OpenAI Codex CLI" }

func (Codex) Detect(dir string) bool {
	// Workspace-scoped detection only (a repo-local .codex/ dir). The MCP config
	// itself is user-global, but Detect describes evidence in this workspace.
	_, err := os.Stat(filepath.Join(dir, ".codex"))
	return err == nil
}

func (c Codex) Install(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}

	exe, err := ResolveExe(opts.Exe)
	if err != nil {
		return rep, err
	}

	cfgPath, err := userPath(".codex", "config.toml")
	if err != nil {
		return rep, err
	}
	created, preexisted, err := WriteCodexConfig(cfgPath, exe, opts.GraphPath)
	if err != nil {
		return rep, err
	}
	rep.Wrote = append(rep.Wrote, cfgPath)
	if !created && preexisted {
		rep.Notes = append(rep.Notes, fmt.Sprintf("%s: refreshed existing chiton entry", cfgPath))
	}

	rulesPath := filepath.Join(opts.Dir, "AGENTS.md")
	if _, _, err := UpsertRulesBlock(rulesPath, manifests.RulesPrompt); err != nil {
		return rep, err
	}
	rep.Wrote = append(rep.Wrote, rulesPath)

	rep.Notes = append(rep.Notes, "MCP config is user-scope (~/.codex/config.toml); it applies to Codex in every project.")
	return rep, nil
}

func (c Codex) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}

	cfgPath, err := userPath(".codex", "config.toml")
	if err != nil {
		return rep, err
	}
	changed, err := RemoveCodexEntry(cfgPath)
	if err != nil {
		return rep, err
	}
	if changed {
		rep.Wrote = append(rep.Wrote, cfgPath)
	} else {
		rep.Skipped = append(rep.Skipped, cfgPath)
	}

	if err := uninstallSharedAgents(&rep, c.Name(), opts); err != nil {
		return rep, err
	}
	return rep, nil
}

func (c Codex) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: c.Name(), Detected: c.Detect(opts.Dir)}

	cfgPath, err := userPath(".codex", "config.toml")
	if err != nil {
		return rep, err
	}
	hasMCP, err := CodexConfigHasChiton(cfgPath)
	if err != nil {
		rep.Issues = append(rep.Issues, fmt.Sprintf("%s: %v", cfgPath, err))
	} else if hasMCP {
		rep.OK = append(rep.OK, cfgPath+": chiton server registered")
	} else {
		rep.Issues = append(rep.Issues, cfgPath+": no chiton entry (run `chiton install --assistant codex`)")
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

func init() { Register(Codex{}) }
