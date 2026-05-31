package install

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/install/manifests"
)

// OpenCode installs chiton for the OpenCode terminal agent.
//
// Per-repo layout:
//
//	opencode.json                    ← MCP server under the "mcp" key
//	AGENTS.md                        ← always-on rules block (repo root)
//
// With --user-scope the MCP config is written to
// ~/.config/opencode/opencode.json (the same path on every platform).
type OpenCode struct{}

func (OpenCode) Name() string        { return "opencode" }
func (OpenCode) Description() string { return "OpenCode terminal agent" }

func (OpenCode) Detect(dir string) bool {
	for _, rel := range []string{
		"opencode.json",
		"opencode.jsonc",
		filepath.Join(".opencode", "opencode.json"),
		".opencode",
	} {
		if _, err := os.Stat(filepath.Join(dir, rel)); err == nil {
			return true
		}
	}
	return false
}

func (o OpenCode) mcpPath(opts Options) (string, error) {
	if opts.UserScope {
		return userPath(".config", "opencode", "opencode.json")
	}
	return filepath.Join(opts.Dir, "opencode.json"), nil
}

func (o OpenCode) Install(opts Options) (Report, error) {
	rep := Report{Installer: o.Name()}

	exe, err := ResolveExe(opts.Exe)
	if err != nil {
		return rep, err
	}

	mcpPath, err := o.mcpPath(opts)
	if err != nil {
		return rep, err
	}
	created, preexisted, err := WriteOpenCodeConfig(mcpPath, exe, opts.GraphPath)
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

	rep.Notes = append(rep.Notes, "Restart OpenCode to pick up the new MCP server.")
	return rep, nil
}

func (o OpenCode) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: o.Name()}

	mcpPath, err := o.mcpPath(opts)
	if err != nil {
		return rep, err
	}
	changed, err := RemoveOpenCodeEntry(mcpPath)
	if err != nil {
		return rep, err
	}
	if changed {
		rep.Wrote = append(rep.Wrote, mcpPath)
	} else {
		rep.Skipped = append(rep.Skipped, mcpPath)
	}

	if err := uninstallSharedAgents(&rep, o.Name(), opts); err != nil {
		return rep, err
	}
	return rep, nil
}

func (o OpenCode) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: o.Name(), Detected: o.Detect(opts.Dir)}

	mcpPath, err := o.mcpPath(opts)
	if err != nil {
		return rep, err
	}
	hasMCP, err := OpenCodeConfigHasChiton(mcpPath)
	if err != nil {
		rep.Issues = append(rep.Issues, fmt.Sprintf("%s: %v", mcpPath, err))
	} else if hasMCP {
		rep.OK = append(rep.OK, mcpPath+": chiton server registered")
	} else {
		rep.Issues = append(rep.Issues, mcpPath+": no chiton entry (run `chiton install --assistant opencode`)")
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

func init() { Register(OpenCode{}) }
