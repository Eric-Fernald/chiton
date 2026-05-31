package install

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/install/manifests"
)

// CopilotCLI installs chiton for the GitHub Copilot CLI (the `copilot`
// terminal agent — distinct from the VS Code Copilot extension, which
// CopilotVSCode handles).
//
// The fully-documented, no-caveats target is the user-scope config file
// (~/.copilot/mcp-config.json, or $COPILOT_HOME/mcp-config.json). Rules go to
// AGENTS.md, Copilot CLI's primary repo instructions file.
//
//	~/.copilot/mcp-config.json       ← MCP server registration (mcpServers key)
//	AGENTS.md                        ← always-on rules block (repo root)
type CopilotCLI struct{}

func (CopilotCLI) Name() string        { return "copilot-cli" }
func (CopilotCLI) Description() string { return "GitHub Copilot CLI (terminal agent)" }

func (CopilotCLI) Detect(dir string) bool {
	// Workspace-scoped: a repo-local .copilot/ dir. The persistent MCP config is
	// user-global, but Detect reports evidence in this workspace only.
	_, err := os.Stat(filepath.Join(dir, ".copilot"))
	return err == nil
}

// copilotEntry is Copilot CLI's documented local-server entry shape. The bare
// {command,args} form also works, but emitting type/env/tools matches the
// official example and is cross-client safe.
type copilotEntry struct {
	Type    string            `json:"type"`
	Command string            `json:"command"`
	Args    []string          `json:"args"`
	Env     map[string]string `json:"env"`
	Tools   []string          `json:"tools"`
}

func chitonCopilotEntry(exe, graphPath string) copilotEntry {
	return copilotEntry{
		Type:    "local",
		Command: exe,
		Args:    ChitonServeArgs(graphPath),
		Env:     map[string]string{},
		Tools:   []string{"*"},
	}
}

func (c CopilotCLI) mcpPath() (string, error) {
	dir, err := copilotConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "mcp-config.json"), nil
}

func (c CopilotCLI) Install(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}

	exe, err := ResolveExe(opts.Exe)
	if err != nil {
		return rep, err
	}

	mcpPath, err := c.mcpPath()
	if err != nil {
		return rep, err
	}
	created, preexisted, err := WriteMCPConfig(mcpPath, chitonCopilotEntry(exe, opts.GraphPath))
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

	rep.Notes = append(rep.Notes, "MCP config is user-scope (~/.copilot/mcp-config.json); it applies to Copilot CLI in every project.")
	return rep, nil
}

func (c CopilotCLI) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}

	mcpPath, err := c.mcpPath()
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

	if err := uninstallSharedAgents(&rep, c.Name(), opts); err != nil {
		return rep, err
	}
	return rep, nil
}

func (c CopilotCLI) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: c.Name(), Detected: c.Detect(opts.Dir)}

	mcpPath, err := c.mcpPath()
	if err != nil {
		return rep, err
	}
	hasMCP, err := MCPConfigHasChiton(mcpPath)
	if err != nil {
		rep.Issues = append(rep.Issues, fmt.Sprintf("%s: %v", mcpPath, err))
	} else if hasMCP {
		rep.OK = append(rep.OK, mcpPath+": chiton server registered")
	} else {
		rep.Issues = append(rep.Issues, mcpPath+": no chiton entry (run `chiton install --assistant copilot-cli`)")
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

func init() { Register(CopilotCLI{}) }
