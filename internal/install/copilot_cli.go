package install

import (
	"path/filepath"
)

// CopilotCLI installs chiton for the GitHub Copilot CLI terminal agent (not the VS Code extension): MCP in
// the user-scope ~/.copilot/mcp-config.json (or $COPILOT_HOME) and rules in AGENTS.md.
type CopilotCLI struct{}

func (CopilotCLI) Name() string        { return "copilot-cli" }
func (CopilotCLI) Description() string { return "GitHub Copilot CLI (terminal agent)" }

func (CopilotCLI) Detect(dir string) bool {
	// Workspace-scoped: a repo-local .copilot/ dir. The persistent MCP config is
	// user-global, but Detect reports evidence in this workspace only.
	return anyExists(dir, ".copilot")
}

// copilotEntry is Copilot CLI's documented local-server shape; bare {command,args} also works, but
// type/env/tools match the official example and are cross-client safe.
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

func (CopilotCLI) mcpPath() (string, error) {
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
	if err := rep.writeServer(mcpServersTable, mcpPath, chitonCopilotEntry(exe, opts.GraphPath)); err != nil {
		return rep, err
	}

	if err := rep.upsertRules(filepath.Join(opts.Dir, "AGENTS.md")); err != nil {
		return rep, err
	}

	rep.Notes = append(rep.Notes, "MCP config is user-scope (~/.copilot/mcp-config.json); it applies to Copilot CLI in every project.")
	return rep, nil
}

func (c CopilotCLI) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}

	mcpPath, err := c.mcpPath()
	if err != nil {
		return rep, err
	}
	if err := rep.removeServer(mcpServersTable, mcpPath); err != nil {
		return rep, err
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
	hasMCP := rep.checkMCP(mcpPath, mcpServersTable.has)
	hasRules := rep.checkRules(filepath.Join(opts.Dir, "AGENTS.md"))

	rep.Registered = hasMCP && hasRules
	return rep, nil
}

func init() { Register(CopilotCLI{}) }
