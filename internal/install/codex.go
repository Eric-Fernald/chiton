package install

import (
	"path/filepath"
)

// Codex installs chiton for the OpenAI Codex CLI: [mcp_servers.chiton] in the user-scope
// ~/.codex/config.toml (the project file loads only for trusted projects) plus rules in AGENTS.md.
type Codex struct{}

func (Codex) Name() string        { return "codex" }
func (Codex) Description() string { return "OpenAI Codex CLI" }

func (Codex) Detect(dir string) bool {
	// Workspace-scoped detection only (a repo-local .codex/ dir). The MCP config
	// itself is user-global, but Detect describes evidence in this workspace.
	return anyExists(dir, ".codex")
}

func (Codex) mcpPath() (string, error) {
	return userPath(".codex", "config.toml")
}

func (c Codex) Install(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}

	exe, err := ResolveExe(opts.Exe)
	if err != nil {
		return rep, err
	}

	cfgPath, err := c.mcpPath()
	if err != nil {
		return rep, err
	}
	if err := rep.writeServer(codexTable, cfgPath, chitonCodexEntry(exe, opts.GraphPath)); err != nil {
		return rep, err
	}

	if err := rep.upsertRules(filepath.Join(opts.Dir, "AGENTS.md")); err != nil {
		return rep, err
	}

	rep.Notes = append(rep.Notes, "MCP config is user-scope (~/.codex/config.toml); it applies to Codex in every project.")
	return rep, nil
}

func (c Codex) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}

	cfgPath, err := c.mcpPath()
	if err != nil {
		return rep, err
	}
	if err := rep.removeServer(codexTable, cfgPath); err != nil {
		return rep, err
	}

	if err := uninstallSharedAgents(&rep, c.Name(), opts); err != nil {
		return rep, err
	}
	return rep, nil
}

func (c Codex) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: c.Name(), Detected: c.Detect(opts.Dir)}

	cfgPath, err := c.mcpPath()
	if err != nil {
		return rep, err
	}
	hasMCP := rep.checkMCP(cfgPath, codexTable.has)
	hasRules := rep.checkRules(filepath.Join(opts.Dir, "AGENTS.md"))

	rep.Registered = hasMCP && hasRules
	return rep, nil
}

func init() { Register(Codex{}) }
