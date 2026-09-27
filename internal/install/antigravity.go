package install

import (
	"path/filepath"
)

// Antigravity installs chiton for Google Antigravity. Its MCP config is user-global only
// (~/.gemini/config/mcp_config.json). Rules go to AGENTS.md to avoid colliding with Gemini CLI's GEMINI.md.
type Antigravity struct{}

func (Antigravity) Name() string        { return "antigravity" }
func (Antigravity) Description() string { return "Google Antigravity IDE" }

func (Antigravity) Detect(dir string) bool {
	// .agent/rules is the strongest repo-local Antigravity signal (GEMINI.md /
	// AGENTS.md are shared with other tools, and the MCP config is user-global).
	return anyExists(dir, filepath.Join(".agent", "rules"))
}

func (Antigravity) mcpPath() (string, error) {
	return userPath(".gemini", "config", "mcp_config.json")
}

func (a Antigravity) Install(opts Options) (Report, error) {
	rep := Report{Installer: a.Name()}

	exe, err := ResolveExe(opts.Exe)
	if err != nil {
		return rep, err
	}

	mcpPath, err := a.mcpPath()
	if err != nil {
		return rep, err
	}
	if err := rep.writeServer(mcpServersTable, mcpPath, ChitonServerEntry(exe, opts.GraphPath)); err != nil {
		return rep, err
	}

	if err := rep.upsertRules(filepath.Join(opts.Dir, "AGENTS.md")); err != nil {
		return rep, err
	}

	rep.Notes = append(rep.Notes, "MCP config is user-scope (~/.gemini/config/mcp_config.json); it applies to Antigravity in every project.")
	return rep, nil
}

func (a Antigravity) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: a.Name()}

	mcpPath, err := a.mcpPath()
	if err != nil {
		return rep, err
	}
	if err := rep.removeServer(mcpServersTable, mcpPath); err != nil {
		return rep, err
	}

	if err := uninstallSharedAgents(&rep, a.Name(), opts); err != nil {
		return rep, err
	}
	return rep, nil
}

func (a Antigravity) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: a.Name(), Detected: a.Detect(opts.Dir)}

	mcpPath, err := a.mcpPath()
	if err != nil {
		return rep, err
	}
	hasMCP := rep.checkMCP(mcpPath, mcpServersTable.has)
	hasRules := rep.checkRules(filepath.Join(opts.Dir, "AGENTS.md"))

	rep.Registered = hasMCP && hasRules
	return rep, nil
}

func init() { Register(Antigravity{}) }
