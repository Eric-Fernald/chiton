package install

import (
	"path/filepath"
)

// OpenCode installs chiton for the OpenCode terminal agent: opencode.json (~/.config/opencode/opencode.json
// on every platform with --user-scope) plus a rules block in AGENTS.md.
type OpenCode struct{}

func (OpenCode) Name() string        { return "opencode" }
func (OpenCode) Description() string { return "OpenCode terminal agent" }

func (OpenCode) Detect(dir string) bool {
	return anyExists(dir, "opencode.json", "opencode.jsonc", filepath.Join(".opencode", "opencode.json"), ".opencode")
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
	if err := rep.writeServer(openCodeTable, mcpPath, chitonOpenCodeEntry(exe, opts.GraphPath)); err != nil {
		return rep, err
	}

	if err := rep.upsertRules(filepath.Join(opts.Dir, "AGENTS.md")); err != nil {
		return rep, err
	}

	rep.Notes = append(rep.Notes, "Restart OpenCode to pick up the new MCP server.")
	return rep, nil
}

func (o OpenCode) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: o.Name()}

	mcpPath, err := o.mcpPath(opts)
	if err != nil {
		return rep, err
	}
	if err := rep.removeServer(openCodeTable, mcpPath); err != nil {
		return rep, err
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
	hasMCP := rep.checkMCP(mcpPath, openCodeTable.has)
	hasRules := rep.checkRules(filepath.Join(opts.Dir, "AGENTS.md"))

	rep.Registered = hasMCP && hasRules
	return rep, nil
}

func init() { Register(OpenCode{}) }
