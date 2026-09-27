package install

import (
	"path/filepath"
)

// CopilotVSCode installs chiton for GitHub Copilot Chat in VS Code: .vscode/mcp.json plus an always-on
// block in .github/copilot-instructions.md.
type CopilotVSCode struct{}

func (CopilotVSCode) Name() string { return "copilot-vscode" }
func (CopilotVSCode) Description() string {
	return "GitHub Copilot Chat in VS Code"
}

func (CopilotVSCode) Detect(dir string) bool {
	return anyExists(dir, filepath.Join(".github", "copilot-instructions.md"), filepath.Join(".vscode", "mcp.json"), ".vscode")
}

func (c CopilotVSCode) Install(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}
	if opts.UserScope {
		return rep, ErrUserScopeUnsupported
	}

	exe, err := ResolveExe(opts.Exe)
	if err != nil {
		return rep, err
	}

	mcpPath := filepath.Join(opts.Dir, ".vscode", "mcp.json")
	if err := rep.writeServer(mcpServersTable, mcpPath, ChitonServerEntry(exe, opts.GraphPath)); err != nil {
		return rep, err
	}

	if err := rep.upsertRules(filepath.Join(opts.Dir, ".github", "copilot-instructions.md")); err != nil {
		return rep, err
	}

	rep.Notes = append(rep.Notes, "Reload the VS Code window to pick up the new MCP server.")
	return rep, nil
}

func (c CopilotVSCode) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}

	if err := rep.removeServer(mcpServersTable, filepath.Join(opts.Dir, ".vscode", "mcp.json")); err != nil {
		return rep, err
	}

	if err := rep.removeRules(filepath.Join(opts.Dir, ".github", "copilot-instructions.md")); err != nil {
		return rep, err
	}
	return rep, nil
}

func (c CopilotVSCode) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: c.Name(), Detected: c.Detect(opts.Dir)}

	hasMCP := rep.checkMCP(filepath.Join(opts.Dir, ".vscode", "mcp.json"), mcpServersTable.has)
	hasRules := rep.checkRules(filepath.Join(opts.Dir, ".github", "copilot-instructions.md"))

	rep.Registered = hasMCP && hasRules
	return rep, nil
}

func init() { Register(CopilotVSCode{}) }
