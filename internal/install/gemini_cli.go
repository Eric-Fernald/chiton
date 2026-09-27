package install

import (
	"path/filepath"
)

// GeminiCLI installs chiton for Google's Gemini CLI: .gemini/settings.json (~/.gemini/settings.json with
// --user-scope) plus a rules block in GEMINI.md, which Gemini CLI auto-loads instead of AGENTS.md.
type GeminiCLI struct{}

func (GeminiCLI) Name() string        { return "gemini-cli" }
func (GeminiCLI) Description() string { return "Google Gemini CLI" }

func (GeminiCLI) Detect(dir string) bool {
	return anyExists(dir, filepath.Join(".gemini", "settings.json"), "GEMINI.md", ".geminiignore", ".gemini")
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
	if err := rep.writeServer(mcpServersTable, mcpPath, ChitonServerEntry(exe, opts.GraphPath)); err != nil {
		return rep, err
	}

	if err := rep.upsertRules(filepath.Join(opts.Dir, "GEMINI.md")); err != nil {
		return rep, err
	}

	rep.Notes = append(rep.Notes, "Restart Gemini CLI to pick up the new MCP server.")
	return rep, nil
}

func (g GeminiCLI) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: g.Name()}

	mcpPath, err := g.mcpPath(opts)
	if err != nil {
		return rep, err
	}
	if err := rep.removeServer(mcpServersTable, mcpPath); err != nil {
		return rep, err
	}

	if err := rep.removeRules(filepath.Join(opts.Dir, "GEMINI.md")); err != nil {
		return rep, err
	}
	return rep, nil
}

func (g GeminiCLI) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: g.Name(), Detected: g.Detect(opts.Dir)}

	mcpPath, err := g.mcpPath(opts)
	if err != nil {
		return rep, err
	}
	hasMCP := rep.checkMCP(mcpPath, mcpServersTable.has)
	hasRules := rep.checkRules(filepath.Join(opts.Dir, "GEMINI.md"))

	rep.Registered = hasMCP && hasRules
	return rep, nil
}

func init() { Register(GeminiCLI{}) }
