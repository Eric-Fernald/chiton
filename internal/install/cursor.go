package install

import (
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/install/manifests"
)

// Cursor installs chiton for the Cursor editor and CLI: .cursor/mcp.json (~/.cursor/mcp.json with
// --user-scope) plus an always-on .cursor/rules/chiton.mdc, which is always written into the repo.
type Cursor struct{}

func (Cursor) Name() string        { return "cursor" }
func (Cursor) Description() string { return "Cursor editor + CLI" }

func (Cursor) Detect(dir string) bool {
	return anyExists(dir, filepath.Join(".cursor", "mcp.json"), filepath.Join(".cursor", "rules"), ".cursorrules", ".cursor")
}

func (c Cursor) mcpPath(opts Options) (string, error) {
	if opts.UserScope {
		return userPath(".cursor", "mcp.json")
	}
	return filepath.Join(opts.Dir, ".cursor", "mcp.json"), nil
}

func (c Cursor) rulePath(opts Options) string {
	return filepath.Join(opts.Dir, ".cursor", "rules", "chiton.mdc")
}

func (c Cursor) Install(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}

	exe, err := ResolveExe(opts.Exe)
	if err != nil {
		return rep, err
	}

	mcpPath, err := c.mcpPath(opts)
	if err != nil {
		return rep, err
	}
	if err := rep.writeServer(mcpServersTable, mcpPath, ChitonServerEntry(exe, opts.GraphPath)); err != nil {
		return rep, err
	}

	rulePath := c.rulePath(opts)
	if err := writeFile(rulePath, []byte(cursorRuleContent())); err != nil {
		return rep, err
	}
	rep.Wrote = append(rep.Wrote, rulePath)

	rep.Notes = append(rep.Notes, "Restart Cursor (or reload the window) to pick up the new MCP server.")
	return rep, nil
}

func (c Cursor) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}

	mcpPath, err := c.mcpPath(opts)
	if err != nil {
		return rep, err
	}
	if err := rep.removeServer(mcpServersTable, mcpPath); err != nil {
		return rep, err
	}

	if err := rep.removeIfExists(c.rulePath(opts), os.Remove); err != nil {
		return rep, err
	}
	return rep, nil
}

func (c Cursor) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: c.Name(), Detected: c.Detect(opts.Dir)}

	mcpPath, err := c.mcpPath(opts)
	if err != nil {
		return rep, err
	}
	hasMCP := rep.checkMCP(mcpPath, mcpServersTable.has)
	hasRule := rep.check(c.rulePath(opts), fileExists, ": chiton project rule present", ": no chiton rule")

	rep.Registered = hasMCP && hasRule
	return rep, nil
}

// cursorRuleContent wraps RulesPrompt in the MDC frontmatter Cursor requires: a rule without frontmatter
// is silently ignored, and alwaysApply keeps it in context for every request.
func cursorRuleContent() string {
	return "---\n" +
		"description: chiton knowledge-graph context — call chiton MCP tools before grep/file reads for architecture and navigation questions\n" +
		"alwaysApply: true\n" +
		"---\n\n" +
		manifests.RulesPrompt
}

func init() { Register(Cursor{}) }
