package install

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/install/manifests"
)

// Cursor installs chiton for the Cursor editor (and its CLI, which shares the
// same config).
//
// Per-repo layout:
//
//	.cursor/mcp.json                 ← MCP server registration (mcpServers key)
//	.cursor/rules/chiton.mdc         ← always-on project rule (MDC + frontmatter)
//
// With --user-scope the MCP config is written to ~/.cursor/mcp.json instead;
// the project rule is always written into the repo.
type Cursor struct{}

func (Cursor) Name() string        { return "cursor" }
func (Cursor) Description() string { return "Cursor editor + CLI" }

func (Cursor) Detect(dir string) bool {
	for _, rel := range []string{
		filepath.Join(".cursor", "mcp.json"),
		filepath.Join(".cursor", "rules"),
		".cursorrules",
		".cursor",
	} {
		if _, err := os.Stat(filepath.Join(dir, rel)); err == nil {
			return true
		}
	}
	return false
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
	created, preexisted, err := WriteMCPConfig(mcpPath, ChitonServerEntry(exe, opts.GraphPath))
	if err != nil {
		return rep, err
	}
	rep.Wrote = append(rep.Wrote, mcpPath)
	if !created && preexisted {
		rep.Notes = append(rep.Notes, fmt.Sprintf("%s: refreshed existing chiton entry", mcpPath))
	}

	rulePath := c.rulePath(opts)
	if err := os.MkdirAll(filepath.Dir(rulePath), 0o755); err != nil {
		return rep, fmt.Errorf("mkdir %s: %w", filepath.Dir(rulePath), err)
	}
	if err := os.WriteFile(rulePath, []byte(cursorRuleContent()), 0o644); err != nil {
		return rep, fmt.Errorf("write %s: %w", rulePath, err)
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
	changed, err := RemoveMCPConfigEntry(mcpPath)
	if err != nil {
		return rep, err
	}
	if changed {
		rep.Wrote = append(rep.Wrote, mcpPath)
	} else {
		rep.Skipped = append(rep.Skipped, mcpPath)
	}

	rulePath := c.rulePath(opts)
	if _, err := os.Stat(rulePath); err == nil {
		if err := os.Remove(rulePath); err != nil {
			return rep, fmt.Errorf("remove %s: %w", rulePath, err)
		}
		rep.Wrote = append(rep.Wrote, rulePath)
	} else {
		rep.Skipped = append(rep.Skipped, rulePath)
	}
	return rep, nil
}

func (c Cursor) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: c.Name(), Detected: c.Detect(opts.Dir)}

	mcpPath, err := c.mcpPath(opts)
	if err != nil {
		return rep, err
	}
	hasMCP, err := MCPConfigHasChiton(mcpPath)
	if err != nil {
		rep.Issues = append(rep.Issues, fmt.Sprintf("%s: %v", mcpPath, err))
	} else if hasMCP {
		rep.OK = append(rep.OK, mcpPath+": chiton server registered")
	} else {
		rep.Issues = append(rep.Issues, mcpPath+": no chiton entry (run `chiton install --assistant cursor`)")
	}

	rulePath := c.rulePath(opts)
	hasRule := false
	if _, err := os.Stat(rulePath); err == nil {
		hasRule = true
		rep.OK = append(rep.OK, rulePath+": chiton project rule present")
	} else {
		rep.Issues = append(rep.Issues, rulePath+": no chiton rule")
	}

	rep.Registered = hasMCP && hasRule
	return rep, nil
}

// cursorRuleContent wraps the shared rules prompt in the MDC frontmatter Cursor
// requires. A .cursor/rules file without frontmatter is silently ignored;
// alwaysApply:true keeps the rule in context for every request.
func cursorRuleContent() string {
	return "---\n" +
		"description: chiton knowledge-graph context — call chiton MCP tools before grep/file reads for architecture and navigation questions\n" +
		"alwaysApply: true\n" +
		"---\n\n" +
		manifests.RulesPrompt
}

func init() { Register(Cursor{}) }
