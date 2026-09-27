package install

import (
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/install/manifests"
)

// ClaudeCode installs chiton for Claude Code: .mcp.json, a CLAUDE.md rules block, and two skills under
// .claude/skills/, byte-identical to the plugin's because both embed the same skills/ files.
type ClaudeCode struct{}

// claudeSkills are written to .claude/skills/<Dir>/SKILL.md. The directory supplies the skill name, so the
// markdown carries no `name:` frontmatter and works unchanged inside the plugin's namespace.
var claudeSkills = []struct {
	Dir      string
	Markdown string
}{
	{Dir: "chiton", Markdown: manifests.SkillMarkdown},
	{Dir: "chiton-subagent-tasks", Markdown: manifests.SubagentTasksSkillMarkdown},
}

// claudeSkillPath is the on-disk location of one installed skill.
func claudeSkillPath(dir, skill string) string {
	return filepath.Join(dir, ".claude", "skills", skill, "SKILL.md")
}

func (ClaudeCode) Name() string        { return "claude-code" }
func (ClaudeCode) Description() string { return "Anthropic Claude Code (CLI + IDE extension)" }

func (ClaudeCode) Detect(dir string) bool {
	return anyExists(dir, "CLAUDE.md", ".claude")
}

func (c ClaudeCode) Install(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}
	if opts.UserScope {
		return rep, ErrUserScopeUnsupported
	}

	exe, err := ResolveExe(opts.Exe)
	if err != nil {
		return rep, err
	}

	mcpPath := filepath.Join(opts.Dir, ".mcp.json")
	if err := rep.writeServer(mcpServersTable, mcpPath, ChitonServerEntry(exe, opts.GraphPath)); err != nil {
		return rep, err
	}

	if err := rep.upsertRules(filepath.Join(opts.Dir, "CLAUDE.md")); err != nil {
		return rep, err
	}

	for _, skill := range claudeSkills {
		skillPath := claudeSkillPath(opts.Dir, skill.Dir)
		if err := writeFile(skillPath, []byte(skill.Markdown)); err != nil {
			return rep, err
		}
		rep.Wrote = append(rep.Wrote, skillPath)
	}

	if opts.WithHooks {
		rep.Notes = append(rep.Notes, "PreToolUse hooks not yet implemented; rerun without --with-hooks to silence this note")
	}

	rep.Notes = append(rep.Notes, "Restart Claude Code to pick up the new MCP server.")
	return rep, nil
}

func (c ClaudeCode) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}

	if err := rep.removeServer(mcpServersTable, filepath.Join(opts.Dir, ".mcp.json")); err != nil {
		return rep, err
	}

	if err := rep.removeRules(filepath.Join(opts.Dir, "CLAUDE.md")); err != nil {
		return rep, err
	}

	for _, skill := range claudeSkills {
		if err := rep.removeIfExists(filepath.Dir(claudeSkillPath(opts.Dir, skill.Dir)), os.RemoveAll); err != nil {
			return rep, err
		}
	}
	return rep, nil
}

func (c ClaudeCode) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: c.Name(), Detected: c.Detect(opts.Dir)}

	hasMCP := rep.checkMCP(filepath.Join(opts.Dir, ".mcp.json"), mcpServersTable.has)
	hasRules := rep.checkRules(filepath.Join(opts.Dir, "CLAUDE.md"))
	for _, skill := range claudeSkills {
		rep.check(claudeSkillPath(opts.Dir, skill.Dir), fileExists, ": skill installed", ": skill not installed")
	}

	rep.Registered = hasMCP && hasRules
	return rep, nil
}

func init() { Register(ClaudeCode{}) }
