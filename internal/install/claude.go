package install

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/install/manifests"
)

// ClaudeCode installs chiton for Anthropic's Claude Code CLI.
//
// Per-repo layout:
//
//	.mcp.json                          ← MCP server registration
//	CLAUDE.md                          ← always-on rules block
//	.claude/skills/chiton/SKILL.md     ← /chiton slash-command reference
type ClaudeCode struct{}

func (ClaudeCode) Name() string        { return "claude-code" }
func (ClaudeCode) Description() string { return "Anthropic Claude Code (CLI + IDE extension)" }

func (ClaudeCode) Detect(dir string) bool {
	// Any of: .mcp.json with a Claude convention, CLAUDE.md, or a .claude/
	// directory is enough evidence the user is using Claude Code here.
	for _, rel := range []string{"CLAUDE.md", ".claude"} {
		if _, err := os.Stat(filepath.Join(dir, rel)); err == nil {
			return true
		}
	}
	return false
}

func (c ClaudeCode) Install(opts Options) (Report, error) {
	if opts.UserScope {
		return Report{Installer: c.Name()}, ErrUserScopeUnsupported
	}
	rep := Report{Installer: c.Name()}

	exe, err := ResolveExe(opts.Exe)
	if err != nil {
		return rep, err
	}

	mcpPath := filepath.Join(opts.Dir, ".mcp.json")
	created, preexisted, err := WriteMCPConfig(mcpPath, ChitonServerEntry(exe, opts.GraphPath))
	if err != nil {
		return rep, err
	}
	rep.Wrote = append(rep.Wrote, mcpPath)
	if !created && preexisted {
		rep.Notes = append(rep.Notes, fmt.Sprintf("%s: refreshed existing chiton entry", mcpPath))
	}

	rulesPath := filepath.Join(opts.Dir, "CLAUDE.md")
	if _, _, err := UpsertRulesBlock(rulesPath, manifests.RulesPrompt); err != nil {
		return rep, err
	}
	rep.Wrote = append(rep.Wrote, rulesPath)

	skillPath := filepath.Join(opts.Dir, ".claude", "skills", "chiton", "SKILL.md")
	if err := os.MkdirAll(filepath.Dir(skillPath), 0o755); err != nil {
		return rep, fmt.Errorf("mkdir %s: %w", filepath.Dir(skillPath), err)
	}
	if err := os.WriteFile(skillPath, []byte(manifests.SkillMarkdown), 0o644); err != nil {
		return rep, fmt.Errorf("write %s: %w", skillPath, err)
	}
	rep.Wrote = append(rep.Wrote, skillPath)

	if opts.WithHooks {
		rep.Notes = append(rep.Notes, "PreToolUse hooks not yet implemented; rerun without --with-hooks to silence this note")
	}

	rep.Notes = append(rep.Notes, "Restart Claude Code to pick up the new MCP server.")
	return rep, nil
}

func (c ClaudeCode) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}

	mcpPath := filepath.Join(opts.Dir, ".mcp.json")
	changed, err := RemoveMCPConfigEntry(mcpPath)
	if err != nil {
		return rep, err
	}
	if changed {
		rep.Wrote = append(rep.Wrote, mcpPath)
	} else {
		rep.Skipped = append(rep.Skipped, mcpPath)
	}

	rulesPath := filepath.Join(opts.Dir, "CLAUDE.md")
	changed, err = RemoveRulesBlock(rulesPath)
	if err != nil {
		return rep, err
	}
	if changed {
		rep.Wrote = append(rep.Wrote, rulesPath)
	} else {
		rep.Skipped = append(rep.Skipped, rulesPath)
	}

	skillDir := filepath.Join(opts.Dir, ".claude", "skills", "chiton")
	if _, err := os.Stat(skillDir); err == nil {
		if err := os.RemoveAll(skillDir); err != nil {
			return rep, fmt.Errorf("remove %s: %w", skillDir, err)
		}
		rep.Wrote = append(rep.Wrote, skillDir)
	} else {
		rep.Skipped = append(rep.Skipped, skillDir)
	}
	return rep, nil
}

func (c ClaudeCode) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: c.Name(), Detected: c.Detect(opts.Dir)}

	mcpPath := filepath.Join(opts.Dir, ".mcp.json")
	hasMCP, err := MCPConfigHasChiton(mcpPath)
	if err != nil {
		rep.Issues = append(rep.Issues, fmt.Sprintf("%s: %v", mcpPath, err))
	} else if hasMCP {
		rep.OK = append(rep.OK, mcpPath+": chiton server registered")
	} else {
		rep.Issues = append(rep.Issues, mcpPath+": no chiton entry (run `chiton install --assistant claude-code`)")
	}

	rulesPath := filepath.Join(opts.Dir, "CLAUDE.md")
	hasRules, err := RulesFileHasChiton(rulesPath)
	if err != nil {
		rep.Issues = append(rep.Issues, fmt.Sprintf("%s: %v", rulesPath, err))
	} else if hasRules {
		rep.OK = append(rep.OK, rulesPath+": chiton rules block present")
	} else {
		rep.Issues = append(rep.Issues, rulesPath+": no chiton rules block")
	}

	skillPath := filepath.Join(opts.Dir, ".claude", "skills", "chiton", "SKILL.md")
	if _, err := os.Stat(skillPath); err == nil {
		rep.OK = append(rep.OK, skillPath+": skill installed")
	} else {
		rep.Issues = append(rep.Issues, skillPath+": skill not installed")
	}

	rep.Registered = hasMCP && hasRules
	return rep, nil
}

func init() { Register(ClaudeCode{}) }
