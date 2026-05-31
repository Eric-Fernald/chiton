package install

import (
	"path/filepath"
)

// Several installers (Codex, OpenCode, Antigravity, Copilot CLI) all read the
// same repo-root AGENTS.md, so they share one chiton rules block there. That
// sharing is correct on install (UpsertRulesBlock keeps a single block), but
// naive removal on uninstall would strip the block — and possibly delete
// AGENTS.md — out from under the other assistants still installed, violating
// the Uninstall contract ("removes only entries chiton owns, never the user's
// other tool registrations"). The block is therefore reference-counted: it is
// removed only when the last AGENTS.md-using assistant is uninstalled.

// agentsMDUsers are the installer names whose rules live in the shared
// AGENTS.md. Installers with a dedicated rules file (Claude Code → CLAUDE.md,
// Gemini CLI → GEMINI.md, Cursor → .cursor/rules, Copilot VS Code →
// copilot-instructions.md, Aider → CONVENTIONS.md) are not listed here.
var agentsMDUsers = []string{"codex", "opencode", "antigravity", "copilot-cli"}

// agentsMDUserHasMCP reports whether the named AGENTS.md-using installer still
// has its chiton MCP entry registered. It checks every scope that installer can
// write to, so a scope mismatch between uninstalls errs toward keeping the
// shared block rather than orphaning a sibling.
func agentsMDUserHasMCP(name string, opts Options) bool {
	switch name {
	case "codex":
		if p, err := userPath(".codex", "config.toml"); err == nil {
			if ok, _ := CodexConfigHasChiton(p); ok {
				return true
			}
		}
	case "opencode":
		if ok, _ := OpenCodeConfigHasChiton(filepath.Join(opts.Dir, "opencode.json")); ok {
			return true
		}
		if p, err := userPath(".config", "opencode", "opencode.json"); err == nil {
			if ok, _ := OpenCodeConfigHasChiton(p); ok {
				return true
			}
		}
	case "antigravity":
		if p, err := userPath(".gemini", "config", "mcp_config.json"); err == nil {
			if ok, _ := MCPConfigHasChiton(p); ok {
				return true
			}
		}
	case "copilot-cli":
		if p, err := (CopilotCLI{}).mcpPath(); err == nil {
			if ok, _ := MCPConfigHasChiton(p); ok {
				return true
			}
		}
	}
	return false
}

// removeSharedAgentsBlock strips chiton's block from the shared AGENTS.md in
// opts.Dir, but only when no AGENTS.md-using installer other than self still has
// its MCP registered. self is the installer being uninstalled (its own MCP
// entry should already be removed before this is called).
//
// Returns whether the block was removed, and whether it was deliberately kept
// because another assistant still depends on it.
func removeSharedAgentsBlock(self string, opts Options) (removed, kept bool, err error) {
	for _, name := range agentsMDUsers {
		if name == self {
			continue
		}
		if agentsMDUserHasMCP(name, opts) {
			return false, true, nil
		}
	}
	removed, err = RemoveRulesBlock(filepath.Join(opts.Dir, "AGENTS.md"))
	return removed, false, err
}

// uninstallSharedAgents removes the shared AGENTS.md block (reference-counted)
// and records the outcome on rep. self is the installer being uninstalled.
func uninstallSharedAgents(rep *Report, self string, opts Options) error {
	rulesPath := filepath.Join(opts.Dir, "AGENTS.md")
	removed, kept, err := removeSharedAgentsBlock(self, opts)
	if err != nil {
		return err
	}
	switch {
	case removed:
		rep.Wrote = append(rep.Wrote, rulesPath)
	case kept:
		rep.Skipped = append(rep.Skipped, rulesPath)
		rep.Notes = append(rep.Notes, rulesPath+": kept chiton rules block (still used by another installed assistant)")
	default:
		rep.Skipped = append(rep.Skipped, rulesPath)
	}
	return nil
}
