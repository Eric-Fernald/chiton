package install

import (
	"path/filepath"
)

// AGENTS.md is shared by several installers, so its chiton block is reference-counted: uninstalling one
// removes it only when no other AGENTS.md user still has its MCP entry (Uninstall must not break siblings).

// agentsMDUsers are the installer names whose rules live in the shared AGENTS.md.
var agentsMDUsers = []string{"codex", "opencode", "antigravity", "copilot-cli"}

// agentsMDUserHasMCP reports whether the named installer still has its chiton MCP entry in any scope it can
// write, so a scope mismatch between uninstalls errs toward keeping the shared block.
func agentsMDUserHasMCP(name string, opts Options) bool {
	registered := func(path string, err error, probe func(string) (bool, error)) bool {
		if err != nil {
			return false
		}
		ok, _ := probe(path)
		return ok
	}
	switch name {
	case "codex":
		p, err := Codex{}.mcpPath()
		return registered(p, err, codexTable.has)
	case "opencode":
		for _, userScope := range []bool{false, true} {
			scoped := opts
			scoped.UserScope = userScope
			if p, err := (OpenCode{}).mcpPath(scoped); registered(p, err, openCodeTable.has) {
				return true
			}
		}
	case "antigravity":
		p, err := Antigravity{}.mcpPath()
		return registered(p, err, mcpServersTable.has)
	case "copilot-cli":
		p, err := CopilotCLI{}.mcpPath()
		return registered(p, err, mcpServersTable.has)
	}
	return false
}

// uninstallSharedAgents strips the AGENTS.md block and records the outcome on rep, unless an installer other
// than self (the one being uninstalled, whose MCP entry must already be gone) still has its MCP registered.
func uninstallSharedAgents(rep *Report, self string, opts Options) error {
	rulesPath := filepath.Join(opts.Dir, "AGENTS.md")
	for _, name := range agentsMDUsers {
		if name != self && agentsMDUserHasMCP(name, opts) {
			rep.record(rulesPath, false)
			rep.Notes = append(rep.Notes, rulesPath+": kept chiton rules block (still used by another installed assistant)")
			return nil
		}
	}
	return rep.removeRules(rulesPath)
}
