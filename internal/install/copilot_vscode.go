package install

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/install/manifests"
)

// CopilotVSCode installs chiton for GitHub Copilot Chat in VS Code.
//
// Per-repo layout:
//
//	.vscode/mcp.json                                 ← MCP server registration
//	.github/copilot-instructions.md                  ← always-on instructions
type CopilotVSCode struct{}

func (CopilotVSCode) Name() string { return "copilot-vscode" }
func (CopilotVSCode) Description() string {
	return "GitHub Copilot Chat in VS Code"
}

func (CopilotVSCode) Detect(dir string) bool {
	for _, rel := range []string{
		filepath.Join(".github", "copilot-instructions.md"),
		filepath.Join(".vscode", "mcp.json"),
		".vscode",
	} {
		if _, err := os.Stat(filepath.Join(dir, rel)); err == nil {
			return true
		}
	}
	return false
}

func (c CopilotVSCode) Install(opts Options) (Report, error) {
	if opts.UserScope {
		return Report{Installer: c.Name()}, ErrUserScopeUnsupported
	}
	rep := Report{Installer: c.Name()}

	exe, err := ResolveExe(opts.Exe)
	if err != nil {
		return rep, err
	}

	mcpPath := filepath.Join(opts.Dir, ".vscode", "mcp.json")
	created, preexisted, err := WriteMCPConfig(mcpPath, ChitonServerEntry(exe, opts.GraphPath))
	if err != nil {
		return rep, err
	}
	rep.Wrote = append(rep.Wrote, mcpPath)
	if !created && preexisted {
		rep.Notes = append(rep.Notes, fmt.Sprintf("%s: refreshed existing chiton entry", mcpPath))
	}

	rulesPath := filepath.Join(opts.Dir, ".github", "copilot-instructions.md")
	if _, _, err := UpsertRulesBlock(rulesPath, manifests.RulesPrompt); err != nil {
		return rep, err
	}
	rep.Wrote = append(rep.Wrote, rulesPath)

	rep.Notes = append(rep.Notes, "Reload the VS Code window to pick up the new MCP server.")
	return rep, nil
}

func (c CopilotVSCode) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: c.Name()}

	mcpPath := filepath.Join(opts.Dir, ".vscode", "mcp.json")
	changed, err := RemoveMCPConfigEntry(mcpPath)
	if err != nil {
		return rep, err
	}
	if changed {
		rep.Wrote = append(rep.Wrote, mcpPath)
	} else {
		rep.Skipped = append(rep.Skipped, mcpPath)
	}

	rulesPath := filepath.Join(opts.Dir, ".github", "copilot-instructions.md")
	changed, err = RemoveRulesBlock(rulesPath)
	if err != nil {
		return rep, err
	}
	if changed {
		rep.Wrote = append(rep.Wrote, rulesPath)
	} else {
		rep.Skipped = append(rep.Skipped, rulesPath)
	}
	return rep, nil
}

func (c CopilotVSCode) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: c.Name(), Detected: c.Detect(opts.Dir)}

	mcpPath := filepath.Join(opts.Dir, ".vscode", "mcp.json")
	hasMCP, err := MCPConfigHasChiton(mcpPath)
	if err != nil {
		rep.Issues = append(rep.Issues, fmt.Sprintf("%s: %v", mcpPath, err))
	} else if hasMCP {
		rep.OK = append(rep.OK, mcpPath+": chiton server registered")
	} else {
		rep.Issues = append(rep.Issues, mcpPath+": no chiton entry (run `chiton install --assistant copilot-vscode`)")
	}

	rulesPath := filepath.Join(opts.Dir, ".github", "copilot-instructions.md")
	hasRules, err := RulesFileHasChiton(rulesPath)
	if err != nil {
		rep.Issues = append(rep.Issues, fmt.Sprintf("%s: %v", rulesPath, err))
	} else if hasRules {
		rep.OK = append(rep.OK, rulesPath+": chiton rules block present")
	} else {
		rep.Issues = append(rep.Issues, rulesPath+": no chiton rules block")
	}

	rep.Registered = hasMCP && hasRules
	return rep, nil
}

func init() { Register(CopilotVSCode{}) }
