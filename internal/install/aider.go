package install

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/install/manifests"
)

// Aider installs chiton's guidance for the Aider pair-programmer.
//
// Aider has no native MCP support (the MCP PRs remain unmerged as of 2026), so
// this installer cannot register chiton's MCP server. Instead it wires up the
// one integration Aider does offer — a conventions file — and reports that the
// MCP tools are unavailable, so the assistant uses the `chiton` CLI fallback
// documented in the rules block.
//
//	CONVENTIONS.md                   ← always-on rules block (repo root)
//	.aider.conf.yml                  ← adds CONVENTIONS.md to its `read:` list
//
// CONVENTIONS.md is otherwise NOT auto-loaded; the `read:` entry is what makes
// Aider pick it up on every run.
type Aider struct{}

func (Aider) Name() string        { return "aider" }
func (Aider) Description() string { return "Aider pair-programmer (rules only; no MCP support)" }

func (Aider) Detect(dir string) bool {
	for _, rel := range []string{
		".aider.conf.yml",
		".aider.conf.yaml",
		"CONVENTIONS.md",
		".aiderignore",
	} {
		if _, err := os.Stat(filepath.Join(dir, rel)); err == nil {
			return true
		}
	}
	return false
}

const aiderConventionsFile = "CONVENTIONS.md"

func (a Aider) Install(opts Options) (Report, error) {
	if opts.UserScope {
		return Report{Installer: a.Name()}, ErrUserScopeUnsupported
	}
	rep := Report{Installer: a.Name()}

	conventionsPath := filepath.Join(opts.Dir, aiderConventionsFile)
	if _, _, err := UpsertRulesBlock(conventionsPath, manifests.RulesPrompt); err != nil {
		return rep, err
	}
	rep.Wrote = append(rep.Wrote, conventionsPath)

	confPath := filepath.Join(opts.Dir, ".aider.conf.yml")
	if _, _, err := AddAiderRead(confPath, aiderConventionsFile); err != nil {
		return rep, err
	}
	rep.Wrote = append(rep.Wrote, confPath)

	rep.Notes = append(rep.Notes,
		"Aider has no MCP support; chiton's MCP tools are unavailable here.",
		"The rules block tells Aider to shell out to the `chiton` CLI instead — run `chiton build .` first.")
	return rep, nil
}

func (a Aider) Uninstall(opts Options) (Report, error) {
	rep := Report{Installer: a.Name()}

	confPath := filepath.Join(opts.Dir, ".aider.conf.yml")
	changed, err := RemoveAiderRead(confPath, aiderConventionsFile)
	if err != nil {
		return rep, err
	}
	if changed {
		rep.Wrote = append(rep.Wrote, confPath)
	} else {
		rep.Skipped = append(rep.Skipped, confPath)
	}

	conventionsPath := filepath.Join(opts.Dir, aiderConventionsFile)
	changed, err = RemoveRulesBlock(conventionsPath)
	if err != nil {
		return rep, err
	}
	if changed {
		rep.Wrote = append(rep.Wrote, conventionsPath)
	} else {
		rep.Skipped = append(rep.Skipped, conventionsPath)
	}
	return rep, nil
}

func (a Aider) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: a.Name(), Detected: a.Detect(opts.Dir)}

	conventionsPath := filepath.Join(opts.Dir, aiderConventionsFile)
	hasRules, err := RulesFileHasChiton(conventionsPath)
	if err != nil {
		rep.Issues = append(rep.Issues, fmt.Sprintf("%s: %v", conventionsPath, err))
	} else if hasRules {
		rep.OK = append(rep.OK, conventionsPath+": chiton rules block present")
	} else {
		rep.Issues = append(rep.Issues, conventionsPath+": no chiton rules block")
	}

	confPath := filepath.Join(opts.Dir, ".aider.conf.yml")
	hasRead, err := AiderConfHasRead(confPath, aiderConventionsFile)
	if err != nil {
		rep.Issues = append(rep.Issues, fmt.Sprintf("%s: %v", confPath, err))
	} else if hasRead {
		rep.OK = append(rep.OK, confPath+": reads CONVENTIONS.md")
	} else {
		rep.Issues = append(rep.Issues, confPath+": CONVENTIONS.md not in `read:` (run `chiton install --assistant aider`)")
	}

	// No MCP row: Aider cannot host chiton's MCP server. "Registered" here means
	// the conventions wiring is in place.
	rep.Registered = hasRules && hasRead
	return rep, nil
}

func init() { Register(Aider{}) }
