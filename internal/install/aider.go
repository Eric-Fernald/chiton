package install

import (
	"path/filepath"
)

// Aider installs chiton's guidance for Aider, which has no MCP support. It writes a CONVENTIONS.md rules
// block and a `read:` entry in .aider.conf.yml, without which Aider never loads CONVENTIONS.md.
type Aider struct{}

func (Aider) Name() string        { return "aider" }
func (Aider) Description() string { return "Aider pair-programmer (rules only; no MCP support)" }

func (Aider) Detect(dir string) bool {
	return anyExists(dir, ".aider.conf.yml", ".aider.conf.yaml", "CONVENTIONS.md", ".aiderignore")
}

const aiderConventionsFile = "CONVENTIONS.md"

func (a Aider) Install(opts Options) (Report, error) {
	rep := Report{Installer: a.Name()}
	if opts.UserScope {
		return rep, ErrUserScopeUnsupported
	}

	if err := rep.upsertRules(filepath.Join(opts.Dir, aiderConventionsFile)); err != nil {
		return rep, err
	}

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
	rep.record(confPath, changed)

	if err := rep.removeRules(filepath.Join(opts.Dir, aiderConventionsFile)); err != nil {
		return rep, err
	}
	return rep, nil
}

func (a Aider) Doctor(opts Options) (DoctorReport, error) {
	rep := DoctorReport{Installer: a.Name(), Detected: a.Detect(opts.Dir)}

	hasRules := rep.checkRules(filepath.Join(opts.Dir, aiderConventionsFile))
	hasRead := rep.check(filepath.Join(opts.Dir, ".aider.conf.yml"),
		func(p string) (bool, error) { return AiderConfHasRead(p, aiderConventionsFile) },
		": reads CONVENTIONS.md",
		": CONVENTIONS.md not in `read:` (run `chiton install --assistant aider`)")

	// Aider can't host MCP servers, so it has no MCP row and "Registered" means the conventions wiring.
	rep.Registered = hasRules && hasRead
	return rep, nil
}

func init() { Register(Aider{}) }
