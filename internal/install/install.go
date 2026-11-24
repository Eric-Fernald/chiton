// Package install registers chiton with AI assistants. Each Installer writes one assistant's MCP config
// and its rules and skill files, keeping platform quirks out of the dispatcher.
package install

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/install/manifests"
)

// Options is the bundle of inputs every installer receives.
type Options struct {
	// Dir is the workspace directory. Per-repo files (.mcp.json, CLAUDE.md, ...) resolve against it.
	Dir string

	// Exe is the absolute path to the chiton binary that the assistant should
	// launch as its MCP server.
	Exe string

	// UserScope targets the user-global config (e.g. ~/.cursor/ instead of .cursor/). Installers
	// without a user-global location return ErrUserScopeUnsupported.
	UserScope bool

	// WithHooks, when true, opts in to the assistant's PreToolUse / equivalent
	// hook integration. Off by default because hooks are platform-fragile.
	WithHooks bool

	// GraphPath is recorded in the MCP config so `chiton serve` uses the same store as `chiton build`.
	// An empty GraphPath falls back to chiton's default.
	GraphPath string
}

// Report is what an installer returns describing what it did.
type Report struct {
	Installer string   `json:"installer"`
	Wrote     []string `json:"wrote,omitempty"`
	Skipped   []string `json:"skipped,omitempty"`
	Notes     []string `json:"notes,omitempty"`
}

// DoctorReport is what an installer returns for a health check.
type DoctorReport struct {
	Installer  string   `json:"installer"`
	Detected   bool     `json:"detected"`   // any assistant evidence in this repo
	Registered bool     `json:"registered"` // chiton already wired up
	OK         []string `json:"ok,omitempty"`
	Issues     []string `json:"issues,omitempty"`
}

// ErrUserScopeUnsupported indicates the installer has no user-global config
// location and the caller passed UserScope=true.
var ErrUserScopeUnsupported = errors.New("installer does not support --user-scope")

// Installer is the abstraction each per-assistant module implements.
type Installer interface {
	// Name is the short slug used by the --assistant flag.
	Name() string

	// Description is a one-line human-readable label for help output.
	Description() string

	// Detect returns true if there is evidence in dir that this assistant is
	// (or has been) used in this workspace. Used by auto-detect.
	Detect(dir string) bool

	// Install writes the assistant's config and rules files. It must be idempotent, so a second run yields
	// the same files without duplicating content.
	Install(opts Options) (Report, error)

	// Uninstall reverses Install. Removes only entries chiton owns, never the
	// user's other tool registrations or rules content.
	Uninstall(opts Options) (Report, error)

	// Doctor inspects current state and reports what's wired up and what's not.
	// Never writes anything.
	Doctor(opts Options) (DoctorReport, error)
}

// registered is the global set of installer implementations. Each per-assistant
// file appends to this in its init().
var registered []Installer

// Register adds an installer to the global list. Panics on duplicate name.
func Register(i Installer) {
	if Lookup(i.Name()) != nil {
		panic(fmt.Sprintf("install: duplicate installer name %q", i.Name()))
	}
	registered = append(registered, i)
}

// Available returns every registered installer, ordered by Name.
func Available() []Installer {
	out := slices.Clone(registered)
	slices.SortFunc(out, func(a, b Installer) int { return strings.Compare(a.Name(), b.Name()) })
	return out
}

// Lookup returns the installer with the given name, or nil if not found.
func Lookup(name string) Installer {
	for _, i := range registered {
		if i.Name() == name {
			return i
		}
	}
	return nil
}

// Resolve turns the --assistant value into installers. It accepts a name, a comma-separated list, "all",
// or "auto" (same as "") to pick every installer whose Detect matches, and errors if none match.
func Resolve(name, dir string) ([]Installer, error) {
	switch name {
	case "", "auto":
		var detected []Installer
		for _, i := range Available() {
			if i.Detect(dir) {
				detected = append(detected, i)
			}
		}
		if len(detected) > 0 {
			return detected, nil
		}
		return nil, fmt.Errorf("no assistant evidence detected in %s — pass --assistant <name> or --assistant all", dir)
	case "all":
		return Available(), nil
	}
	var out []Installer
	for part := range strings.SplitSeq(name, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		i := Lookup(part)
		if i == nil {
			return nil, fmt.Errorf("unknown assistant %q — see 'chiton install --help' for supported names", part)
		}
		out = append(out, i)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no assistants resolved from %q", name)
	}
	return out, nil
}

// ResolveExe returns the chiton executable path to record, which is provided if set and otherwise the
// current executable with symlinks resolved.
func ResolveExe(provided string) (string, error) {
	if provided != "" {
		return provided, nil
	}
	exe, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("resolve executable: %w", err)
	}
	resolved, err := filepath.EvalSymlinks(exe)
	if err != nil {
		// EvalSymlinks may fail on transient FS states; fall back to the raw
		// path rather than aborting.
		return exe, nil
	}
	return resolved, nil
}

// record files path under Wrote when changed, else under Skipped.
func (r *Report) record(path string, changed bool) {
	if changed {
		r.Wrote = append(r.Wrote, path)
	} else {
		r.Skipped = append(r.Skipped, path)
	}
}

// writeServer upserts entry as the chiton server in t's config at path and records it, noting when an
// existing chiton entry was refreshed.
func (r *Report) writeServer(t serverTable, path string, entry any) error {
	created, preexisted, err := t.write(path, entry)
	if err != nil {
		return err
	}
	r.Wrote = append(r.Wrote, path)
	if !created && preexisted {
		r.Notes = append(r.Notes, fmt.Sprintf("%s: refreshed existing chiton entry", path))
	}
	return nil
}

// removeServer removes the chiton server from t's config at path and records whether anything changed.
func (r *Report) removeServer(t serverTable, path string) error {
	changed, err := t.remove(path)
	if err != nil {
		return err
	}
	r.record(path, changed)
	return nil
}

// upsertRules writes the shared chiton rules block into path.
func (r *Report) upsertRules(path string) error {
	if _, _, err := UpsertRulesBlock(path, manifests.RulesPrompt); err != nil {
		return err
	}
	r.Wrote = append(r.Wrote, path)
	return nil
}

// removeRules strips the chiton rules block from path.
func (r *Report) removeRules(path string) error {
	changed, err := RemoveRulesBlock(path)
	if err != nil {
		return err
	}
	r.record(path, changed)
	return nil
}

// removeIfExists deletes path with remove when it exists.
func (r *Report) removeIfExists(path string, remove func(string) error) error {
	if _, err := os.Stat(path); err != nil {
		r.Skipped = append(r.Skipped, path)
		return nil
	}
	if err := remove(path); err != nil {
		return fmt.Errorf("remove %s: %w", path, err)
	}
	r.Wrote = append(r.Wrote, path)
	return nil
}

// check runs probe on path, files the outcome under OK or Issues, and reports whether it passed.
func (d *DoctorReport) check(path string, probe func(string) (bool, error), okMsg, missingMsg string) bool {
	ok, err := probe(path)
	switch {
	case err != nil:
		d.Issues = append(d.Issues, fmt.Sprintf("%s: %v", path, err))
		return false
	case ok:
		d.OK = append(d.OK, path+okMsg)
	default:
		d.Issues = append(d.Issues, path+missingMsg)
	}
	return ok
}

// checkMCP checks that the MCP config at path registers chiton.
func (d *DoctorReport) checkMCP(path string, probe func(string) (bool, error)) bool {
	return d.check(path, probe, ": chiton server registered",
		": no chiton entry (run `chiton install --assistant "+d.Installer+"`)")
}

// checkRules checks that the rules file at path carries the chiton block.
func (d *DoctorReport) checkRules(path string) bool {
	return d.check(path, RulesFileHasChiton, ": chiton rules block present", ": no chiton rules block")
}
