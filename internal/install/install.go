// Package install registers chiton with AI assistants.
//
// Each supported assistant has its own Installer implementation, which knows
// how to write that assistant's MCP config and rules/skill files. The package
// keeps platform-specific quirks isolated; the dispatcher just iterates over
// the configured Installers.
package install

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Options is the bundle of inputs every installer receives.
type Options struct {
	// Dir is the workspace directory the installer should write into.
	// Per-repo files (.mcp.json, CLAUDE.md, .vscode/mcp.json, ...) are
	// resolved relative to this directory.
	Dir string

	// Exe is the absolute path to the chiton binary that the assistant should
	// launch as its MCP server.
	Exe string

	// UserScope, when true, asks the installer to write into the user-global
	// config location instead of the per-repo one (e.g. ~/.claude/ instead of
	// .claude/). Not every installer supports user-scope; those that don't
	// return ErrUserScopeUnsupported.
	UserScope bool

	// WithHooks, when true, opts in to the assistant's PreToolUse / equivalent
	// hook integration. Off by default because hooks are platform-fragile.
	WithHooks bool

	// GraphPath is the path the installer should record in the MCP config so
	// `chiton serve` uses the same store as `chiton build`. Optional; when
	// empty the installer falls back to chiton's default.
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

	// Install writes the assistant's config and rules files. Implementations
	// must be idempotent: running twice in a row should yield the same files
	// without duplicating content.
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
	for _, existing := range registered {
		if existing.Name() == i.Name() {
			panic(fmt.Sprintf("install: duplicate installer name %q", i.Name()))
		}
	}
	registered = append(registered, i)
}

// Available returns every registered installer, ordered by Name.
func Available() []Installer {
	out := make([]Installer, len(registered))
	copy(out, registered)
	sort.Slice(out, func(i, j int) bool { return out[i].Name() < out[j].Name() })
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

// Resolve turns the --assistant flag value into a list of installers to act
// on. Supports:
//   - explicit name ("claude-code")
//   - comma-separated list ("claude-code,cursor")
//   - "all" — every registered installer
//   - "auto" or "" — every installer whose Detect() returns true, falling
//     back to every installer if none match
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
	for _, part := range strings.Split(name, ",") {
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

// ResolveExe returns the chiton executable path the installer should record.
// Prefers a value from Options.Exe, otherwise falls back to the current
// executable.
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
