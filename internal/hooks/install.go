package hooks

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// HookMarker is the comment chiton writes into every hook script so we can
// recognise (and safely replace/remove) hooks chiton owns without clobbering
// hooks the user wrote.
const HookMarker = "# managed by chiton — do not edit"

// InstallReport summarises a hooks install/uninstall pass.
type InstallReport struct {
	Wrote   []string `json:"wrote,omitempty"`
	Skipped []string `json:"skipped,omitempty"`
	Notes   []string `json:"notes,omitempty"`
}

// Install writes chiton-owned git hooks for each stage into hooksDir.
//
// If a stage's hook file already exists and is not chiton-owned, it is left
// alone and reported in Notes — chiton never overwrites third-party hooks.
// Existing chiton-owned hooks are refreshed in place.
func Install(hooksDir, exe string, stages []Stage) (InstallReport, error) {
	rep := InstallReport{}
	if err := os.MkdirAll(hooksDir, 0o755); err != nil {
		return rep, fmt.Errorf("mkdir %s: %w", hooksDir, err)
	}
	for _, stage := range stages {
		path := filepath.Join(hooksDir, string(stage))
		owned, err := isChitonOwned(path)
		if err != nil {
			return rep, err
		}
		if existed, _ := exists(path); existed && !owned {
			rep.Notes = append(rep.Notes, fmt.Sprintf("%s: pre-existing non-chiton hook left intact (rename or remove it, then re-run)", path))
			rep.Skipped = append(rep.Skipped, path)
			continue
		}
		script := scriptFor(stage, exe)
		if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
			return rep, fmt.Errorf("write %s: %w", path, err)
		}
		// Best-effort chmod for non-Windows; ignored on Windows where 0o755
		// is meaningless to NTFS but Git for Windows runs the script anyway.
		if runtime.GOOS != "windows" {
			_ = os.Chmod(path, 0o755)
		}
		rep.Wrote = append(rep.Wrote, path)
	}
	return rep, nil
}

// Uninstall removes chiton-owned hook scripts. Hooks that are not chiton-owned
// are left untouched.
func Uninstall(hooksDir string, stages []Stage) (InstallReport, error) {
	rep := InstallReport{}
	for _, stage := range stages {
		path := filepath.Join(hooksDir, string(stage))
		existed, _ := exists(path)
		if !existed {
			rep.Skipped = append(rep.Skipped, path)
			continue
		}
		owned, err := isChitonOwned(path)
		if err != nil {
			return rep, err
		}
		if !owned {
			rep.Notes = append(rep.Notes, fmt.Sprintf("%s: not chiton-owned, left in place", path))
			rep.Skipped = append(rep.Skipped, path)
			continue
		}
		if err := os.Remove(path); err != nil {
			return rep, fmt.Errorf("remove %s: %w", path, err)
		}
		rep.Wrote = append(rep.Wrote, path)
	}
	return rep, nil
}

// Status reports which hooks are present and whether they are chiton-owned.
type StatusEntry struct {
	Stage       Stage  `json:"stage"`
	Path        string `json:"path"`
	Exists      bool   `json:"exists"`
	ChitonOwned bool   `json:"chiton_owned"`
}

// Status returns one StatusEntry per stage in stages.
func Status(hooksDir string, stages []Stage) ([]StatusEntry, error) {
	out := make([]StatusEntry, 0, len(stages))
	for _, stage := range stages {
		path := filepath.Join(hooksDir, string(stage))
		entry := StatusEntry{Stage: stage, Path: path}
		ex, err := exists(path)
		if err != nil {
			return nil, err
		}
		entry.Exists = ex
		if ex {
			owned, err := isChitonOwned(path)
			if err != nil {
				return nil, err
			}
			entry.ChitonOwned = owned
		}
		out = append(out, entry)
	}
	return out, nil
}

// scriptFor returns the shell content for a hook of stage that invokes the
// given chiton executable. Gating stages (pre-commit/pre-push) run the
// finding-gate (`hooks run`); sync stages (post-commit/post-checkout) re-index
// the changed files (`sync git`). Plain POSIX sh so Git for Windows can execute
// it through its bundled bash.
func scriptFor(stage Stage, exe string) string {
	var invocation string
	switch stage {
	case StagePostCommit:
		invocation = fmt.Sprintf("exec %s sync git --stage post-commit", shellQuote(exe))
	case StagePostCheckout:
		// git passes <old> <new> <branch-flag> as positional args; forward them.
		invocation = fmt.Sprintf("exec %s sync git --stage post-checkout \"$@\"", shellQuote(exe))
	default:
		invocation = fmt.Sprintf("exec %s hooks run --stage %s", shellQuote(exe), string(stage))
	}
	return strings.Join([]string{
		"#!/bin/sh",
		HookMarker,
		invocation,
		"",
	}, "\n")
}

func isChitonOwned(path string) (bool, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("read %s: %w", path, err)
	}
	return strings.Contains(string(b), HookMarker), nil
}

func exists(path string) (bool, error) {
	if _, err := os.Stat(path); err == nil {
		return true, nil
	} else if os.IsNotExist(err) {
		return false, nil
	} else {
		return false, err
	}
}

// shellQuote single-quotes s for safe use as a POSIX sh argument. Embedded
// single quotes are encoded as '\''.
func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}
