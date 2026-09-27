// Package git turns git history into chiton sync change-sets by shelling out to git (assumed present
// wherever hooks run) and parsing --name-status output.
package git

import (
	"bytes"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
)

// Op is the kind of change a file underwent.
type Op int

const (
	// OpUpsert means the file was added, modified, copied, or is the new side
	// of a rename — its current content should be (re-)extracted.
	OpUpsert Op = iota
	// OpDelete means the file was removed (or is the old side of a rename) and
	// its nodes/edges should be deleted from the graph.
	OpDelete
)

func (o Op) String() string {
	if o == OpDelete {
		return "delete"
	}
	return "upsert"
}

// Change is one file's transition. Path is an absolute filesystem path.
type Change struct {
	Path string
	Op   Op
}

// Available reports whether git is on PATH and root is inside a work tree.
func Available(root string) bool {
	cmd := exec.Command("git", "-C", root, "rev-parse", "--is-inside-work-tree")
	out, err := cmd.Output()
	return err == nil && strings.TrimSpace(string(out)) == "true"
}

// LastCommitChanges returns the files touched by HEAD, diffed against its first parent because a plain
// `git diff-tree HEAD` prints nothing for a merge. A parentless root commit reports all files as added.
func LastCommitChanges(root string) ([]Change, error) {
	if parent, ok := firstParent(root); ok {
		out, err := run(root, "diff", "--name-status", parent, "HEAD")
		if err != nil {
			return nil, err
		}
		return parseNameStatus(root, out), nil
	}
	out, err := run(root, "diff-tree", "--no-commit-id", "--name-status", "-r", "--root", "HEAD")
	if err != nil {
		return nil, err
	}
	return parseNameStatus(root, out), nil
}

// firstParent reports HEAD's first parent ("HEAD^") and whether HEAD has one (the initial commit doesn't).
func firstParent(root string) (string, bool) {
	// rev-parse --verify --quiet exits non-zero (and silent) when HEAD^ does not
	// resolve — i.e. on the root commit.
	if _, err := run(root, "rev-parse", "--verify", "--quiet", "HEAD^"); err != nil {
		return "", false
	}
	return "HEAD^", true
}

// RangeChanges returns the files that differ between two refs (e.g. the old and
// new HEAD a post-checkout hook receives).
func RangeChanges(root, from, to string) ([]Change, error) {
	if err := validateRef(from); err != nil {
		return nil, err
	}
	if err := validateRef(to); err != nil {
		return nil, err
	}
	out, err := run(root, "diff", "--name-status", "--end-of-options", from, to)
	if err != nil {
		return nil, err
	}
	return parseNameStatus(root, out), nil
}

// ChangedSince returns the files that differ between ref and the current HEAD.
func ChangedSince(root, ref string) ([]Change, error) {
	if err := validateRef(ref); err != nil {
		return nil, err
	}
	out, err := run(root, "diff", "--name-status", "--end-of-options", ref, "HEAD")
	if err != nil {
		return nil, err
	}
	return parseNameStatus(root, out), nil
}

// validateRef rejects option-like refs, since one such as "--output=evil" would turn `git diff` into an
// arbitrary file write. Valid ref names never start with '-', and callers also pass --end-of-options.
func validateRef(ref string) error {
	if ref == "" {
		return fmt.Errorf("git: empty ref")
	}
	if strings.HasPrefix(ref, "-") {
		return fmt.Errorf("git: refusing option-like ref %q", ref)
	}
	return nil
}

func run(root string, args ...string) (string, error) {
	// core.quotepath=false stops git octal-escaping non-ASCII paths (e.g. "caf\303\251.go"), which
	// would otherwise skip or mis-delete UTF-8-named files.
	full := append([]string{"-C", root, "-c", "core.quotepath=false"}, args...)
	cmd := exec.Command("git", full...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.String(), nil
}

// parseNameStatus parses `git diff --name-status` output into Changes with absolute paths. A rename
// (R###) deletes the old path and upserts the new one. A copy (C###) upserts only the destination.
func parseNameStatus(root, out string) []Change {
	if absRoot, err := filepath.Abs(root); err == nil {
		root = absRoot
	}
	var changes []Change
	for line := range strings.SplitSeq(out, "\n") {
		line = strings.TrimRight(line, "\r")
		if line == "" {
			continue
		}
		fields := strings.Split(line, "\t")
		if len(fields) < 2 {
			continue
		}
		status := fields[0]
		if status == "" {
			continue
		}
		switch status[0] {
		case 'D':
			changes = append(changes, Change{Path: abs(root, fields[1]), Op: OpDelete})
		case 'R':
			// fields: [R### old new] — delete old, upsert new.
			if len(fields) >= 3 {
				changes = append(changes, Change{Path: abs(root, fields[1]), Op: OpDelete})
				changes = append(changes, Change{Path: abs(root, fields[2]), Op: OpUpsert})
			}
		case 'C':
			// fields: [C### src dest] — upsert the new copy.
			if len(fields) >= 3 {
				changes = append(changes, Change{Path: abs(root, fields[2]), Op: OpUpsert})
			}
		case 'A', 'M', 'T':
			changes = append(changes, Change{Path: abs(root, fields[1]), Op: OpUpsert})
		default:
			// U (unmerged), X, B and any unknown status: treat conservatively as
			// an upsert of the named path so the graph re-syncs to current state.
			changes = append(changes, Change{Path: abs(root, fields[len(fields)-1]), Op: OpUpsert})
		}
	}
	return changes
}

// abs resolves a git repo-relative path (forward slashes) against the already-absolute root.
func abs(root, rel string) string {
	if filepath.IsAbs(rel) {
		return filepath.Clean(rel)
	}
	return filepath.Join(root, filepath.FromSlash(rel))
}
