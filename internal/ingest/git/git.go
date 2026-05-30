// Package git turns git history into chiton sync change-sets. It shells out to
// the git binary (assumed present wherever git hooks run) and parses
// --name-status output into the Change events the live-sync engine consumes.
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

// LastCommitChanges returns the files touched by HEAD (the just-made commit).
// Uses --root so the initial commit reports its files as additions.
func LastCommitChanges(root string) ([]Change, error) {
	out, err := run(root, "diff-tree", "--no-commit-id", "--name-status", "-r", "--root", "HEAD")
	if err != nil {
		return nil, err
	}
	return parseNameStatus(root, out), nil
}

// RangeChanges returns the files that differ between two refs (e.g. the old and
// new HEAD a post-checkout hook receives).
func RangeChanges(root, from, to string) ([]Change, error) {
	out, err := run(root, "diff", "--name-status", from, to)
	if err != nil {
		return nil, err
	}
	return parseNameStatus(root, out), nil
}

// ChangedSince returns the files that differ between ref and the current HEAD.
func ChangedSince(root, ref string) ([]Change, error) {
	out, err := run(root, "diff", "--name-status", ref, "HEAD")
	if err != nil {
		return nil, err
	}
	return parseNameStatus(root, out), nil
}

func run(root string, args ...string) (string, error) {
	// core.quotepath=false stops git from octal-escaping non-ASCII path bytes
	// (e.g. "caf\303\251.go"), which would otherwise reach the filesystem as a
	// bogus path and silently skip / mis-delete UTF-8-named files.
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

// parseNameStatus parses `git diff --name-status` / `diff-tree` output into
// Changes, resolving repo-relative paths against root.
//
// Lines look like:
//
//	M\tpath/to/file.go
//	A\tnew.go
//	D\tgone.go
//	R100\told/path.go\tnew/path.go     (rename → delete old, upsert new)
//	C75\tsrc.go\tcopy.go               (copy → upsert new only)
func parseNameStatus(root, out string) []Change {
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

// abs resolves a git repo-relative path (forward slashes) to an absolute
// filesystem path. root is resolved to absolute first so Change.Path is
// absolute regardless of the process working directory (honoring the
// Change.Path contract even when --root is given as ".").
func abs(root, rel string) string {
	if filepath.IsAbs(rel) {
		return filepath.Clean(rel)
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		absRoot = root
	}
	return filepath.Join(absRoot, filepath.FromSlash(rel))
}
