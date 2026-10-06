// Package git turns git history into chiton sync change-sets by shelling out to git (assumed present
// wherever hooks run) and parsing --name-status output.
package git

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"strconv"
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
	if err := ValidateRef(from); err != nil {
		return nil, err
	}
	if err := ValidateRef(to); err != nil {
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
	if err := ValidateRef(ref); err != nil {
		return nil, err
	}
	out, err := run(root, "diff", "--name-status", "--end-of-options", ref, "HEAD")
	if err != nil {
		return nil, err
	}
	return parseNameStatus(root, out), nil
}

// ValidateRef rejects option-like refs, since one such as "--output=evil" would turn `git diff` into an
// arbitrary file write. Valid ref names never start with '-', and callers also pass --end-of-options.
func ValidateRef(ref string) error {
	if ref == "" {
		return fmt.Errorf("git: empty ref")
	}
	if strings.HasPrefix(ref, "-") {
		return fmt.Errorf("git: refusing option-like ref %q", ref)
	}
	return nil
}

func run(root string, args ...string) (string, error) {
	out, err := runBytes(context.Background(), root, nil, args...)
	return string(out), err
}

// runBytes runs git in root with stdin (nil for none) and returns stdout; ctx ending kills git.
func runBytes(ctx context.Context, root string, stdin io.Reader, args ...string) ([]byte, error) {
	// core.quotepath=false stops git octal-escaping non-ASCII paths (e.g. "caf\303\251.go"), which
	// would otherwise skip or mis-delete UTF-8-named files.
	full := append([]string{"-C", root, "-c", "core.quotepath=false"}, args...)
	cmd := exec.CommandContext(ctx, "git", full...)
	cmd.Stdin = stdin
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if cerr := ctx.Err(); cerr != nil {
			return nil, cerr
		}
		return nil, fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}

// StagedFile is one path's content as staged in the index.
type StagedFile struct {
	Path    string // absolute
	Content []byte
}

// StagedFiles returns the index content of each added, copied, modified, or renamed path keep accepts,
// read through one `git cat-file --batch`, so a pre-commit gate checks exactly what the commit records.
func StagedFiles(ctx context.Context, root string, keep func(path string) bool) ([]StagedFile, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	names, err := runBytes(ctx, root, nil, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
	if err != nil {
		return nil, err
	}
	var files []StagedFile
	var req bytes.Buffer
	for rel := range strings.SplitSeq(string(names), "\x00") {
		// cat-file --batch reads one object name per line, so a path holding a newline can't be requested.
		if rel == "" || strings.ContainsRune(rel, '\n') {
			continue
		}
		if p := abs(absRoot, rel); keep(p) {
			files = append(files, StagedFile{Path: p})
			req.WriteString(":" + rel + "\n")
		}
	}
	if len(files) == 0 {
		return nil, nil
	}
	blobs, err := runBytes(ctx, root, &req, "cat-file", "--batch")
	if err != nil {
		return nil, err
	}
	r := bufio.NewReader(bytes.NewReader(blobs))
	for i := range files {
		if files[i].Content, err = readBatchObject(r); err != nil {
			return nil, fmt.Errorf("git cat-file: %w", err)
		}
	}
	return files, nil
}

// readBatchObject reads one `git cat-file --batch` record, "<oid> <type> <size>\n<content>\n". A missing
// object ("<name> missing\n") yields nil content.
func readBatchObject(r *bufio.Reader) ([]byte, error) {
	header, err := r.ReadString('\n')
	if err != nil {
		return nil, err
	}
	if strings.HasSuffix(header, " missing\n") {
		return nil, nil
	}
	fields := strings.Fields(header)
	if len(fields) != 3 {
		return nil, fmt.Errorf("unexpected header %q", header)
	}
	size, err := strconv.Atoi(fields[2])
	if err != nil {
		return nil, fmt.Errorf("unexpected header %q", header)
	}
	buf := make([]byte, size+1) // the content plus its trailing LF
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	return buf[:size], nil
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
