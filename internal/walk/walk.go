// Package walk is the repository walk every indexer shares: one ignore policy (built-in dependency, build, and
// output directories plus the root's .chitonignore) and a file walk that honours it and never follows links.
package walk

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
	"syscall"
)

// IgnoreFile is read from the walk root: one gitignore-style name or glob per line, "#" comments, trailing "/"
// for directories, a leading or inner "/" anchors to the root, "!name" re-includes a DefaultDirs entry.
const IgnoreFile = ".chitonignore"

// DefaultDirs are directory base names skipped at any depth: VCS metadata, chiton's own output (so store and
// cache writes never feed the watcher), and dependency, virtualenv, and build-output trees.
var DefaultDirs = []string{
	".git", ".hg", ".svn",
	".chiton", "chiton-out",
	"node_modules", "vendor",
	".venv", "venv", "__pycache__",
	"target", "dist",
}

// Policy decides which paths under one root a walk skips. Build it with LoadPolicy.
type Policy struct {
	root     string
	dirNames map[string]bool
	patterns []pattern
}

type pattern struct {
	glob     string // path.Match pattern over slash-separated paths
	anchored bool   // match the root-relative path instead of each base name
	dirOnly  bool
}

// LoadPolicy returns the policy for root: DefaultDirs plus the patterns in root's IgnoreFile, if present.
func LoadPolicy(root string) (*Policy, error) {
	p := &Policy{root: filepath.Clean(root), dirNames: make(map[string]bool, len(DefaultDirs))}
	for _, d := range DefaultDirs {
		p.dirNames[d] = true
	}
	data, err := os.ReadFile(filepath.Join(root, IgnoreFile))
	if errors.Is(err, fs.ErrNotExist) || errors.Is(err, syscall.ENOTDIR) {
		return p, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", IgnoreFile, err)
	}
	p.parse(string(data))
	return p, nil
}

// parse adds IgnoreFile lines to p, dropping blank, comment, and malformed ones.
func (p *Policy) parse(data string) {
	for line := range strings.Lines(data) {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if name, ok := strings.CutPrefix(line, "!"); ok {
			delete(p.dirNames, strings.Trim(name, "/"))
			continue
		}
		pt := pattern{dirOnly: strings.HasSuffix(line, "/")}
		glob := strings.TrimSuffix(line, "/")
		pt.anchored = strings.Contains(glob, "/")
		pt.glob = strings.TrimPrefix(glob, "/")
		if _, err := path.Match(pt.glob, ""); pt.glob == "" || err != nil {
			continue
		}
		p.patterns = append(p.patterns, pt)
	}
}

// Skip reports whether target, or a directory between the root and it, is skipped. isDir says whether target
// is itself a directory; the root and paths outside it are never skipped.
func (p *Policy) Skip(target string, isDir bool) bool {
	rel, err := filepath.Rel(p.root, target)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return false
	}
	parts := strings.Split(filepath.ToSlash(rel), "/")
	for i, name := range parts {
		dir := isDir || i < len(parts)-1
		if (dir && p.dirNames[name]) || p.matches(strings.Join(parts[:i+1], "/"), name, dir) {
			return true
		}
	}
	return false
}

// matches reports whether an IgnoreFile pattern skips the entry at root-relative rel with base name name.
func (p *Policy) matches(rel, name string, isDir bool) bool {
	for _, pt := range p.patterns {
		if pt.dirOnly && !isDir {
			continue
		}
		subject := name
		if pt.anchored {
			subject = rel
		}
		if ok, _ := path.Match(pt.glob, subject); ok {
			return true
		}
	}
	return false
}

// Stats counts what a Files walk skipped.
type Stats struct {
	Unreadable int // entries below the root whose stat or directory read failed
}

// Files calls fn with each regular file under root that p keeps, in lexical order, skipping symlinks, junctions,
// and devices. Unreadable entries go to onErr and are skipped; a bad root, cancelled ctx, or fn error stops it.
func Files(ctx context.Context, root string, p *Policy, fn func(path string) error, onErr func(path string, err error)) (Stats, error) {
	var st Stats
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if cerr := ctx.Err(); cerr != nil {
			return cerr
		}
		if err != nil {
			if path == root {
				return err
			}
			st.Unreadable++
			if onErr != nil {
				onErr(path, err)
			}
			return nil
		}
		if d.IsDir() {
			if path != root && p.Skip(path, true) {
				return filepath.SkipDir
			}
			return nil
		}
		// A symlink or junction may point outside the repo; its in-repo target, if any, is walked directly.
		if !d.Type().IsRegular() || p.Skip(path, false) {
			return nil
		}
		return fn(path)
	})
	return st, err
}
