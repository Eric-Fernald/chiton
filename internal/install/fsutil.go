package install

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/fsutil"
)

// readDoc reads and parses the config map at path (doc is nil for a JSON or YAML "null"). For a missing
// file it returns found=false and no error.
func readDoc(path string, unmarshal func([]byte, any) error) (doc map[string]any, found bool, err error) {
	existing, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("read %s: %w", path, err)
	}
	doc = map[string]any{}
	if err := unmarshal(existing, &doc); err != nil {
		return nil, true, fmt.Errorf("parse %s: %w", path, err)
	}
	return doc, true, nil
}

// openDoc creates path's parent directory and prepares the config map at path for an upsert. A missing,
// empty, or null file yields an empty map, and created reports whether the file was missing or empty.
func openDoc(path string, unmarshal func([]byte, any) error) (doc map[string]any, created bool, err error) {
	if err := mkdirParent(path); err != nil {
		return nil, false, err
	}
	existing, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return nil, false, fmt.Errorf("read %s: %w", path, err)
	}
	doc = map[string]any{}
	if len(existing) > 0 {
		if err := unmarshal(existing, &doc); err != nil {
			return nil, false, fmt.Errorf("parse %s: %w", path, err)
		}
	}
	if doc == nil {
		doc = map[string]any{}
	}
	return doc, len(existing) == 0, nil
}

// mcpConfigPerm is the mode a new MCP config gets. These files often hold other servers' API keys in env
// blocks, so they start owner-only.
const mcpConfigPerm os.FileMode = 0o600

// saveDoc marshals doc and atomically replaces path's contents with it (see writeKeepingMode).
func saveDoc(path string, doc map[string]any, marshal func(any) ([]byte, error), newPerm os.FileMode) error {
	out, err := marshal(doc)
	if err != nil {
		return fmt.Errorf("marshal %s: %w", path, err)
	}
	if err := writeKeepingMode(path, out, newPerm); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

// saveOrRemove saves doc to path, or deletes path instead when empty. A symlink is saved through rather than
// deleted, so the file it names never keeps a stale chiton entry.
func saveOrRemove(path string, doc map[string]any, empty bool, marshal func(any) ([]byte, error), newPerm os.FileMode) error {
	if !empty || isSymlink(path) {
		return saveDoc(path, doc, marshal, newPerm)
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove empty %s: %w", path, err)
	}
	return nil
}

// writeKeepingMode atomically replaces the file path names, following a symlink so the link survives. An
// existing file keeps its exact permissions; a new one gets newPerm minus the umask.
func writeKeepingMode(path string, data []byte, newPerm os.FileMode) error {
	target, err := resolveTarget(path)
	if err != nil {
		return err
	}
	fi, err := os.Stat(target)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	perm := newPerm
	if fi != nil {
		perm = fi.Mode().Perm()
	}
	if err := fsutil.WriteFileAtomic(target, data, perm); err != nil {
		return err
	}
	if fi == nil {
		return nil
	}
	return os.Chmod(target, perm) // WriteFileAtomic applied the umask; restore the file's own mode
}

// resolveTarget returns the file path names after following symlinks. A missing path resolves to itself; a
// dangling symlink is an error rather than a silent replacement.
func resolveTarget(path string) (string, error) {
	if !isSymlink(path) {
		return path, nil
	}
	target, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", fmt.Errorf("resolve symlink %s: %w", path, err)
	}
	return target, nil
}

// isSymlink reports whether path itself is a symbolic link.
func isSymlink(path string) bool {
	fi, err := os.Lstat(path)
	return err == nil && fi.Mode()&os.ModeSymlink != 0
}

// mkdirParent creates path's parent directory.
func mkdirParent(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", filepath.Dir(path), err)
	}
	return nil
}

// writeFile creates path's parent directory and writes data to path.
func writeFile(path string, data []byte) error {
	if err := mkdirParent(path); err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

// anyExists reports whether any of rels exists under dir.
func anyExists(dir string, rels ...string) bool {
	for _, rel := range rels {
		if _, err := os.Stat(filepath.Join(dir, rel)); err == nil {
			return true
		}
	}
	return false
}

// fileExists is a Doctor probe. It treats a failed stat as "not installed" and never returns an error.
func fileExists(path string) (bool, error) {
	_, err := os.Stat(path)
	return err == nil, nil
}
