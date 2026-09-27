package install

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/fsutil"
)

// readDoc reads and parses the config map at path (doc is nil for a JSON/YAML "null"). A missing file is
// found=false with no error.
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

// openDoc prepares the config map at path for an upsert: a missing, empty, or null file is an empty map,
// which created reports. It creates path's parent directory.
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

// saveDoc marshals doc and atomically replaces path with it.
func saveDoc(path string, doc map[string]any, marshal func(any) ([]byte, error)) error {
	out, err := marshal(doc)
	if err != nil {
		return fmt.Errorf("marshal %s: %w", path, err)
	}
	if err := fsutil.WriteFileAtomic(path, out, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

// saveOrRemove saves doc to path, or deletes path instead when empty.
func saveOrRemove(path string, doc map[string]any, empty bool, marshal func(any) ([]byte, error)) error {
	if !empty {
		return saveDoc(path, doc, marshal)
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove empty %s: %w", path, err)
	}
	return nil
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

// fileExists is a Doctor probe: a failed stat just means "not installed", never an error.
func fileExists(path string) (bool, error) {
	_, err := os.Stat(path)
	return err == nil, nil
}
