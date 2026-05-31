package install

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"gopkg.in/yaml.v3"
)

// Aider has no native MCP support, so chiton integrates with it purely through
// its conventions mechanism: a CONVENTIONS.md rules file that Aider only loads
// when it is named in the `read:` list of .aider.conf.yml (or passed via
// --read). These helpers merge a filename into that `read:` list, which Aider
// accepts as either a scalar string or a sequence.

// AddAiderRead ensures file appears in the `read:` list of the Aider config at
// path, creating the file/key as needed and preserving all other settings.
//
// Returns whether the file was created (true) or updated in place (false),
// plus whether file was already present in `read:`.
func AddAiderRead(path, file string) (created, preexisted bool, err error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return false, false, fmt.Errorf("mkdir %s: %w", filepath.Dir(path), err)
	}
	existing, readErr := os.ReadFile(path)
	if readErr != nil && !os.IsNotExist(readErr) {
		return false, false, fmt.Errorf("read %s: %w", path, readErr)
	}

	doc := map[string]any{}
	if len(existing) > 0 {
		if err := yaml.Unmarshal(existing, &doc); err != nil {
			return false, false, fmt.Errorf("parse %s: %w", path, err)
		}
		if doc == nil {
			doc = map[string]any{}
		}
	}

	reads := normalizeStringList(doc["read"])
	for _, r := range reads {
		if r == file {
			preexisted = true
		}
	}
	if !preexisted {
		reads = append(reads, file)
	}
	doc["read"] = reads

	out, err := yaml.Marshal(doc)
	if err != nil {
		return false, false, fmt.Errorf("marshal %s: %w", path, err)
	}
	created = len(existing) == 0
	if err := os.WriteFile(path, out, 0o644); err != nil {
		return false, false, fmt.Errorf("write %s: %w", path, err)
	}
	return created, preexisted, nil
}

// RemoveAiderRead removes file from the `read:` list of the Aider config at
// path. If `read:` becomes empty the key is dropped; if the file is left with
// no settings it is deleted.
//
// Returns whether anything was changed.
func RemoveAiderRead(path, file string) (changed bool, err error) {
	existing, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("read %s: %w", path, err)
	}
	doc := map[string]any{}
	if err := yaml.Unmarshal(existing, &doc); err != nil {
		return false, fmt.Errorf("parse %s: %w", path, err)
	}
	if doc == nil {
		return false, nil
	}

	reads := normalizeStringList(doc["read"])
	kept := reads[:0]
	for _, r := range reads {
		if r == file {
			changed = true
			continue
		}
		kept = append(kept, r)
	}
	if !changed {
		return false, nil
	}
	if len(kept) == 0 {
		delete(doc, "read")
	} else {
		doc["read"] = kept
	}

	if len(doc) == 0 {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return true, fmt.Errorf("remove empty %s: %w", path, err)
		}
		return true, nil
	}
	out, err := yaml.Marshal(doc)
	if err != nil {
		return true, fmt.Errorf("marshal %s: %w", path, err)
	}
	if err := os.WriteFile(path, out, 0o644); err != nil {
		return true, fmt.Errorf("write %s: %w", path, err)
	}
	return true, nil
}

// AiderConfHasRead reports whether the Aider config at path lists file in its
// `read:` setting. Missing/unreadable file → false, nil.
func AiderConfHasRead(path, file string) (bool, error) {
	existing, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	doc := map[string]any{}
	if err := yaml.Unmarshal(existing, &doc); err != nil {
		return false, err
	}
	return slices.Contains(normalizeStringList(doc["read"]), file), nil
}

// normalizeStringList coerces a YAML value that may be a scalar string or a
// sequence into a []string. Non-string elements are dropped.
func normalizeStringList(v any) []string {
	switch t := v.(type) {
	case nil:
		return nil
	case string:
		if t == "" {
			return nil
		}
		return []string{t}
	case []string:
		return t
	case []any:
		out := make([]string, 0, len(t))
		for _, e := range t {
			if s, ok := e.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}
