package install

import (
	"errors"
	"slices"

	"gopkg.in/yaml.v3"
)

// Aider loads CONVENTIONS.md only when .aider.conf.yml names it under `read:`, which Aider accepts as
// either a scalar string or a sequence. These helpers merge a filename into that list.

// AddAiderRead adds file to the `read:` list of the Aider config at path, keeping other settings.
// It reports whether the file was created and whether file was already listed.
func AddAiderRead(path, file string) (created, preexisted bool, err error) {
	doc, created, err := openDoc(path, yaml.Unmarshal)
	if err != nil {
		return false, false, err
	}
	reads := normalizeStringList(doc["read"])
	preexisted = slices.Contains(reads, file)
	if !preexisted {
		reads = append(reads, file)
	}
	doc["read"] = reads
	if err := saveDoc(path, doc, yaml.Marshal, 0o644); err != nil {
		return false, false, err
	}
	return created, preexisted, nil
}

// RemoveAiderRead removes file from the `read:` list at path, dropping an emptied key and deleting a
// file left with no settings. It reports whether anything changed.
func RemoveAiderRead(path, file string) (changed bool, err error) {
	doc, found, err := readDoc(path, yaml.Unmarshal)
	if !found || err != nil {
		return false, err
	}
	reads := normalizeStringList(doc["read"]) // a null doc has no reads and returns unchanged below
	n := len(reads)
	reads = slices.DeleteFunc(reads, func(r string) bool { return r == file })
	if len(reads) == n {
		return false, nil
	}
	if len(reads) == 0 {
		delete(doc, "read")
	} else {
		doc["read"] = reads
	}
	return true, saveOrRemove(path, doc, len(doc) == 0, yaml.Marshal, 0o644)
}

// AiderConfHasRead reports whether the Aider config at path lists file in its `read:` setting.
// It returns false, nil for a missing file.
func AiderConfHasRead(path, file string) (bool, error) {
	doc, _, err := readDoc(path, yaml.Unmarshal)
	if err != nil {
		return false, errors.Unwrap(err) // return the bare cause because Doctor prefixes the path itself
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
