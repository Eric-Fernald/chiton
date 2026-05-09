// Package cache tracks content_sha256 per source_uri so chiton build can skip
// re-extraction of files whose content has not changed since the last run.
package cache

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// Cache maps source_uri to the SHA256 hex string last seen for that file.
type Cache struct {
	path    string
	entries map[string]string
}

// Load opens the cache file at path, creating an empty cache if the file does
// not yet exist. Returns an error only on genuine I/O or parse failures.
func Load(path string) (*Cache, error) {
	c := &Cache{path: path, entries: map[string]string{}}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return c, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(data, &c.entries); err != nil {
		return nil, err
	}
	return c, nil
}

// Changed reports whether uri is new or its sha256 differs from the cached value.
func (c *Cache) Changed(uri, sha256 string) bool {
	return c.entries[uri] != sha256
}

// Mark records sha256 as the current hash for uri.
// Call this only after a successful extract + upsert so a failed upsert
// doesn't leave a stale cache entry that would suppress re-extraction.
func (c *Cache) Mark(uri, sha256 string) {
	c.entries[uri] = sha256
}

// Save writes the cache to disk. It creates the parent directory if needed
// and writes atomically via a temp file so a crash mid-write leaves the
// previous cache intact.
func (c *Cache) Save() error {
	data, err := json.MarshalIndent(c.entries, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(c.path), 0o755); err != nil {
		return err
	}
	tmp := c.path + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, c.path)
}
