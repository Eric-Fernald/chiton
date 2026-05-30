// Package cache tracks content_sha256 per source_uri so chiton build can skip
// re-extraction of files whose content has not changed since the last run.
//
// The cache is safe for concurrent use: the live-sync engine (Phase 3) drives
// it from a worker pool, so Changed/Mark/Delete/Save are all guarded by a mutex.
package cache

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
)

// Cache maps source_uri to the SHA256 hex string last seen for that file.
type Cache struct {
	path string

	mu      sync.RWMutex
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
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.entries[uri] != sha256
}

// Mark records sha256 as the current hash for uri.
// Call this only after a successful extract + upsert so a failed upsert
// doesn't leave a stale cache entry that would suppress re-extraction.
func (c *Cache) Mark(uri, sha256 string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries[uri] = sha256
}

// Delete forgets the cached hash for uri. Used when a file is removed so a
// later re-creation with identical content is still re-extracted.
func (c *Cache) Delete(uri string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.entries, uri)
}

// URIs returns a snapshot of every source_uri currently cached. Used by the
// live-sync one-shot reconciler to find files that were indexed previously but
// are now gone from the tree.
func (c *Cache) URIs() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]string, 0, len(c.entries))
	for uri := range c.entries {
		out = append(out, uri)
	}
	return out
}

// Save writes the cache to disk. It creates the parent directory if needed
// and writes atomically via a temp file so a crash mid-write leaves the
// previous cache intact.
func (c *Cache) Save() error {
	c.mu.RLock()
	data, err := json.MarshalIndent(c.entries, "", "  ")
	c.mu.RUnlock()
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
