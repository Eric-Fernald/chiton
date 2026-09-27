// Package cache tracks content_sha256 per source_uri so builds skip unchanged files. It is safe for
// concurrent use because the live-sync engine drives it from a worker pool.
package cache

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"

	"github.com/Eric-Fernald/chiton/internal/fsutil"
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

// Mark records sha256 as uri's current hash. Call it only after a successful extract and upsert, so a
// failed upsert doesn't leave an entry that suppresses re-extraction.
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

// URIs returns a snapshot of every cached source_uri, so the reconciler can find files since removed.
func (c *Cache) URIs() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]string, 0, len(c.entries))
	for uri := range c.entries {
		out = append(out, uri)
	}
	return out
}

// Save writes the cache to disk, creating the parent directory if needed. It writes atomically, so a crash
// mid-write leaves the previous cache intact and concurrent Saves never share a temp file.
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
	return fsutil.WriteFileAtomic(c.path, append(data, '\n'), 0o644)
}
