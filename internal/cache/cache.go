// Package cache tracks content_sha256 per source_uri so builds skip unchanged files. It is safe for
// concurrent use because the live-sync engine drives it from a worker pool.
package cache

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/Eric-Fernald/chiton/internal/fsutil"
)

// Cache maps source_uri to the SHA256 hex string last seen for that file, for one store epoch.
type Cache struct {
	path      string
	epoch     string
	discarded bool
	corrupt   error

	mu      sync.RWMutex
	entries map[string]string
}

// fileFormat is cache.json's layout. Entries are only valid for the store whose epoch recorded them.
type fileFormat struct {
	StoreEpoch string            `json:"store_epoch"`
	Entries    map[string]string `json:"entries"`
}

// Load opens the cache at path for the store identified by storeEpoch. It starts empty, so every file is
// re-extracted, when the file is missing, unparsable, or was written for another store (or before epochs).
func Load(path, storeEpoch string) (*Cache, error) {
	c := &Cache{path: path, epoch: storeEpoch, entries: map[string]string{}}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return c, nil
	}
	if err != nil {
		return nil, err
	}
	var f fileFormat
	if err := json.Unmarshal(data, &f); err != nil {
		// The cache only saves work, so a truncated or hand-edited file costs a re-extract, not the run.
		c.corrupt = fmt.Errorf("parse %s: %w", path, err)
		return c, nil
	}
	if f.StoreEpoch != storeEpoch {
		c.discarded = true
		return c, nil
	}
	if f.Entries != nil {
		c.entries = f.Entries
	}
	return c, nil
}

// Discarded reports whether Load dropped entries recorded for a different store.
func (c *Cache) Discarded() bool { return c.discarded }

// Corrupt returns why Load could not parse the cache file and so started empty, or nil.
func (c *Cache) Corrupt() error { return c.corrupt }

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

// Save writes the cache and its store epoch to disk, creating the parent directory if needed. It writes
// atomically, so a crash mid-write leaves the previous cache intact and concurrent Saves never share a temp file.
func (c *Cache) Save() error {
	c.mu.RLock()
	data, err := json.MarshalIndent(fileFormat{StoreEpoch: c.epoch, Entries: c.entries}, "", "  ")
	c.mu.RUnlock()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(c.path), 0o755); err != nil {
		return err
	}
	return fsutil.WriteFileAtomic(c.path, append(data, '\n'), 0o644)
}
