package cache

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func saved(t *testing.T, path, epoch string) {
	t.Helper()
	c, err := Load(path, epoch)
	if err != nil {
		t.Fatal(err)
	}
	c.Mark("file:///a.go", "sha-a")
	if err := c.Save(); err != nil {
		t.Fatal(err)
	}
}

func TestLoadKeepsEntriesForSameStore(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.json")
	saved(t, path, "epoch-1")
	c, err := Load(path, "epoch-1")
	if err != nil {
		t.Fatal(err)
	}
	if c.Changed("file:///a.go", "sha-a") || c.Discarded() {
		t.Fatalf("same store: Changed = %v, Discarded = %v; want the entry kept", c.Changed("file:///a.go", "sha-a"), c.Discarded())
	}
}

func TestLoadDiscardsEntriesFromAnotherStore(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.json")
	saved(t, path, "epoch-1")
	c, err := Load(path, "epoch-2")
	if err != nil {
		t.Fatal(err)
	}
	if !c.Changed("file:///a.go", "sha-a") || !c.Discarded() || len(c.URIs()) != 0 {
		t.Fatalf("other store: kept %v (Discarded = %v); want every entry dropped", c.URIs(), c.Discarded())
	}
}

func TestLoadDiscardsPreEpochCache(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.json")
	if err := os.WriteFile(path, []byte(`{"file:///a.go": "sha-a"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := Load(path, "epoch-1")
	if err != nil {
		t.Fatal(err)
	}
	if !c.Changed("file:///a.go", "sha-a") || !c.Discarded() {
		t.Fatal("a pre-epoch cache must be discarded")
	}
}

func TestSaveRecordsStoreEpoch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.json")
	saved(t, path, "epoch-1")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		StoreEpoch string            `json:"store_epoch"`
		Entries    map[string]string `json:"entries"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.StoreEpoch != "epoch-1" || got.Entries["file:///a.go"] != "sha-a" {
		t.Fatalf("saved cache = %+v", got)
	}
}

// TestLoadStartsEmptyOnUnparsableFile checks a truncated, empty, or mis-shaped cache.json costs a re-extract,
// not the run: Load starts empty, says why through Corrupt, and the next Save repairs the file.
func TestLoadStartsEmptyOnUnparsableFile(t *testing.T) {
	for name, content := range map[string]string{
		"empty":     "",
		"truncated": `{"store_epoch": "epoch-1", "entries": {"file:///a.go": "sh`,
		"not JSON":  "<<<<<<< HEAD\n",
		"an array":  `["file:///a.go"]`,
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "cache.json")
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
			c, err := Load(path, "epoch-1")
			if err != nil {
				t.Fatalf("Load = %v; want an empty cache", err)
			}
			if !c.Changed("file:///a.go", "sha-a") || len(c.URIs()) != 0 {
				t.Fatalf("kept %v; want every entry dropped", c.URIs())
			}
			if cerr := c.Corrupt(); cerr == nil || !strings.Contains(cerr.Error(), path) {
				t.Fatalf("Corrupt = %v; want the parse error naming %s", cerr, path)
			}
			c.Mark("file:///a.go", "sha-a")
			if err := c.Save(); err != nil {
				t.Fatal(err)
			}
			if c, err = Load(path, "epoch-1"); err != nil || c.Corrupt() != nil || c.Changed("file:///a.go", "sha-a") {
				t.Fatalf("after Save: Load err = %v; want the repaired cache to load with its entry", err)
			}
		})
	}
}

// TestLoadReturnsReadError checks a cache that cannot be read at all (here a directory) still fails, naming it.
func TestLoadReturnsReadError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.json")
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path, "epoch-1"); err == nil || !strings.Contains(err.Error(), path) {
		t.Fatalf("Load = %v; want a read error naming %s", err, path)
	}
}
