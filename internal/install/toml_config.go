package install

import (
	"fmt"
	"os"
	"path/filepath"

	toml "github.com/pelletier/go-toml/v2"
)

// Codex CLI stores all of its settings — model, approval policy, and MCP
// servers — in a single TOML file (~/.codex/config.toml). MCP servers live
// under [mcp_servers.<name>] tables (underscore, plural), each with a
// command string and an args array for a stdio launch. These helpers do a
// TOML-aware merge so unrelated settings and other [mcp_servers.*] tables are
// preserved.

// codexServersKey is the top-level TOML table holding per-server subtables.
const codexServersKey = "mcp_servers"

// WriteCodexConfig upserts the chiton [mcp_servers.chiton] table into a Codex
// config.toml at path, preserving every other setting and server table.
//
// Returns whether the file was created (true) or updated in place (false),
// plus whether a chiton entry was already present.
func WriteCodexConfig(path, exe, graphPath string) (created, preexisted bool, err error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return false, false, fmt.Errorf("mkdir %s: %w", filepath.Dir(path), err)
	}

	existing, readErr := os.ReadFile(path)
	if readErr != nil && !os.IsNotExist(readErr) {
		return false, false, fmt.Errorf("read %s: %w", path, readErr)
	}

	doc := map[string]any{}
	if len(existing) > 0 {
		if err := toml.Unmarshal(existing, &doc); err != nil {
			return false, false, fmt.Errorf("parse %s: %w", path, err)
		}
	}

	raw, present := doc[codexServersKey]
	servers, ok := raw.(map[string]any)
	if present && !ok {
		return false, false, fmt.Errorf("refusing to overwrite %s: %q is not a TOML table", path, codexServersKey)
	}
	if servers == nil {
		servers = map[string]any{}
	}
	_, preexisted = servers["chiton"]
	servers["chiton"] = map[string]any{
		"command": exe,
		"args":    ChitonServeArgs(graphPath),
	}
	doc[codexServersKey] = servers

	out, err := toml.Marshal(doc)
	if err != nil {
		return false, false, fmt.Errorf("marshal %s: %w", path, err)
	}
	created = len(existing) == 0
	if err := writeFileAtomic(path, out, 0o644); err != nil {
		return false, false, fmt.Errorf("write %s: %w", path, err)
	}
	return created, preexisted, nil
}

// RemoveCodexEntry removes the chiton table from a Codex config.toml. If the
// file is left with no settings at all it is deleted.
//
// Returns whether anything was changed.
func RemoveCodexEntry(path string) (changed bool, err error) {
	existing, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("read %s: %w", path, err)
	}
	doc := map[string]any{}
	if err := toml.Unmarshal(existing, &doc); err != nil {
		return false, fmt.Errorf("parse %s: %w", path, err)
	}
	servers, _ := doc[codexServersKey].(map[string]any)
	if servers == nil {
		return false, nil
	}
	if _, ok := servers["chiton"]; !ok {
		return false, nil
	}
	delete(servers, "chiton")
	if len(servers) == 0 {
		delete(doc, codexServersKey)
	} else {
		doc[codexServersKey] = servers
	}

	if len(doc) == 0 {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return true, fmt.Errorf("remove empty %s: %w", path, err)
		}
		return true, nil
	}
	out, err := toml.Marshal(doc)
	if err != nil {
		return true, fmt.Errorf("marshal %s: %w", path, err)
	}
	if err := writeFileAtomic(path, out, 0o644); err != nil {
		return true, fmt.Errorf("write %s: %w", path, err)
	}
	return true, nil
}

// CodexConfigHasChiton reports whether path has a chiton [mcp_servers.chiton]
// table. Missing/unreadable file → false, nil.
func CodexConfigHasChiton(path string) (bool, error) {
	existing, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	doc := map[string]any{}
	if err := toml.Unmarshal(existing, &doc); err != nil {
		return false, err
	}
	servers, _ := doc[codexServersKey].(map[string]any)
	if servers == nil {
		return false, nil
	}
	_, ok := servers["chiton"]
	return ok, nil
}
