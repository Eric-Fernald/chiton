package install

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// OpenCode uses a JSON (or JSONC) config whose MCP servers live under a
// top-level "mcp" object (singular — NOT "mcpServers"). A local stdio server
// is { "type": "local", "command": [<exe>, <args...>], "enabled": true }:
// the executable and its arguments share ONE flat "command" array; there is
// no separate "args" field. These helpers merge into that shape, preserving
// other servers and top-level keys (model, theme, instructions, ...).

const openCodeSchema = "https://opencode.ai/config.json"

// openCodeEntry is the chiton local-server entry in OpenCode's "mcp" object.
type openCodeEntry struct {
	Type    string   `json:"type"`
	Command []string `json:"command"`
	Enabled bool     `json:"enabled"`
}

// WriteOpenCodeConfig upserts the chiton MCP entry into an OpenCode config at
// path. exe is the chiton binary; graphPath may be empty.
//
// Returns whether the file was created (true) or updated in place (false),
// plus whether a chiton entry was already present.
func WriteOpenCodeConfig(path, exe, graphPath string) (created, preexisted bool, err error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return false, false, fmt.Errorf("mkdir %s: %w", filepath.Dir(path), err)
	}

	existing, readErr := os.ReadFile(path)
	if readErr != nil && !os.IsNotExist(readErr) {
		return false, false, fmt.Errorf("read %s: %w", path, readErr)
	}

	var doc map[string]any
	if len(existing) > 0 {
		if err := json.Unmarshal(jsoncToJSON(existing), &doc); err != nil {
			return false, false, fmt.Errorf("parse %s: %w", path, err)
		}
	}
	if doc == nil {
		doc = map[string]any{}
	}
	if _, ok := doc["$schema"]; !ok {
		doc["$schema"] = openCodeSchema
	}

	servers, _ := doc["mcp"].(map[string]any)
	if servers == nil {
		servers = map[string]any{}
	}
	_, preexisted = servers["chiton"]
	servers["chiton"] = openCodeEntry{
		Type:    "local",
		Command: append([]string{exe}, ChitonServeArgs(graphPath)...),
		Enabled: true,
	}
	doc["mcp"] = servers

	out, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return false, false, fmt.Errorf("marshal %s: %w", path, err)
	}
	created = len(existing) == 0
	if err := os.WriteFile(path, append(out, '\n'), 0o644); err != nil {
		return false, false, fmt.Errorf("write %s: %w", path, err)
	}
	return created, preexisted, nil
}

// RemoveOpenCodeEntry removes the chiton entry from an OpenCode config. If the
// file is left with nothing but the "$schema" key (or empty), it is deleted.
//
// Returns whether anything was changed.
func RemoveOpenCodeEntry(path string) (changed bool, err error) {
	existing, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("read %s: %w", path, err)
	}
	var doc map[string]any
	if err := json.Unmarshal(jsoncToJSON(existing), &doc); err != nil {
		return false, fmt.Errorf("parse %s: %w", path, err)
	}
	servers, _ := doc["mcp"].(map[string]any)
	if servers == nil {
		return false, nil
	}
	if _, ok := servers["chiton"]; !ok {
		return false, nil
	}
	delete(servers, "chiton")
	if len(servers) == 0 {
		delete(doc, "mcp")
	} else {
		doc["mcp"] = servers
	}

	// A file holding only the schema marker we added is an empty shell.
	if len(doc) == 0 || (len(doc) == 1 && doc["$schema"] == openCodeSchema) {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return true, fmt.Errorf("remove empty %s: %w", path, err)
		}
		return true, nil
	}
	out, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return true, fmt.Errorf("marshal %s: %w", path, err)
	}
	if err := os.WriteFile(path, append(out, '\n'), 0o644); err != nil {
		return true, fmt.Errorf("write %s: %w", path, err)
	}
	return true, nil
}

// OpenCodeConfigHasChiton reports whether path has a chiton entry under "mcp".
// Missing/unreadable file → false, nil.
func OpenCodeConfigHasChiton(path string) (bool, error) {
	existing, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	var doc map[string]any
	if err := json.Unmarshal(jsoncToJSON(existing), &doc); err != nil {
		return false, err
	}
	servers, _ := doc["mcp"].(map[string]any)
	if servers == nil {
		return false, nil
	}
	_, ok := servers["chiton"]
	return ok, nil
}
