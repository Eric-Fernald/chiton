package install

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// ServerEntry is one MCP server's command line, as it appears in any MCP
// config JSON file (.mcp.json, .vscode/mcp.json, etc.). All known assistants
// use this same shape, so one struct serves all installers.
type ServerEntry struct {
	Command string   `json:"command"`
	Args    []string `json:"args"`
}

// ChitonServeArgs returns the argv (after the binary) that launches chiton's
// stdio MCP server. graphPath may be empty (the server then uses its
// configured/default graph_path). Shared by every installer so all assistants
// launch the server identically, regardless of their config file format.
func ChitonServeArgs(graphPath string) []string {
	args := []string{"serve", "--stdio"}
	if graphPath != "" {
		args = append(args, "--graph", graphPath)
	}
	return args
}

// ChitonServerEntry returns the canonical chiton MCP entry for a given exe
// path and graph path. graphPath may be empty.
func ChitonServerEntry(exe, graphPath string) ServerEntry {
	return ServerEntry{Command: exe, Args: ChitonServeArgs(graphPath)}
}

// WriteMCPConfig writes (or merges) the chiton entry into an MCP config file
// at path. If the file already exists, the chiton entry is upserted under
// "mcpServers" and any unrelated keys are preserved.
//
// entry is any JSON-serialisable value: most assistants want a ServerEntry
// (command + args), but some (e.g. Copilot CLI) carry extra fields like
// type/env/tools, so the entry type is left open.
//
// Returns whether the file was created (true) or updated in place (false),
// plus whether a chiton entry was already present before this call.
func WriteMCPConfig(path string, entry any) (created, preexisted bool, err error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return false, false, fmt.Errorf("mkdir %s: %w", filepath.Dir(path), err)
	}

	existing, readErr := os.ReadFile(path)
	if readErr != nil && !os.IsNotExist(readErr) {
		return false, false, fmt.Errorf("read %s: %w", path, readErr)
	}

	var doc map[string]any
	if len(existing) > 0 {
		if err := json.Unmarshal(existing, &doc); err != nil {
			return false, false, fmt.Errorf("parse %s: %w", path, err)
		}
	}
	if doc == nil {
		doc = map[string]any{}
	}

	servers, _ := doc["mcpServers"].(map[string]any)
	if servers == nil {
		servers = map[string]any{}
	}
	_, preexisted = servers["chiton"]
	servers["chiton"] = entry
	doc["mcpServers"] = servers

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

// RemoveMCPConfigEntry removes the chiton entry from an MCP config file at
// path. If the file is left with no entries under "mcpServers" and no other
// keys, the file itself is deleted to avoid leaving an empty shell behind.
//
// Returns whether anything was changed (false if no chiton entry existed).
func RemoveMCPConfigEntry(path string) (changed bool, err error) {
	existing, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("read %s: %w", path, err)
	}

	var doc map[string]any
	if err := json.Unmarshal(existing, &doc); err != nil {
		return false, fmt.Errorf("parse %s: %w", path, err)
	}
	servers, _ := doc["mcpServers"].(map[string]any)
	if servers == nil {
		return false, nil
	}
	if _, ok := servers["chiton"]; !ok {
		return false, nil
	}
	delete(servers, "chiton")
	if len(servers) == 0 {
		delete(doc, "mcpServers")
	} else {
		doc["mcpServers"] = servers
	}

	if len(doc) == 0 {
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

// MCPConfigHasChiton reports whether path exists and already has a chiton
// entry under mcpServers. Missing/unreadable file → false, nil.
func MCPConfigHasChiton(path string) (bool, error) {
	existing, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	var doc map[string]any
	if err := json.Unmarshal(existing, &doc); err != nil {
		return false, err
	}
	servers, _ := doc["mcpServers"].(map[string]any)
	if servers == nil {
		return false, nil
	}
	_, ok := servers["chiton"]
	return ok, nil
}
