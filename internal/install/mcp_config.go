package install

import (
	"encoding/json"
	"errors"
	"fmt"
)

// ServerEntry is the {command, args} MCP server entry used under "mcpServers" in JSON configs
// (.mcp.json, .vscode/mcp.json, ...).
type ServerEntry struct {
	Command string   `json:"command"`
	Args    []string `json:"args"`
}

// ChitonServeArgs returns the argv after the binary that launches chiton's stdio MCP server, shared by
// every installer so all assistants launch it identically. graphPath may be empty.
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

// serverTable is a config format holding MCP servers in a map under one top-level key. Its helpers
// touch only the "chiton" entry and preserve every other server and setting.
type serverTable struct {
	key       string // top-level key holding the server map
	kind      string // what that key must hold, for the refusal error
	unmarshal func([]byte, any) error
	marshal   func(any) ([]byte, error)
	seed      func(doc map[string]any)      // if set, adds defaults on write
	shell     func(doc map[string]any) bool // if set, reports whether a non-empty doc still counts as empty
}

// mcpServersTable is the JSON/JSONC "mcpServers" format shared by most assistants.
var mcpServersTable = serverTable{
	key:       "mcpServers",
	kind:      "JSON object",
	unmarshal: unmarshalJSONC,
	marshal:   marshalJSON,
}

func unmarshalJSONC(b []byte, v any) error { return json.Unmarshal(jsoncToJSON(b), v) }

func marshalJSON(v any) ([]byte, error) {
	out, err := json.MarshalIndent(v, "", "  ")
	return append(out, '\n'), err
}

// write upserts entry as the chiton server at path, creating the file if needed. It refuses to replace
// a server key that is not a map, so a malformed config is never clobbered.
func (t serverTable) write(path string, entry any) (created, preexisted bool, err error) {
	doc, created, err := openDoc(path, t.unmarshal)
	if err != nil {
		return false, false, err
	}
	if t.seed != nil {
		t.seed(doc)
	}

	raw, present := doc[t.key]
	servers, ok := raw.(map[string]any)
	if present && !ok {
		return false, false, fmt.Errorf("refusing to overwrite %s: %q is not a %s", path, t.key, t.kind)
	}
	if servers == nil {
		servers = map[string]any{}
	}
	_, preexisted = servers["chiton"]
	servers["chiton"] = entry
	doc[t.key] = servers

	if err := saveDoc(path, doc, t.marshal); err != nil {
		return false, false, err
	}
	return created, preexisted, nil
}

// remove deletes the chiton server from path, deleting the file if nothing else is left in it.
func (t serverTable) remove(path string) (changed bool, err error) {
	doc, found, err := readDoc(path, t.unmarshal)
	if !found || err != nil {
		return false, err
	}
	servers, _ := doc[t.key].(map[string]any)
	if _, ok := servers["chiton"]; !ok {
		return false, nil
	}
	delete(servers, "chiton")
	if len(servers) == 0 {
		delete(doc, t.key)
	}
	return true, saveOrRemove(path, doc, len(doc) == 0 || (t.shell != nil && t.shell(doc)), t.marshal)
}

// has reports whether path registers a chiton server. It returns false, nil for a missing file.
func (t serverTable) has(path string) (bool, error) {
	doc, _, err := readDoc(path, t.unmarshal)
	if err != nil {
		return false, errors.Unwrap(err) // return the bare cause because Doctor prefixes the path itself
	}
	servers, _ := doc[t.key].(map[string]any)
	_, ok := servers["chiton"]
	return ok, nil
}
