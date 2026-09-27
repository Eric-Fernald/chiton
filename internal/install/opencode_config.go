package install

// OpenCode keeps MCP servers under a top-level "mcp" object (not "mcpServers"), and a local server's
// executable and arguments share one flat "command" array. There is no separate "args" field.

const openCodeSchema = "https://opencode.ai/config.json"

// openCodeEntry is the chiton local-server entry in OpenCode's "mcp" object.
type openCodeEntry struct {
	Type    string   `json:"type"`
	Command []string `json:"command"`
	Enabled bool     `json:"enabled"`
}

// openCodeTable adds the $schema marker on write and treats a file holding only that marker as empty.
var openCodeTable = serverTable{
	key:       "mcp",
	kind:      "JSON object",
	unmarshal: unmarshalJSONC,
	marshal:   marshalJSON,
	seed: func(doc map[string]any) {
		if _, ok := doc["$schema"]; !ok {
			doc["$schema"] = openCodeSchema
		}
	},
	shell: func(doc map[string]any) bool { return len(doc) == 1 && doc["$schema"] == openCodeSchema },
}

// chitonOpenCodeEntry is the chiton entry under OpenCode's "mcp" (graphPath may be empty).
func chitonOpenCodeEntry(exe, graphPath string) openCodeEntry {
	return openCodeEntry{
		Type:    "local",
		Command: append([]string{exe}, ChitonServeArgs(graphPath)...),
		Enabled: true,
	}
}
