package install

import (
	toml "github.com/pelletier/go-toml/v2"
)

// codexTable is Codex CLI's single config.toml, where MCP servers are [mcp_servers.<name>] tables
// alongside unrelated settings (model, approval policy, ...).
var codexTable = serverTable{
	key:       "mcp_servers",
	kind:      "TOML table",
	unmarshal: toml.Unmarshal,
	marshal:   toml.Marshal,
}

// chitonCodexEntry is the [mcp_servers.chiton] table (graphPath may be empty).
func chitonCodexEntry(exe, graphPath string) map[string]any {
	return map[string]any{
		"command": exe,
		"args":    ChitonServeArgs(graphPath),
	}
}
