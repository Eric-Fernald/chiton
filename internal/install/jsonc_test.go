package install

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestJSONCToJSON(t *testing.T) {
	in := `{
  // a line comment
  "a": 1, /* a block comment */
  "b": "http://x//y /* not a comment */", // string content must survive
  "c": [1, 2,],
}`
	var doc map[string]any
	stripped := jsoncToJSON([]byte(in))
	if err := json.Unmarshal(stripped, &doc); err != nil {
		t.Fatalf("stripped JSONC did not parse: %v\n%s", err, stripped)
	}
	if doc["b"] != "http://x//y /* not a comment */" {
		t.Errorf("string content corrupted: %v", doc["b"])
	}
	if arr, _ := doc["c"].([]any); len(arr) != 2 {
		t.Errorf("trailing comma in array not handled: %v", doc["c"])
	}
}

// TestWriteMCPConfigAcceptsJSONC verifies an existing comment-laden config (the
// documented OpenCode/VS Code JSONC format) is merged rather than rejected, and
// that unrelated servers are preserved.
func TestWriteMCPConfigAcceptsJSONC(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mcp.json")
	jsonc := "{\n" +
		"  // my servers\n" +
		"  \"mcpServers\": {\n" +
		"    \"other\": { \"command\": \"x\", \"args\": [] }, /* keep me */\n" +
		"  },\n" +
		"}\n"
	if err := os.WriteFile(path, []byte(jsonc), 0o644); err != nil {
		t.Fatal(err)
	}

	if _, _, err := WriteMCPConfig(path, ChitonServerEntry("chiton", "")); err != nil {
		t.Fatalf("WriteMCPConfig on JSONC config: %v", err)
	}

	has, err := MCPConfigHasChiton(path)
	if err != nil || !has {
		t.Fatalf("chiton entry missing after write (has=%v err=%v)", has, err)
	}
	b, _ := os.ReadFile(path)
	var doc map[string]any
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("written output is not valid JSON: %v", err)
	}
	servers, _ := doc["mcpServers"].(map[string]any)
	if _, ok := servers["other"]; !ok {
		t.Error("pre-existing 'other' server was dropped")
	}
}

// TestWriteOpenCodeConfigAcceptsJSONC is the OpenCode ("mcp" object) analogue.
func TestWriteOpenCodeConfigAcceptsJSONC(t *testing.T) {
	path := filepath.Join(t.TempDir(), "opencode.jsonc")
	jsonc := "{\n" +
		"  \"$schema\": \"https://opencode.ai/config.json\",\n" +
		"  \"mcp\": {\n" +
		"    \"other\": { \"type\": \"local\", \"command\": [\"x\"], \"enabled\": true }, // comment\n" +
		"  },\n" +
		"}\n"
	if err := os.WriteFile(path, []byte(jsonc), 0o644); err != nil {
		t.Fatal(err)
	}

	if _, _, err := WriteOpenCodeConfig(path, "chiton", ""); err != nil {
		t.Fatalf("WriteOpenCodeConfig on JSONC config: %v", err)
	}
	has, err := OpenCodeConfigHasChiton(path)
	if err != nil || !has {
		t.Fatalf("chiton entry missing after write (has=%v err=%v)", has, err)
	}
}
