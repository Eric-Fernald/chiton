package install

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
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

// TestWriteMCPConfigAcceptsJSONC verifies a JSONC config, as OpenCode and VS Code document it, is merged
// rather than rejected, preserving unrelated servers.
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

	if _, _, err := mcpServersTable.write(path, ChitonServerEntry("chiton", "")); err != nil {
		t.Fatalf("write on JSONC config: %v", err)
	}

	has, err := mcpServersTable.has(path)
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

// TestWriteMCPConfigRejectsNonObject verifies a non-object "mcpServers" is refused with an error, so a
// malformed config is never clobbered.
func TestWriteMCPConfigRejectsNonObject(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mcp.json")
	if err := os.WriteFile(path, []byte(`{"mcpServers": "oops-a-string"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	original, _ := os.ReadFile(path)
	if _, _, err := mcpServersTable.write(path, ChitonServerEntry("chiton", "")); err == nil {
		t.Fatal("expected an error when mcpServers is not an object")
	}
	// The original file must be untouched (no clobber).
	now, _ := os.ReadFile(path)
	if string(now) != string(original) {
		t.Errorf("file was modified despite the error:\n%s", now)
	}
}

// TestUpsertRulesBlockCollapsesDuplicates verifies a file that somehow contains
// two chiton blocks ends with exactly one after an upsert.
func TestUpsertRulesBlockCollapsesDuplicates(t *testing.T) {
	path := filepath.Join(t.TempDir(), "CLAUDE.md")
	dup := "# My notes\n\n" +
		ChitonBlockStart + "\nold one\n" + ChitonBlockEnd + "\n\n" +
		"middle user text\n\n" +
		ChitonBlockStart + "\nold two\n" + ChitonBlockEnd + "\n"
	if err := os.WriteFile(path, []byte(dup), 0o644); err != nil {
		t.Fatal(err)
	}

	if _, _, err := UpsertRulesBlock(path, "fresh content"); err != nil {
		t.Fatalf("UpsertRulesBlock: %v", err)
	}
	out, _ := os.ReadFile(path)
	if n := strings.Count(string(out), ChitonBlockStart); n != 1 {
		t.Fatalf("expected exactly 1 chiton block, got %d:\n%s", n, out)
	}
	if !strings.Contains(string(out), "fresh content") {
		t.Errorf("canonical block content missing:\n%s", out)
	}
	if !strings.Contains(string(out), "middle user text") {
		t.Errorf("user text between blocks was lost:\n%s", out)
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

	if _, _, err := openCodeTable.write(path, chitonOpenCodeEntry("chiton", "")); err != nil {
		t.Fatalf("write on JSONC config: %v", err)
	}
	has, err := openCodeTable.has(path)
	if err != nil || !has {
		t.Fatalf("chiton entry missing after write (has=%v err=%v)", has, err)
	}
}
