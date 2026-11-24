package mcpsrv

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	mcpsdk "github.com/mark3labs/mcp-go/server"
)

// rpcResult sends one JSON-RPC request to s and decodes its result into out.
func rpcResult(t *testing.T, s *mcpsdk.MCPServer, method string, params, out any) {
	t.Helper()
	req, err := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := json.Marshal(s.HandleMessage(context.Background(), req))
	if err != nil {
		t.Fatal(err)
	}
	var env struct {
		Result json.RawMessage `json:"result"`
		Error  json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(resp, &env); err != nil {
		t.Fatalf("unmarshal %s: %v", resp, err)
	}
	if env.Error != nil {
		t.Fatalf("%s returned an error: %s", method, env.Error)
	}
	if err := json.Unmarshal(env.Result, out); err != nil {
		t.Fatalf("unmarshal result %s: %v", env.Result, err)
	}
}

func listedTools(t *testing.T, s *mcpsdk.MCPServer) []string {
	t.Helper()
	var res struct {
		Tools []struct {
			Name string `json:"name"`
		} `json:"tools"`
	}
	rpcResult(t, s, "tools/list", map[string]any{}, &res)
	var names []string
	for _, tool := range res.Tools {
		names = append(names, tool.Name)
	}
	slices.Sort(names)
	return names
}

// callServerTool calls tool on s and returns the reply text and whether it was flagged as an error.
func callServerTool(t *testing.T, s *mcpsdk.MCPServer, tool string, args map[string]any) (string, bool) {
	t.Helper()
	var res struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	rpcResult(t, s, "tools/call", map[string]any{"name": tool, "arguments": args}, &res)
	if len(res.Content) == 0 {
		t.Fatalf("%s returned no content", tool)
	}
	return res.Content[0].Text, res.IsError
}

// TestUnbuiltServerListsTheSameTools checks a server started before the graph exists advertises every tool
// a built one does, so an assistant sees chiton's tools from the first session.
func TestUnbuiltServerListsTheSameTools(t *testing.T) {
	built := listedTools(t, New(openStore(t), nil, Deps{}))
	unbuilt := listedTools(t, NewUnbuilt(filepath.Join(t.TempDir(), "graph.kuzu"), "1.0.0"))
	if len(built) == 0 {
		t.Fatal("the built server lists no tools")
	}
	if !slices.Equal(built, unbuilt) {
		t.Errorf("unbuilt server tools = %v, want %v", unbuilt, built)
	}
}

// TestUnbuiltServerToolsSayHowToBuild checks every kind of tool answers with the build command and the path
// the graph is expected at, flagged as an error.
func TestUnbuiltServerToolsSayHowToBuild(t *testing.T) {
	path := filepath.Join(t.TempDir(), "chiton-out", "graph.kuzu")
	s := NewUnbuilt(path, "1.0.0")
	for _, tool := range []string{"graph_summary", "get_node", "semantic_search", "submit_extraction"} {
		text, isErr := callServerTool(t, s, tool, map[string]any{"label": "x", "query": "x"})
		if !isErr || !strings.Contains(text, "chiton build .") || !strings.Contains(text, path) {
			t.Errorf("%s replied (error=%v) %q, want an error naming 'chiton build .' and %s", tool, isErr, text, path)
		}
	}
}

// TestUnbuiltServerNoticesALaterBuild checks that once the graph appears, the reply says to reconnect and no
// longer says to build.
func TestUnbuiltServerNoticesALaterBuild(t *testing.T) {
	path := filepath.Join(t.TempDir(), "chiton-out", "graph.kuzu")
	s := NewUnbuilt(path, "1.0.0")
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	text, isErr := callServerTool(t, s, "graph_summary", map[string]any{})
	if !isErr || !strings.Contains(text, "reconnect") || strings.Contains(text, "chiton build .") {
		t.Errorf("reply (error=%v) %q, want an error saying to reconnect, without the build command", isErr, text)
	}
}
