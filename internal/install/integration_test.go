package install

import (
	"context"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	mcpclient "github.com/mark3labs/mcp-go/client"
	mcplib "github.com/mark3labs/mcp-go/mcp"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

// TestInstallers_ServerLaunch launches the MCP server from the command each installer wrote into its
// config and checks a real tools/call. Aider is absent because it has no MCP support.
func TestInstallers_ServerLaunch(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the chiton binary and spawns subprocesses; skipped in -short")
	}
	bin := buildChiton(t)

	home := t.TempDir()
	setHome(t, home)

	probes := []struct {
		name       string
		configPath func(repo, home string) string
		recover    func(t *testing.T, path string) (string, []string)
	}{
		{"cursor", func(repo, _ string) string { return filepath.Join(repo, ".cursor", "mcp.json") }, recoverMCPServers},
		{"gemini-cli", func(repo, _ string) string { return filepath.Join(repo, ".gemini", "settings.json") }, recoverMCPServers},
		{"opencode", func(repo, _ string) string { return filepath.Join(repo, "opencode.json") }, recoverOpenCode},
		{"codex", func(_, home string) string { return filepath.Join(home, ".codex", "config.toml") }, recoverCodex},
		{"antigravity", func(_, home string) string { return filepath.Join(home, ".gemini", "config", "mcp_config.json") }, recoverMCPServers},
		{"copilot-cli", func(_, home string) string { return filepath.Join(home, ".copilot", "mcp-config.json") }, recoverMCPServers},
	}

	for _, p := range probes {
		t.Run(p.name, func(t *testing.T) {
			in := Lookup(p.name)
			if in == nil {
				t.Fatalf("installer %q not registered", p.name)
			}
			repo := t.TempDir()
			store := filepath.Join(t.TempDir(), "graph.badger")
			seedStore(t, store)

			opts := Options{Dir: repo, Exe: bin, GraphPath: store}
			if _, err := in.Install(opts); err != nil {
				t.Fatalf("Install: %v", err)
			}

			cmd, args := p.recover(t, p.configPath(repo, home))
			if cmd != bin {
				t.Fatalf("config records command %q, want built binary %q", cmd, bin)
			}
			assertToolCallSucceeds(t, cmd, args)
		})
	}
}

// assertToolCallSucceeds starts the recovered command as an MCP server, does the
// initialize handshake, and calls query_graph for the seeded node.
func assertToolCallSucceeds(t *testing.T, command string, args []string) {
	t.Helper()
	cli, err := mcpclient.NewStdioMCPClient(command, nil, args...)
	if err != nil {
		t.Fatalf("start MCP server: %v", err)
	}
	defer cli.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	initReq := mcplib.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcplib.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcplib.Implementation{Name: "chiton-itest", Version: "0"}
	if _, err := cli.Initialize(ctx, initReq); err != nil {
		t.Fatalf("initialize: %v", err)
	}

	callReq := mcplib.CallToolRequest{}
	callReq.Params.Name = "query_graph"
	callReq.Params.Arguments = map[string]any{"query": "fetchall"}
	res, err := cli.CallTool(ctx, callReq)
	if err != nil {
		t.Fatalf("tools/call query_graph: %v", err)
	}
	if res.IsError {
		t.Fatalf("query_graph returned an error result: %s", toolText(t, res))
	}
	if txt := toolText(t, res); !strings.Contains(txt, "fetchall") {
		t.Fatalf("expected query_graph to find the seeded node; got: %s", txt)
	}
}

func toolText(t *testing.T, res *mcplib.CallToolResult) string {
	t.Helper()
	if res == nil || len(res.Content) == 0 {
		t.Fatal("empty tool result")
	}
	tc, ok := mcplib.AsTextContent(res.Content[0])
	if !ok {
		t.Fatalf("expected text content, got %T", res.Content[0])
	}
	return tc.Text
}

// buildChiton compiles the chiton CLI into a temp dir and returns its path. If
// the Go toolchain is unavailable the test is skipped rather than failed.
func buildChiton(t *testing.T) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "chiton")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "go", "build", "-o", bin, "github.com/Eric-Fernald/chiton/cmd/chiton").CombinedOutput()
	if err != nil {
		t.Skipf("could not build chiton binary (Go toolchain required): %v\n%s", err, out)
	}
	return bin
}

// seedStore creates a badger graph store at path holding one symbol node so a
// query_graph call has something to find.
func seedStore(t *testing.T, path string) {
	t.Helper()
	gs, err := badgerstore.Open(path)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	n := model.Node{
		ID:         model.NewNodeID("file://x.go", "fetchall"),
		Kind:       model.KindSymbol,
		Label:      "fetchall",
		SourceURI:  "file://x.go",
		Provenance: model.ProvExtracted,
		Confidence: 1,
	}
	if err := gs.UpsertNodes(context.Background(), []model.Node{n}); err != nil {
		gs.Close()
		t.Fatalf("seed node: %v", err)
	}
	if err := gs.Close(); err != nil {
		t.Fatalf("close store: %v", err)
	}
}

// --- config recovery (read back the launch command the installer wrote) ---

func recoverMCPServers(t *testing.T, path string) (string, []string) {
	t.Helper()
	servers, _ := readJSON(t, path)["mcpServers"].(map[string]any)
	entry, _ := servers["chiton"].(map[string]any)
	return asString(entry["command"]), asStringSlice(entry["args"])
}

func recoverOpenCode(t *testing.T, path string) (string, []string) {
	t.Helper()
	mcp, _ := readJSON(t, path)["mcp"].(map[string]any)
	entry, _ := mcp["chiton"].(map[string]any)
	cmd := asStringSlice(entry["command"])
	if len(cmd) == 0 {
		t.Fatalf("opencode chiton.command empty in %s", path)
	}
	return cmd[0], cmd[1:]
}

func recoverCodex(t *testing.T, path string) (string, []string) {
	t.Helper()
	servers, _ := readTOML(t, path)["mcp_servers"].(map[string]any)
	entry, _ := servers["chiton"].(map[string]any)
	return asString(entry["command"]), asStringSlice(entry["args"])
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}

func asStringSlice(v any) []string {
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(arr))
	for _, e := range arr {
		if s, ok := e.(string); ok {
			out = append(out, s)
		}
	}
	return out
}
