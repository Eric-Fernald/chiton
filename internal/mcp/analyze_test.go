package mcpsrv

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	mcplib "github.com/mark3labs/mcp-go/mcp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/llm" // register LLM rules in the default registry
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

func resultText(t *testing.T, res *mcplib.CallToolResult) string {
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

func toolReq(args map[string]any) mcplib.CallToolRequest {
	return mcplib.CallToolRequest{Params: mcplib.CallToolParams{Arguments: args}}
}

func openStore(t *testing.T) *badgerstore.Store {
	t.Helper()
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	return gs
}

func TestDiffFindingsHandlerResolved(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	dir := analyze.BaselineDir(t.TempDir())

	// Baseline holds one finding; the live store has none → it is resolved.
	base := []analyze.FindingView{{ID: "x", RuleID: "r", Severity: "high", Message: "m"}}
	if _, err := analyze.SaveBaseline(dir, "c1", base, time.Now()); err != nil {
		t.Fatal(err)
	}

	res, err := diffFindingsHandler(gs, dir, ".")(ctx, toolReq(map[string]any{"baseline": "latest"}))
	if err != nil {
		t.Fatal(err)
	}
	var d analyze.DiffResult
	if err := json.Unmarshal([]byte(resultText(t, res)), &d); err != nil {
		t.Fatalf("unmarshal diff: %v\n%s", err, resultText(t, res))
	}
	if len(d.Resolved) != 1 || d.Resolved[0].ID != "x" {
		t.Fatalf("expected 1 resolved, got %+v", d)
	}
	if d.Baseline != "c1" {
		t.Fatalf("baseline label = %q", d.Baseline)
	}
}

func TestDiffFindingsHandlerNoBaseline(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	res, err := diffFindingsHandler(gs, analyze.BaselineDir(t.TempDir()), ".")(ctx, toolReq(map[string]any{}))
	if err != nil {
		t.Fatal(err)
	}
	if got := resultText(t, res); !strings.Contains(got, "no baseline") {
		t.Fatalf("expected no-baseline message, got %q", got)
	}
}

func TestSubmitAnalysisHandlerBindsArguments(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	svc := analyze.NewLLMService(gs, nil, 0.5, t.TempDir())

	res, err := submitAnalysisHandler(svc)(ctx, toolReq(map[string]any{
		"task_id":    "task1",
		"rule_id":    "llm/injection-risk",
		"target_uri": "file://x.go",
		"findings": []any{
			map[string]any{"line": float64(3), "message": "untrusted input into query", "confidence": 0.9},
		},
	}))
	if err != nil {
		t.Fatal(err)
	}
	var rep analyze.LLMReport
	if err := json.Unmarshal([]byte(resultText(t, res)), &rep); err != nil {
		t.Fatalf("unmarshal report: %v\n%s", err, resultText(t, res))
	}
	if rep.FindingsAdded != 1 {
		t.Fatalf("expected 1 finding added, got %+v", rep)
	}

	// The finding landed as an INFERRED LLM finding.
	all, _ := gs.AllNodes(ctx)
	var llmFindings int
	for _, n := range all {
		if n.Kind == model.KindFinding && n.Provenance == model.ProvInferred {
			llmFindings++
		}
	}
	if llmFindings != 1 {
		t.Fatalf("expected 1 INFERRED finding in store, got %d", llmFindings)
	}
}

func TestSubmitAnalysisReadOnlyStore(t *testing.T) {
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "graph.badger")
	rw, err := badgerstore.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	rw.Close()
	ro, err := badgerstore.OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ro.Close()

	svc := analyze.NewLLMService(ro, nil, 0.5, t.TempDir())
	res, err := submitAnalysisHandler(svc)(ctx, toolReq(map[string]any{
		"task_id":  "t1",
		"rule_id":  "llm/injection-risk",
		"findings": []any{map[string]any{"line": float64(1), "message": "x", "confidence": 0.9}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if got := resultText(t, res); !strings.Contains(got, "read-only") || !strings.Contains(got, "writable") {
		t.Fatalf("expected an actionable read-only message, got %q", got)
	}
}

func TestPendingAnalysesNilService(t *testing.T) {
	res, err := pendingAnalysesHandler(nil, ".")(context.Background(), toolReq(map[string]any{}))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(resultText(t, res), "unavailable") {
		t.Fatalf("expected unavailable message, got %q", resultText(t, res))
	}
}

// Task specs carry file contents, so pending_analyses must not walk outside the project root.
func TestPendingAnalysesStaysInsideProjectRoot(t *testing.T) {
	ctx := context.Background()
	root, outside := t.TempDir(), t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "in.go"), []byte("package in\n\nfunc F() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outside, "secret.go"), []byte("package s\n\nconst key = \"hunter2\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	h := pendingAnalysesHandler(analyze.NewLLMService(openStore(t), nil, 0.5, t.TempDir()), root)

	for _, p := range []string{outside, "..", filepath.Join("..", filepath.Base(outside))} {
		res, err := h(ctx, toolReq(map[string]any{"path": p}))
		if err != nil {
			t.Fatal(err)
		}
		if !res.IsError || strings.Contains(resultText(t, res), "hunter2") {
			t.Errorf("pending_analyses(path=%q) was not refused: %s", p, resultText(t, res))
		}
	}
	res, err := h(ctx, toolReq(map[string]any{"limit": 1}))
	if err != nil {
		t.Fatal(err)
	}
	if res.IsError || !strings.Contains(resultText(t, res), "in.go") {
		t.Errorf("the default path should walk the project root: %s", resultText(t, res))
	}
}

// diff_findings must resolve refs in the project repo, not the server's working directory.
func TestResolveBaselineRefUsesProjectRoot(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	for _, args := range [][]string{{"init", "-q"}, {"commit", "--allow-empty", "-q", "-m", "c"}} {
		cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	want := analyze.GitCommit(root, "HEAD")
	if len(want) != 40 {
		t.Fatalf("setup: GitCommit(HEAD) = %q", want)
	}
	if got := resolveBaselineRef(root, "HEAD"); got != want {
		t.Errorf("resolveBaselineRef(root, HEAD) = %q, want the project's HEAD %q", got, want)
	}
}
