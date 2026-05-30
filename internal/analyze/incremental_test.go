package analyze_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules" // register rules
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

const todoRule = "quality/todo-comment"

func openStore(t *testing.T) *badgerstore.Store {
	t.Helper()
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { gs.Close() })
	return gs
}

func countFindings(t *testing.T, gs *badgerstore.Store, uri, ruleID string) int {
	t.Helper()
	views, err := analyze.List(context.Background(), gs, analyze.ListOptions{Scope: uri})
	if err != nil {
		t.Fatalf("list findings: %v", err)
	}
	n := 0
	for _, v := range views {
		if v.RuleID == ruleID {
			n++
		}
	}
	return n
}

func TestAnalyzeFile_AddsThenPrunes(t *testing.T) {
	gs := openStore(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "a.go")
	uri := "file://" + filepath.ToSlash(mustAbs(t, path))

	// File with a TODO marker.
	write(t, path, "package a\n\n// TODO: fix this\nfunc A() {}\n")
	az := analyze.New(gs, nil)

	if _, err := az.AnalyzeFile(context.Background(), path); err != nil {
		t.Fatalf("AnalyzeFile: %v", err)
	}
	if got := countFindings(t, gs, uri, todoRule); got != 1 {
		t.Fatalf("after add: want 1 todo finding, got %d", got)
	}

	// Rewrite without the TODO; re-analyze should prune the resolved finding.
	write(t, path, "package a\n\nfunc A() {}\n")
	if _, err := az.AnalyzeFile(context.Background(), path); err != nil {
		t.Fatalf("AnalyzeFile #2: %v", err)
	}
	if got := countFindings(t, gs, uri, todoRule); got != 0 {
		t.Fatalf("after fix: want 0 todo findings, got %d", got)
	}
}

func TestPruneFileURI(t *testing.T) {
	gs := openStore(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "b.go")
	uri := "file://" + filepath.ToSlash(mustAbs(t, path))

	write(t, path, "package b\n\n// FIXME later\nfunc B() {}\n")
	az := analyze.New(gs, nil)
	if _, err := az.AnalyzeFile(context.Background(), path); err != nil {
		t.Fatalf("AnalyzeFile: %v", err)
	}
	if got := countFindings(t, gs, uri, todoRule); got != 1 {
		t.Fatalf("want 1 finding before prune, got %d", got)
	}

	n, err := az.PruneFileURI(context.Background(), uri)
	if err != nil {
		t.Fatalf("PruneFileURI: %v", err)
	}
	if n != 1 {
		t.Errorf("PruneFileURI removed %d, want 1", n)
	}
	if got := countFindings(t, gs, uri, todoRule); got != 0 {
		t.Fatalf("want 0 findings after prune, got %d", got)
	}
}

// TestPruneFileURI_RemovesLLMFindings verifies the delete path prunes LLM
// findings (rule_provenance=LLM), not just deterministic ones — otherwise a
// deleted file's LLM findings linger forever (live-sync never re-runs LLM
// rules to supersede them).
func TestPruneFileURI_RemovesLLMFindings(t *testing.T) {
	gs := openStore(t)
	ctx := context.Background()
	uri := "file://" + filepath.ToSlash(mustAbs(t, filepath.Join(t.TempDir(), "svc.go")))

	// Insert a synthetic LLM finding node attached to uri via target_uri.
	id := "llmfinding00000000000000000000aa"
	llm := model.Node{
		ID:         id,
		Kind:       model.KindFinding,
		Label:      "llm finding",
		SourceURI:  "finding://" + id,
		Provenance: model.ProvInferred,
		Confidence: 0.9,
		Attrs: map[string]any{
			"rule_id":         "llm/race-condition-risk",
			"category":        "bugs",
			"severity":        "high",
			"message":         "possible race",
			"target_uri":      uri,
			"rule_provenance": string(analyze.ProvenanceLLM),
		},
	}
	if err := gs.UpsertNodes(ctx, []model.Node{llm}); err != nil {
		t.Fatal(err)
	}

	az := analyze.New(gs, nil)
	n, err := az.PruneFileURI(ctx, uri)
	if err != nil {
		t.Fatalf("PruneFileURI: %v", err)
	}
	if n != 1 {
		t.Errorf("expected 1 LLM finding pruned, got %d", n)
	}
	if got, _ := gs.GetNode(ctx, id); got != nil {
		t.Error("LLM finding node should have been deleted on file delete")
	}
}

func TestAnalyzeFile_UnsupportedIsNoop(t *testing.T) {
	gs := openStore(t)
	path := filepath.Join(t.TempDir(), "notes.txt")
	write(t, path, "TODO: not a source file\n")
	az := analyze.New(gs, nil)
	res, err := az.AnalyzeFile(context.Background(), path)
	if err != nil {
		t.Fatalf("AnalyzeFile: %v", err)
	}
	if res.FilesScanned != 0 {
		t.Errorf("unsupported file should scan 0 files, got %d", res.FilesScanned)
	}
}

func write(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func mustAbs(t *testing.T, path string) string {
	t.Helper()
	abs, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	return abs
}
