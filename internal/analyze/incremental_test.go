package analyze_test

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules" // register rules
	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
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
	uri := model.FileURI(path)

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
	uri := model.FileURI(path)

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

// TestPruneFileURI_RemovesLLMFindings verifies file deletion also prunes LLM findings, which
// live sync never re-runs rules to supersede.
func TestPruneFileURI_RemovesLLMFindings(t *testing.T) {
	gs := openStore(t)
	ctx := context.Background()
	uri := model.FileURI(filepath.Join(t.TempDir(), "svc.go"))

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

// TestAnalyzeFile_PreservesGraphScopeFindings verifies a single-file re-analyze keeps the file's
// deterministic GraphScope findings, which only a full `chiton analyze` regenerates.
func TestAnalyzeFile_PreservesGraphScopeFindings(t *testing.T) {
	gs := openStore(t)
	ctx := context.Background()
	dir := t.TempDir()
	path := filepath.Join(dir, "a.go")
	uri := model.FileURI(path)
	write(t, path, "package a\n\nfunc A() {}\n")

	// Synthetic deterministic GraphScope finding targeting the file, as a full
	// `chiton analyze` would have produced. Its rule is not a FileRule.
	id := "graphfinding0000000000000000aa11"
	gsf := model.Node{
		ID:         id,
		Kind:       model.KindFinding,
		Label:      "architecture/dead-symbol",
		SourceURI:  "finding://" + id,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		Attrs: map[string]any{
			"rule_id":         "architecture/dead-symbol",
			"category":        "architecture",
			"severity":        "low",
			"message":         "A is never referenced",
			"target_uri":      uri,
			"rule_provenance": string(analyze.ProvenanceDeterministic),
		},
	}
	if err := gs.UpsertNodes(ctx, []model.Node{gsf}); err != nil {
		t.Fatal(err)
	}

	az := analyze.New(gs, nil)
	if _, err := az.AnalyzeFile(ctx, path); err != nil {
		t.Fatalf("AnalyzeFile: %v", err)
	}
	if got, _ := gs.GetNode(ctx, id); got == nil {
		t.Error("GraphScope finding was wrongly pruned by a single-file re-analyze")
	}
}

// TestAnalyzeExtractedMatchesAnalyzeFile checks a caller-supplied extraction yields AnalyzeFile's findings.
func TestAnalyzeExtractedMatchesAnalyzeFile(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "a.go")
	src := []byte("package a\n\nimport \"fmt\"\n\n// TODO: fix this\nfunc A() {\n\tdefer fmt.Println(\"x\")\n\tfor {\n\t\tdefer fmt.Println(\"y\")\n\t}\n}\n")
	write(t, path, string(src))

	wantStore, gotStore := openStore(t), openStore(t)
	want, err := analyze.New(wantStore, nil).AnalyzeFile(ctx, path)
	if err != nil {
		t.Fatalf("AnalyzeFile: %v", err)
	}
	nodes, edges, err := extast.ForFile(path).ExtractSource(ctx, path, src)
	if err != nil {
		t.Fatal(err)
	}
	got, err := analyze.New(gotStore, nil).AnalyzeExtracted(ctx, path, src, nodes, edges)
	if err != nil {
		t.Fatalf("AnalyzeExtracted: %v", err)
	}
	if len(want.Findings) < 2 {
		t.Fatalf("fixture should trip several rules, got %d findings", len(want.Findings))
	}
	if got.FilesScanned != want.FilesScanned || got.RulesRun != want.RulesRun || !reflect.DeepEqual(got.Findings, want.Findings) {
		t.Errorf("AnalyzeExtracted = %+v\nwant %+v", got, want)
	}
	wantViews, _ := analyze.List(ctx, wantStore, analyze.ListOptions{})
	gotViews, _ := analyze.List(ctx, gotStore, analyze.ListOptions{})
	if !reflect.DeepEqual(gotViews, wantViews) {
		t.Errorf("stored findings differ:\n got %+v\nwant %+v", gotViews, wantViews)
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
