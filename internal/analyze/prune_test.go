package analyze_test

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

// storeSpy records the prune queries and rule-node upserts that reach the wrapped store.
type storeSpy struct {
	*badgerstore.Store
	filters     []store.NodeFilter
	ruleUpserts int
}

func (s *storeSpy) NodesMatching(ctx context.Context, f store.NodeFilter) ([]model.Node, error) {
	s.filters = append(s.filters, f)
	return s.Store.NodesMatching(ctx, f)
}

func (s *storeSpy) UpsertNodes(ctx context.Context, nodes []model.Node) error {
	for _, n := range nodes {
		if n.Kind == model.KindRule {
			s.ruleUpserts++
		}
	}
	return s.Store.UpsertNodes(ctx, nodes)
}

// syncFile mimics ingest.Engine.SyncFile: read prior findings, replace path's nodes, then re-analyze.
func syncFile(t *testing.T, gs store.GraphStore, az *analyze.Analyzer, path, src string) {
	t.Helper()
	ctx := context.Background()
	write(t, path, src)
	uri := model.FileURI(path)
	prior, err := az.ReadPriorFindings(ctx, uri)
	if err != nil {
		t.Fatal(err)
	}
	nodes, edges, err := extast.ForFile(path).ExtractSource(ctx, path, []byte(src))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gs.DeleteBySourceURI(ctx, uri); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, edges); err != nil {
		t.Fatal(err)
	}
	if _, err := az.ReanalyzeExtracted(ctx, path, []byte(src), nodes, edges, prior); err != nil {
		t.Fatalf("ReanalyzeExtracted: %v", err)
	}
}

// TestReanalyzeExtractedPrunesFromPrior checks live sync prunes a resolved finding without querying the
// store, and leaves another file's findings alone.
func TestReanalyzeExtractedPrunesFromPrior(t *testing.T) {
	spy := &storeSpy{Store: openStore(t)}
	az := analyze.New(spy, nil)
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a.go"), filepath.Join(dir, "b.go")
	syncFile(t, spy, az, a, "package a\n\n// TODO: fix a\nfunc A() {}\n")
	syncFile(t, spy, az, b, "package a\n\n// TODO: fix b\nfunc B() {}\n")

	spy.filters = nil
	syncFile(t, spy, az, a, "package a\n\nfunc A() {}\n")
	if len(spy.filters) != 0 {
		t.Errorf("re-sync with prior findings queried the store: %+v", spy.filters)
	}
	if got := countFindings(t, spy.Store, model.FileURI(a), todoRule); got != 0 {
		t.Errorf("a.go's resolved TODO survived: %d left", got)
	}
	if got := countFindings(t, spy.Store, model.FileURI(b), todoRule); got != 1 {
		t.Errorf("b.go's TODO was touched: want 1, got %d", got)
	}
}

// TestAnalyzeFile_PruneScopedToFile checks a single-file prune queries only that file's findings and rejects a
// probe false positive: another file's finding that merely mentions this file's URI.
func TestAnalyzeFile_PruneScopedToFile(t *testing.T) {
	ctx := context.Background()
	spy := &storeSpy{Store: openStore(t)}
	az := analyze.New(spy, nil)
	dir := t.TempDir()
	pathA, pathB := filepath.Join(dir, "a.go"), filepath.Join(dir, "b.go")
	uriA, uriB := model.FileURI(pathA), model.FileURI(pathB)
	write(t, pathA, "package a\n\n// TODO: fix a\nfunc A() {}\n")
	write(t, pathB, "package a\n\n// TODO: fix b\nfunc B() {}\n")
	for _, p := range []string{pathA, pathB} {
		if _, err := az.AnalyzeFile(ctx, p); err != nil {
			t.Fatalf("AnalyzeFile %s: %v", p, err)
		}
	}
	decoyID := "decoyfinding0000000000000000bb01"
	decoy := model.Node{
		ID: decoyID, Kind: model.KindFinding, Label: todoRule, SourceURI: "finding://" + decoyID,
		Provenance: model.ProvExtracted, Confidence: 1,
		Attrs: map[string]any{
			"rule_id": todoRule, "message": "see " + uriA, "target_uri": uriB,
			"rule_provenance": string(analyze.ProvenanceDeterministic),
		},
	}
	if err := spy.UpsertNodes(ctx, []model.Node{decoy}); err != nil {
		t.Fatal(err)
	}

	write(t, pathA, "package a\n\nfunc A() {}\n")
	spy.filters = nil
	if _, err := az.AnalyzeFile(ctx, pathA); err != nil {
		t.Fatal(err)
	}
	if len(spy.filters) != 1 || spy.filters[0].Attrs["target_uri"] != uriA {
		t.Errorf("prune queries = %+v, want one scoped to target_uri %s", spy.filters, uriA)
	}
	if got := countFindings(t, spy.Store, uriA, todoRule); got != 0 {
		t.Errorf("a.go's resolved TODO survived: %d left", got)
	}
	if got := countFindings(t, spy.Store, uriB, todoRule); got != 2 {
		t.Errorf("b.go's findings (real + decoy) were touched: want 2, got %d", got)
	}
}

// TestPersistUpsertsRuleNodesOnce checks an Analyzer writes each rule node once, not on every file.
func TestPersistUpsertsRuleNodesOnce(t *testing.T) {
	spy := &storeSpy{Store: openStore(t)}
	az := analyze.New(spy, nil)
	dir := t.TempDir()
	for _, name := range []string{"a.go", "b.go"} {
		p := filepath.Join(dir, name)
		write(t, p, "package a\n\n// TODO: x\nfunc F() {}\n")
		before := spy.ruleUpserts
		if _, err := az.AnalyzeFile(context.Background(), p); err != nil {
			t.Fatal(err)
		}
		if name == "b.go" && spy.ruleUpserts != before {
			t.Errorf("second file re-upserted %d rule nodes", spy.ruleUpserts-before)
		}
	}
	if spy.ruleUpserts == 0 {
		t.Error("first file upserted no rule nodes")
	}
}
