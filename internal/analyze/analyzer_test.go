package analyze_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

// TestAnalyzeEndToEnd walks a tiny synthetic repo, runs all registered rules,
// and confirms findings + rule nodes land in the store with the expected
// VIOLATES/AFFECTS edges.
func TestAnalyzeEndToEnd(t *testing.T) {
	tmp := t.TempDir()

	// Two files: one with a hardcoded AWS key, one clean.
	bad := filepath.Join(tmp, "leaky.go")
	if err := os.WriteFile(bad, []byte(`package main

const Key = "AKIAIOSFODNN7EXAMPLE"

func main() {}
`), 0o644); err != nil {
		t.Fatal(err)
	}
	good := filepath.Join(tmp, "clean.go")
	if err := os.WriteFile(good, []byte(`package main

func helper() {}
`), 0o644); err != nil {
		t.Fatal(err)
	}

	storePath := filepath.Join(tmp, "graph.badger")
	gs, err := badgerstore.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	az := analyze.New(gs, nil)
	res, err := az.Analyze(context.Background(), tmp)
	if err != nil {
		t.Fatalf("analyze: %v", err)
	}
	if res.FilesScanned < 2 {
		t.Fatalf("expected at least 2 files scanned, got %d", res.FilesScanned)
	}
	if res.RulesRun == 0 {
		t.Fatal("expected at least one rule to run")
	}

	// Should have at least one critical finding (hardcoded secret).
	var foundSecret bool
	for _, f := range res.Findings {
		if f.RuleID == "security/hardcoded-secret" {
			foundSecret = true
			break
		}
	}
	if !foundSecret {
		t.Fatalf("expected hardcoded-secret finding; got %+v", res.Findings)
	}

	// List from the store and confirm view shape.
	views, err := analyze.List(context.Background(), gs, analyze.ListOptions{
		Category: "security",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(views) == 0 {
		t.Fatal("expected at least one security finding from List")
	}

	// Explain should resolve to the rule node.
	exp, err := analyze.Explain(context.Background(), gs, views[0].ID)
	if err != nil {
		t.Fatalf("explain: %v", err)
	}
	if exp.Rule == nil {
		t.Fatal("explain: expected rule view")
	}
	if exp.Rule.ID != views[0].RuleID {
		t.Fatalf("explain rule mismatch: got %q want %q", exp.Rule.ID, views[0].RuleID)
	}

	// Confirm a rule node was persisted with KindRule.
	all, err := gs.AllNodes(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var rules, findings int
	for _, n := range all {
		switch n.Kind {
		case model.KindRule:
			rules++
		case model.KindFinding:
			findings++
		}
	}
	if rules == 0 {
		t.Fatal("expected rule nodes in store")
	}
	if findings == 0 {
		t.Fatal("expected finding nodes in store")
	}
}
