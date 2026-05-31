package analyze_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
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

// TestAnalyzePrunesResolvedFindings confirms that re-analysing a file after the
// offending code is removed deletes the stale deterministic finding, so the
// store reflects live truth and --diff can report it as resolved.
func TestAnalyzePrunesResolvedFindings(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	leaky := filepath.Join(tmp, "leaky.go")
	if err := os.WriteFile(leaky, []byte("package main\n\nconst Key = \"AKIAIOSFODNN7EXAMPLE\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	gs, err := badgerstore.Open(filepath.Join(tmp, "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	az := analyze.New(gs, nil)
	if _, err := az.Analyze(ctx, tmp); err != nil {
		t.Fatal(err)
	}
	before, _ := analyze.List(ctx, gs, analyze.ListOptions{Category: "security"})
	if len(before) == 0 {
		t.Fatal("expected a secret finding on first analyze")
	}

	// Remove the secret and re-analyze.
	if err := os.WriteFile(leaky, []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := az.Analyze(ctx, tmp); err != nil {
		t.Fatal(err)
	}
	after, _ := analyze.List(ctx, gs, analyze.ListOptions{Category: "security"})
	if len(after) != 0 {
		t.Fatalf("expected the resolved finding to be pruned, still have %d: %+v", len(after), after)
	}
}

// TestPruneDoesNotDeleteLLMFindings confirms a deterministic re-scan never
// prunes LLM (rule_provenance=LLM) findings — they have their own lifecycle.
func TestPruneDoesNotDeleteLLMFindings(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	clean := filepath.Join(tmp, "clean.go")
	if err := os.WriteFile(clean, []byte("package main\n\nfunc helper() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gs, err := badgerstore.Open(filepath.Join(tmp, "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	absClean, _ := filepath.Abs(clean)
	uri := "file://" + filepath.ToSlash(absClean)
	// task_id must match LLMTaskID(rule, sha(file)) — Submit now verifies the
	// target_uri binding to prevent cross-file finding deletion.
	sum := sha256.Sum256([]byte("package main\n\nfunc helper() {}\n"))
	taskID := analyze.LLMTaskID("llm/injection-risk", hex.EncodeToString(sum[:]))
	llm := analyze.NewLLMService(gs, nil, 0.5, filepath.Join(tmp, ".chiton"))
	if _, err := llm.Submit(ctx, analyze.LLMResult{
		TaskID: taskID, RuleID: "llm/injection-risk", TargetURI: uri,
		Findings: []analyze.LLMFindingResult{{Line: 3, Message: "untrusted flow", Confidence: 0.9}},
	}); err != nil {
		t.Fatal(err)
	}

	// A deterministic analyze over the same file must not touch the LLM finding.
	if _, err := analyze.New(gs, nil).Analyze(ctx, tmp); err != nil {
		t.Fatal(err)
	}
	all, _ := gs.AllNodes(ctx)
	var llmFindings int
	for _, n := range all {
		if n.Kind == model.KindFinding {
			if rp, _ := n.Attrs["rule_provenance"].(string); rp == "LLM" {
				llmFindings++
			}
		}
	}
	if llmFindings != 1 {
		t.Fatalf("deterministic analyze deleted an LLM finding: want 1, got %d", llmFindings)
	}
}

// TestPruneRespectsScanScope confirms a partial `analyze <subdir>` never prunes
// findings for files outside the scanned scope.
func TestPruneRespectsScanScope(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	dirA := filepath.Join(tmp, "a")
	dirB := filepath.Join(tmp, "b")
	if err := os.MkdirAll(dirA, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dirB, 0o755); err != nil {
		t.Fatal(err)
	}
	secret := "package p\n\nconst Key = \"AKIAIOSFODNN7EXAMPLE\"\n"
	if err := os.WriteFile(filepath.Join(dirA, "a.go"), []byte(secret), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dirB, "b.go"), []byte(secret), 0o644); err != nil {
		t.Fatal(err)
	}
	gs, err := badgerstore.Open(filepath.Join(tmp, "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	az := analyze.New(gs, nil)
	if _, err := az.Analyze(ctx, tmp); err != nil {
		t.Fatal(err)
	}
	if got := secretCount(ctx, t, gs); got != 2 {
		t.Fatalf("want 2 secret findings after full scan, got %d", got)
	}

	// Fix A's secret and analyze ONLY dirA. B's finding must remain.
	if err := os.WriteFile(filepath.Join(dirA, "a.go"), []byte("package p\n\nfunc f() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := az.Analyze(ctx, dirA); err != nil {
		t.Fatal(err)
	}
	views, _ := analyze.List(ctx, gs, analyze.ListOptions{Category: "security"})
	if len(views) != 1 {
		t.Fatalf("scoped analyze should leave B's finding intact: want 1, got %d: %+v", len(views), views)
	}
	if !strings.Contains(views[0].TargetURI, "/b/") {
		t.Fatalf("surviving finding should be B's, got %s", views[0].TargetURI)
	}
}

func secretCount(ctx context.Context, t *testing.T, gs *badgerstore.Store) int {
	t.Helper()
	views, err := analyze.List(ctx, gs, analyze.ListOptions{Category: "security"})
	if err != nil {
		t.Fatal(err)
	}
	return len(views)
}

// TestDiffEndToEnd exercises the real path the --diff feature ships: Analyze a
// repo, snapshot a baseline from the store, change the code, re-Analyze (which
// prunes resolved findings), and assert Diff reports exactly the right
// new/resolved set — and that a surviving finding's stable FindingID keeps it
// out of BOTH new and resolved.
func TestDiffEndToEnd(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	// File 1 keeps a stable secret across both scans; file 2's secret is removed.
	keep := filepath.Join(tmp, "keep.go")
	gone := filepath.Join(tmp, "gone.go")
	secret := func(name string) string {
		return "package p\n\nconst " + name + " = \"AKIAIOSFODNN7EXAMPLE\"\n"
	}
	if err := os.WriteFile(keep, []byte(secret("K")), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(gone, []byte(secret("G")), 0o644); err != nil {
		t.Fatal(err)
	}
	gs, err := badgerstore.Open(filepath.Join(tmp, "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	az := analyze.New(gs, nil)
	if _, err := az.Analyze(ctx, tmp); err != nil {
		t.Fatal(err)
	}
	baseline, err := analyze.CurrentFindings(ctx, gs)
	if err != nil {
		t.Fatal(err)
	}

	// Remove gone.go's secret and add a brand-new offence in a third file.
	if err := os.WriteFile(gone, []byte("package p\n\nfunc g() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tmp, "new.go"), []byte(secret("N")), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := az.Analyze(ctx, tmp); err != nil {
		t.Fatal(err)
	}
	current, err := analyze.CurrentFindings(ctx, gs)
	if err != nil {
		t.Fatal(err)
	}

	d := analyze.Diff(baseline, current)
	countSecret := func(fs []analyze.FindingView) int {
		n := 0
		for _, f := range fs {
			if f.RuleID == "security/hardcoded-secret" {
				n++
			}
		}
		return n
	}
	if countSecret(d.New) != 1 {
		t.Fatalf("want 1 new secret (new.go), got %d: %+v", countSecret(d.New), d.New)
	}
	if countSecret(d.Resolved) != 1 {
		t.Fatalf("want 1 resolved secret (gone.go), got %d: %+v", countSecret(d.Resolved), d.Resolved)
	}
	// keep.go's secret has a stable FindingID → neither new nor resolved.
	for _, f := range append(append([]analyze.FindingView{}, d.New...), d.Resolved...) {
		if f.RuleID == "security/hardcoded-secret" && strings.Contains(f.TargetURI, "keep.go") {
			t.Fatalf("stable finding spuriously reported as new/resolved: %+v", f)
		}
	}
}
