package bench

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunOnSyntheticCorpus(t *testing.T) {
	dir := t.TempDir()
	if err := GenerateSynthetic(dir, 12); err != nil {
		t.Fatalf("generate: %v", err)
	}
	rep, err := Run(context.Background(), "synthetic", dir, Options{MaxQueries: 4})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if rep.Files != 12 {
		t.Errorf("files: got %d, want 12", rep.Files)
	}
	if rep.Nodes == 0 || rep.Edges == 0 {
		t.Fatalf("expected a non-empty graph; got %d nodes, %d edges", rep.Nodes, rep.Edges)
	}
	if len(rep.Queries) == 0 {
		t.Fatal("expected at least one benchmarked query")
	}
	for _, q := range rep.Queries {
		if q.Naive != rep.CorpusTokens {
			t.Errorf("naive cost should equal corpus tokens (%d); got %d for %q", rep.CorpusTokens, q.Naive, q.Question)
		}
		if q.Chiton <= 0 {
			t.Errorf("chiton cost should be positive; got %d for %q", q.Chiton, q.Question)
		}
		// A depth-1 neighborhood is a subset of the depth-2 one, so it can never cost more.
		if q.ChitonDepth1 <= 0 || q.ChitonDepth1 > q.Chiton {
			t.Errorf("depth-1 cost %d should be positive and at most depth-2 cost %d for %q", q.ChitonDepth1, q.Chiton, q.Question)
		}
		if q.QueryGraph <= 0 {
			t.Errorf("query_graph cost should be positive (the symbol matches its own name); got %d for %q", q.QueryGraph, q.Question)
		}
		// Synthetic functions are named FuncN/helperN — never an external import.
		if got := q.Question; got == "" {
			t.Error("empty question")
		}
	}
}

func TestRunSkipsTestSymbolsByDefault(t *testing.T) {
	dir := t.TempDir()
	if err := GenerateSynthetic(dir, 6); err != nil {
		t.Fatalf("generate: %v", err)
	}
	// A test helper that calls into every file makes it the corpus's highest-degree symbol.
	var calls strings.Builder
	for i := range 6 {
		fmt.Fprintf(&calls, "\tFunc%d(\"x\")\n", i)
	}
	helper := "package synth\n\nfunc busyTestHelper() {\n" + calls.String() + "\tbusyTestHelper2()\n}\n\nfunc busyTestHelper2() { busyTestHelper() }\n"
	if err := os.WriteFile(filepath.Join(dir, "helper_test.go"), []byte(helper), 0o644); err != nil {
		t.Fatalf("write helper: %v", err)
	}

	has := func(rep *Report) bool {
		for _, q := range rep.Queries {
			if strings.Contains(q.Question, "busyTestHelper") {
				return true
			}
		}
		return false
	}
	without, err := Run(context.Background(), "synthetic", dir, Options{MaxQueries: 20})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if has(without) {
		t.Error("a symbol defined in a _test.go file was benchmarked without IncludeTests")
	}
	with, err := Run(context.Background(), "synthetic", dir, Options{MaxQueries: 20, IncludeTests: true})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !has(with) {
		t.Error("IncludeTests did not let the test helper become a query target")
	}
	// The test file stays in the corpus either way.
	if without.Files != with.Files || without.CorpusTokens != with.CorpusTokens {
		t.Errorf("corpus changed with IncludeTests: %d/%d files, %d/%d tokens", without.Files, with.Files, without.CorpusTokens, with.CorpusTokens)
	}
}

func TestBaseIdent(t *testing.T) {
	cases := map[string]string{
		"LinkService.Submit": "Submit",
		"http.Get":           "Get",
		"node":               "node",
		"a.b.c":              "c",
		"trailingdot.":       "trailingdot.", // no identifier after the dot → unchanged
	}
	for in, want := range cases {
		if got := baseIdent(in); got != want {
			t.Errorf("baseIdent(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestEstimateTokens(t *testing.T) {
	if got := estimateTokens(400); got != 100 {
		t.Errorf("estimateTokens(400) = %d, want 100", got)
	}
}
