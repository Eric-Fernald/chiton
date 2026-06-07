package bench

import (
	"context"
	"testing"
)

func TestRunOnSyntheticCorpus(t *testing.T) {
	dir := t.TempDir()
	if err := GenerateSynthetic(dir, 12); err != nil {
		t.Fatalf("generate: %v", err)
	}
	rep, err := Run(context.Background(), "synthetic", dir, 4)
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
		// Synthetic functions are named FuncN/helperN — never an external import.
		if got := q.Question; got == "" {
			t.Error("empty question")
		}
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
