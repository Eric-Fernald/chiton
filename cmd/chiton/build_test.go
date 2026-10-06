package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/config"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// TestBuildReplacesStaleSymbols checks a rebuild drops renamed symbols and deleted files, skips ignored
// directories, and keeps the summary line's shape.
func TestBuildReplacesStaleSymbols(t *testing.T) {
	proj := t.TempDir()
	t.Setenv(config.ProjectDirEnv, proj)
	t.Setenv(plaintextOKEnv, "1")
	src := filepath.Join(proj, "src")
	write := func(rel, body string) {
		t.Helper()
		p := filepath.Join(src, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	build := func() string {
		t.Helper()
		var out bytes.Buffer
		cmd := rootCmd()
		cmd.SetOut(&out)
		cmd.SetErr(io.Discard)
		cmd.SetArgs([]string{"build", src})
		if err := cmd.ExecuteContext(context.Background()); err != nil {
			t.Fatalf("build: %v", err)
		}
		return out.String()
	}

	write("a.go", "package src\nfunc Hello() {}\n")
	write("gone.go", "package src\nfunc Gone() {}\n")
	write(filepath.Join("node_modules", "dep", "x.js"), "function dep() {}\n")
	build()

	write("a.go", "package src\nfunc Greet() {}\n")
	if err := os.Remove(filepath.Join(src, "gone.go")); err != nil {
		t.Fatal(err)
	}
	if out := build(); !strings.Contains(out, "build: done - 1 files scanned, 0 skipped (unchanged)") {
		t.Errorf("second build output = %q, want 1 file scanned", out)
	}

	cfg, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	gs, err := openGraphStore(cfg.GraphPath, true)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	nodes, err := gs.AllNodes(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	labels := map[string]bool{}
	for _, n := range nodes {
		labels[n.Label] = true
	}
	for _, stale := range []string{"Hello", "Gone", "dep"} {
		if labels[stale] {
			t.Errorf("stale or ignored symbol %q is still in the graph", stale)
		}
	}
	if !labels["Greet"] {
		t.Error("Greet missing after rebuild")
	}
}

// TestBuildPrunesDeletedFilesFindings checks build's default --prune drops a vanished file's findings too.
func TestBuildPrunesDeletedFilesFindings(t *testing.T) {
	proj := t.TempDir()
	t.Setenv(config.ProjectDirEnv, proj)
	t.Setenv(plaintextOKEnv, "1")
	gone := filepath.Join(proj, "src", "gone.go")
	if err := os.MkdirAll(filepath.Dir(gone), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(gone, []byte("package src\nfunc Gone() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	build := func() {
		t.Helper()
		cmd := rootCmd()
		cmd.SetOut(io.Discard)
		cmd.SetErr(io.Discard)
		cmd.SetArgs([]string{"build", filepath.Dir(gone)})
		if err := cmd.ExecuteContext(context.Background()); err != nil {
			t.Fatalf("build: %v", err)
		}
	}
	build()

	cfg, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	finding := model.Node{ID: "finding-gone", Kind: model.KindFinding, Label: "stale finding", Provenance: model.ProvExtracted,
		Confidence: 1, Attrs: map[string]any{"target_uri": model.FileURI(gone)}}
	gs, err := openGraphStore(cfg.GraphPath, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertNodes(context.Background(), []model.Node{finding}); err != nil {
		t.Fatal(err)
	}
	gs.Close()

	if err := os.Remove(gone); err != nil {
		t.Fatal(err)
	}
	build()

	gs, err = openGraphStore(cfg.GraphPath, true)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	if n, err := gs.GetNode(context.Background(), finding.ID); err != nil {
		t.Fatal(err)
	} else if n != nil {
		t.Errorf("finding for deleted %s survived build: %+v", gone, n)
	}
}
