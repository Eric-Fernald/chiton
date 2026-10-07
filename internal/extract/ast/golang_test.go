package ast

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func TestGoExtractorCallSites(t *testing.T) {
	src := `package main

import (
	"fmt"
	"net/http"
)

func main() {
	result := fetch("http://example.com")
	fmt.Println(result)
}

func fetch(url string) string {
	resp, err := http.Get(url)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	return fmt.Sprintf("status: %d", resp.StatusCode)
}
`
	f := writeTempFile(t, "main.go", src)
	ex := &GoExtractor{}
	nodes, edges, err := ex.ExtractFile(context.Background(), f)
	if err != nil {
		t.Fatalf("ExtractFile: %v", err)
	}
	logNodesEdges(t, nodes, edges)

	assertLabel(t, nodes, "main")
	assertLabel(t, nodes, "fetch")
	assertLabel(t, nodes, "http.Get")
	assertEdgeKind(t, edges, "CALLS")
	assertEdgeKind(t, edges, "IMPORTS")

	labels := make(map[string]string, len(nodes))
	for _, n := range nodes {
		labels[n.ID] = n.Label
	}

	var foundMainCallsFetch bool
	for _, e := range edges {
		if e.Kind == model.EdgeCalls && labels[e.Src] == "main" && labels[e.Dst] == "fetch" {
			foundMainCallsFetch = true
		}
	}
	if !foundMainCallsFetch {
		t.Error("expected CALLS edge main → fetch but none found")
	}
}

func TestGoExtractorSameNamedMethods(t *testing.T) {
	src := `package p

type A struct{}

func (A) String() string { return helper() }

type B struct{}

func (B) String() string { return "" }

func Close() {}

func (B) Close() {}

func run() { Close() }

func helper() string { return "" }
`
	f := writeTempFile(t, "p.go", src)
	nodes, edges, err := (&GoExtractor{}).ExtractFile(context.Background(), f)
	if err != nil {
		t.Fatalf("ExtractFile: %v", err)
	}
	labels := make(map[string]string, len(nodes))
	for _, n := range nodes {
		labels[n.ID] = n.Label
	}
	calls := map[string]bool{}
	for _, e := range edges {
		if e.Kind == model.EdgeCalls {
			calls[labels[e.Src]+"->"+labels[e.Dst]] = true
		}
	}
	for _, want := range []string{"A.String->helper", "run->Close"} {
		if !calls[want] {
			t.Errorf("missing CALLS %s; got %v", want, calls)
		}
	}
	for _, bad := range []string{"B.String->helper", "run->B.Close"} {
		if calls[bad] {
			t.Errorf("unexpected CALLS %s", bad)
		}
	}
}

// TestGoExtractorDedupes checks repeated calls, a doubly imported package, and duplicate init funcs yield each
// node and edge once.
func TestGoExtractorDedupes(t *testing.T) {
	src := `package p

import (
	"fmt"
	f "fmt"
)

func a() { b(); b(); fmt.Println(); f.Println() }

func b() {}

func init() {}

func init() {}
`
	nodes, edges, err := (&GoExtractor{}).ExtractFile(context.Background(), writeTempFile(t, "p.go", src))
	if err != nil {
		t.Fatalf("ExtractFile: %v", err)
	}
	seen := map[string]bool{}
	for _, n := range nodes {
		if seen[n.ID] {
			t.Errorf("duplicate node %q", n.Label)
		}
		seen[n.ID] = true
	}
	seen = map[string]bool{}
	for _, e := range edges {
		if seen[e.ID] {
			t.Errorf("duplicate %s edge %s -> %s", e.Kind, e.Src, e.Dst)
		}
		seen[e.ID] = true
	}
}

func TestGoExtractorResolvesSiblingFileCalls(t *testing.T) {
	root := t.TempDir()
	write := func(rel, body string) string {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	write("go.mod", "module example.com/mod\n\ngo 1.25\n")
	write("lib/helper.go", "package lib\n\nfunc helper() int { return 1 }\n")
	caller := write("lib/caller.go", `package lib

import "example.com/mod/other"

func Run(local func() int) int {
	n := len("x")
	other := 3
	_ = other
	return helper() + local() + n
}

func Use() { other.Do() }
`)

	nodes, edges, err := (&GoExtractor{}).ExtractFile(context.Background(), caller)
	if err != nil {
		t.Fatal(err)
	}
	var file model.Node
	byID := map[string]model.Node{}
	for _, n := range nodes {
		byID[n.ID] = n
		if n.Kind == model.KindFile {
			file = n
		}
	}
	if got := model.AttrString(file.Attrs, model.AttrImportPath); got != "example.com/mod/lib" {
		t.Errorf("file import_path = %q, want example.com/mod/lib", got)
	}
	if got := model.AttrString(file.Attrs, model.AttrModule); got != "example.com/mod" {
		t.Errorf("file module = %q, want example.com/mod", got)
	}

	calls := map[string]bool{}
	for _, e := range edges {
		if e.Kind == model.EdgeCalls {
			calls[byID[e.Src].Label+"->"+e.Dst] = true
		}
	}
	// helper lives in a sibling file, so the call points at the same reference other packages would use.
	if !calls["Run->"+model.PkgFuncRefID("example.com/mod/lib", "helper")] {
		t.Errorf("Run's call to helper in a sibling file has no CALLS edge to its pkg:// reference; calls: %v", calls)
	}
	if !calls["Use->"+model.PkgFuncRefID("example.com/mod/other", "Do")] {
		t.Errorf("Use's call to other.Do has no CALLS edge; calls: %v", calls)
	}
	// len is a builtin and local is a parameter: neither is a package function.
	for _, name := range []string{"len", "local"} {
		if calls["Run->"+model.PkgFuncRefID("example.com/mod/lib", name)] {
			t.Errorf("Run's call to %s became a package-function edge", name)
		}
	}
}
