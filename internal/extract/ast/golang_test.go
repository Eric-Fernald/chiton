package ast

import (
	"context"
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
