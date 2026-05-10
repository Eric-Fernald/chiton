package ast

import (
	"context"
	"testing"
)

func TestExtractFileCallSites(t *testing.T) {
	ex := &GoExtractor{}
	nodes, edges, err := ex.ExtractFile(context.Background(),
		`C:\Programming Repos\Personal-Coding-Projects\Projects\Go_Projects\fetchall\fetchall.go`)
	if err != nil {
		t.Fatalf("ExtractFile: %v", err)
	}

	t.Logf("total nodes: %d, total edges: %d", len(nodes), len(edges))
	for _, n := range nodes {
		t.Logf("  node: kind=%s label=%q uri=%s", n.Kind, n.Label, n.SourceURI)
	}
	for _, e := range edges {
		t.Logf("  edge: kind=%s src=%s dst=%s", e.Kind, e.Src, e.Dst)
	}

	var foundHTTPGet bool
	for _, n := range nodes {
		if n.Label == "http.Get" {
			foundHTTPGet = true
		}
	}
	if !foundHTTPGet {
		t.Error("expected node with label \"http.Get\" but none found")
	}
}
