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
