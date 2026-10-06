package cluster

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// stub builds an external package node, as the extractors emit for an import.
func stub(id string) model.Node {
	n := makeNode(id, id)
	n.Attrs = map[string]any{"external": true}
	return n
}

// TestBuildRegion checks the hop limit, that an external stub is a leaf, and that an absent seed is Removed.
func TestBuildRegion(t *testing.T) {
	// Path a-b-c-d, plus a imports hub P, which 50 other files also import.
	nodes := []model.Node{makeNode("a", "a"), makeNode("b", "b"), makeNode("c", "c"), makeNode("d", "d"), stub("P")}
	edges := []model.Edge{makeEdge("a", "b"), makeEdge("b", "c"), makeEdge("c", "d"), makeEdge("a", "P")}
	for i := range 50 {
		id := fmt.Sprint("f", i)
		nodes = append(nodes, makeNode(id, id))
		edges = append(edges, makeEdge(id, "P"))
	}
	r, err := BuildRegion([]string{"a", "gone"}, 2, memNeighbors(nodes, edges))
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, n := range r.Nodes {
		got = append(got, n.ID)
	}
	if want := []string{"a", "b", "P", "c"}; !reflect.DeepEqual(got, want) {
		t.Errorf("region nodes = %v, want %v (no importer reached through P, d beyond 2 hops)", got, want)
	}
	if !r.Removed["gone"] || len(r.Removed) != 1 {
		t.Errorf("Removed = %v, want only the absent seed", r.Removed)
	}
	if r.Degree["a"] != 2 || r.Degree["b"] != 2 {
		t.Errorf("Degree = %v, want a:2 b:2 for the expanded nodes", r.Degree)
	}
	if _, ok := r.Degree["P"]; ok {
		t.Error("stub P is a leaf; its adjacency must not be read")
	}
	if len(r.Edges) != 3 { // a-b, a-P, b-c
		t.Errorf("region edges = %d, want 3", len(r.Edges))
	}
}

// TestApplyRegionPinsExistingStub checks a hub stub already in prev keeps its community (and its partners
// outside the region keep theirs) while the region is re-clustered and removed nodes drop out.
func TestApplyRegionPinsExistingStub(t *testing.T) {
	nodes, edges := twoCliques()
	nodes = append(nodes, stub("P"), makeNode("X", "X"))
	edges = append(edges, makeEdge("D", "P"), makeEdge("E", "P"), makeEdge("A", "P"))
	prev := Full(nodes, edges, 1.0, rand.New(rand.NewSource(1)))
	if !sameComm(prev, "P", "D") {
		t.Fatalf("setup: P should start with clique 2, got %+v", prev.Nodes)
	}

	r, err := BuildRegion([]string{"A", "X"}, 2, memNeighbors(nodes[:len(nodes)-1], edges)) // X was deleted
	if err != nil {
		t.Fatal(err)
	}
	res := ApplyRegion(prev, r, 1.0, rand.New(rand.NewSource(2)))
	if !sameComm(res, "P", "D") || !sameComm(res, "D", "E") {
		t.Error("pinned stub P and its untouched partners should keep one community")
	}
	if !sameComm(res, "A", "B") || sameComm(res, "A", "D") {
		t.Error("clique 1 should be re-clustered together and apart from clique 2")
	}
	if _, ok := res.Nodes["X"]; ok || res.NodeCount != len(nodes)-1 {
		t.Errorf("removed X should drop out: NodeCount=%d, X present=%v", res.NodeCount, ok)
	}
	validatePartition(t, res, nodes[:len(nodes)-1])
}

// TestSidecarIsCompactAndRoundTrips checks WriteSidecar emits unindented JSON that LoadSidecar reads back.
func TestSidecarIsCompactAndRoundTrips(t *testing.T) {
	nodes, edges := twoCliques()
	want := Full(nodes, edges, 1.0, nil)
	path := filepath.Join(t.TempDir(), "cluster.json")
	if err := WriteSidecar(path, want, nil); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte("\n ")) {
		t.Errorf("sidecar is indented: %.80s", raw)
	}
	got, err := LoadSidecar(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !got.GeneratedAt.Equal(want.GeneratedAt) || !reflect.DeepEqual(got.Nodes, want.Nodes) || got.CommunityCount != want.CommunityCount {
		t.Errorf("round trip changed the results:\n got %+v\nwant %+v", got, want)
	}
}
