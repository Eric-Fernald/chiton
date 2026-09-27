package cluster

import (
	"fmt"
	"maps"
	"math/rand"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// makeEdge is a helper that builds a model.Edge between two node IDs.
func makeEdge(src, dst string) model.Edge {
	return model.Edge{
		ID:  model.NewEdgeID(src, dst, model.EdgeCalls),
		Src: src, Dst: dst,
		Kind:       model.EdgeCalls,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
	}
}

// makeNode builds a minimal model.Node.
func makeNode(id, label string) model.Node {
	return model.Node{
		ID: id, Label: label,
		Kind:       model.KindSymbol,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
	}
}

// TestTwoCliques verifies that two disjoint cliques are placed in distinct communities.
// Graph: clique {A,B,C} and clique {D,E,F} with no inter-clique edges.
func TestTwoCliques(t *testing.T) {
	nodes := []model.Node{
		makeNode("A", "A"), makeNode("B", "B"), makeNode("C", "C"),
		makeNode("D", "D"), makeNode("E", "E"), makeNode("F", "F"),
	}
	edges := []model.Edge{
		makeEdge("A", "B"), makeEdge("B", "C"), makeEdge("A", "C"), // clique 1
		makeEdge("D", "E"), makeEdge("E", "F"), makeEdge("D", "F"), // clique 2
	}
	rng := rand.New(rand.NewSource(0))
	comm := Run(nodes, edges, 1.0, rng)

	if len(comm) != len(nodes) {
		t.Fatalf("expected %d community entries, got %d", len(nodes), len(comm))
	}

	// Clique 1 should all share the same community; clique 2 should share a different one.
	c1 := comm["A"]
	for _, id := range []string{"B", "C"} {
		if comm[id] != c1 {
			t.Errorf("node %s: expected community %d (same as A), got %d", id, c1, comm[id])
		}
	}
	c2 := comm["D"]
	for _, id := range []string{"E", "F"} {
		if comm[id] != c2 {
			t.Errorf("node %s: expected community %d (same as D), got %d", id, c2, comm[id])
		}
	}
	if c1 == c2 {
		t.Errorf("clique 1 and clique 2 ended up in the same community (%d)", c1)
	}
}

// TestSingleNode verifies graceful handling of a single node.
func TestSingleNode(t *testing.T) {
	nodes := []model.Node{makeNode("X", "X")}
	comm := Run(nodes, nil, 1.0, nil)
	if len(comm) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(comm))
	}
	if comm["X"] != 0 {
		t.Errorf("expected community 0, got %d", comm["X"])
	}
}

// TestEmptyGraph verifies that an empty input returns nil.
func TestEmptyGraph(t *testing.T) {
	if comm := Run(nil, nil, 1.0, nil); comm != nil {
		t.Errorf("expected nil for empty input, got %v", comm)
	}
}

// TestDisconnectedNodes verifies that n isolated nodes yield n communities.
func TestDisconnectedNodes(t *testing.T) {
	nodes := []model.Node{
		makeNode("a", "a"), makeNode("b", "b"), makeNode("c", "c"),
	}
	comm := Run(nodes, nil, 1.0, nil)
	seen := map[int]bool{}
	for _, c := range comm {
		seen[c] = true
	}
	if len(seen) != 3 {
		t.Errorf("expected 3 distinct communities for 3 isolated nodes, got %d", len(seen))
	}
}

// TestRunDeterministic verifies a fixed seed yields identical partitions even when many moves tie on gain.
func TestRunDeterministic(t *testing.T) {
	const n = 300
	var nodes []model.Node
	var edges []model.Edge
	for i := range n {
		id := fmt.Sprint("n", i)
		nodes = append(nodes, makeNode(id, id))
		edges = append(edges, makeEdge(id, fmt.Sprint("n", (i+1)%n)), makeEdge(id, fmt.Sprint("n", (i*7+3)%n)))
	}
	want := Run(nodes, edges, 2.0, rand.New(rand.NewSource(1)))
	for range 5 {
		if got := Run(nodes, edges, 2.0, rand.New(rand.NewSource(1))); !maps.Equal(got, want) {
			t.Fatal("Run gave different partitions for the same seed")
		}
	}
}

// TestCompactIDs verifies that community IDs are always a contiguous 0-based range.
func TestCompactIDs(t *testing.T) {
	nodes := []model.Node{
		makeNode("p", "p"), makeNode("q", "q"), makeNode("r", "r"), makeNode("s", "s"),
	}
	edges := []model.Edge{makeEdge("p", "q"), makeEdge("r", "s")}
	rng := rand.New(rand.NewSource(7))
	comm := Run(nodes, edges, 1.0, rng)

	seen := map[int]bool{}
	for _, c := range comm {
		seen[c] = true
	}
	// IDs should be 0 and 1 (two components).
	for c := range seen {
		if c < 0 || c >= len(seen) {
			t.Errorf("community ID %d out of range [0, %d)", c, len(seen))
		}
	}
}
