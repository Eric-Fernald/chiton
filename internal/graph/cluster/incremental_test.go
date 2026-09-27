package cluster

import (
	"math/rand"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/metrics"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// validatePartition asserts community IDs form a contiguous 0-based range and
// every node has an entry.
func validatePartition(t *testing.T, res *Results, nodes []model.Node) {
	t.Helper()
	if res.NodeCount != len(nodes) {
		t.Errorf("NodeCount = %d, want %d", res.NodeCount, len(nodes))
	}
	if len(res.Nodes) != len(nodes) {
		t.Errorf("len(Nodes) = %d, want %d", len(res.Nodes), len(nodes))
	}
	seen := map[int]bool{}
	for _, n := range nodes {
		nr, ok := res.Nodes[n.ID]
		if !ok {
			t.Errorf("missing node %s in results", n.ID)
			continue
		}
		seen[nr.CommunityID] = true
	}
	for c := range seen {
		if c < 0 || c >= len(seen) {
			t.Errorf("community ID %d outside contiguous range [0,%d)", c, len(seen))
		}
	}
	if res.CommunityCount != len(seen) {
		t.Errorf("CommunityCount = %d, want %d", res.CommunityCount, len(seen))
	}
}

// sameComm reports whether two node IDs share a community in res.
func sameComm(res *Results, a, b string) bool {
	return res.Nodes[a].CommunityID == res.Nodes[b].CommunityID
}

func twoCliques() ([]model.Node, []model.Edge) {
	nodes := []model.Node{
		makeNode("A", "A"), makeNode("B", "B"), makeNode("C", "C"),
		makeNode("D", "D"), makeNode("E", "E"), makeNode("F", "F"),
	}
	edges := []model.Edge{
		makeEdge("A", "B"), makeEdge("B", "C"), makeEdge("A", "C"),
		makeEdge("D", "E"), makeEdge("E", "F"), makeEdge("D", "F"),
	}
	return nodes, edges
}

// TestIncrementalColdStart: prev nil → behaves like a full clustering.
func TestIncrementalColdStart(t *testing.T) {
	nodes, edges := twoCliques()
	res := RunIncremental(nil, nodes, edges, map[string]bool{"A": true}, 1.0, rand.New(rand.NewSource(1)))
	validatePartition(t, res, nodes)
	if !sameComm(res, "A", "B") || !sameComm(res, "B", "C") {
		t.Error("clique 1 should share a community on cold start")
	}
	if !sameComm(res, "D", "E") || !sameComm(res, "E", "F") {
		t.Error("clique 2 should share a community on cold start")
	}
	if sameComm(res, "A", "D") {
		t.Error("disjoint cliques should be in different communities")
	}
	// Cold start computes exact betweenness; bridge-free cliques have 0, but the
	// degree must be populated.
	if res.Nodes["A"].Degree != 2 {
		t.Errorf("degree(A) = %d, want 2", res.Nodes["A"].Degree)
	}
}

// TestIncrementalFreezesUntouched: nodes outside the 2-hop neighborhood keep
// their grouping; betweenness is carried forward.
func TestIncrementalFreezesUntouched(t *testing.T) {
	nodes, edges := twoCliques()
	// Seed prev with a full pass, then stamp a distinctive betweenness on E.
	prev := RunIncremental(nil, nodes, edges, map[string]bool{"A": true}, 1.0, rand.New(rand.NewSource(2)))
	e := prev.Nodes["E"]
	e.Betweenness = 0.42
	prev.Nodes["E"] = e

	// Touch only clique 1. Clique 2 (D,E,F) is >2 hops away (disconnected), so
	// it must be frozen.
	res := RunIncremental(prev, nodes, edges, map[string]bool{"A": true}, 1.0, rand.New(rand.NewSource(3)))
	validatePartition(t, res, nodes)

	if !sameComm(res, "D", "E") || !sameComm(res, "E", "F") {
		t.Error("untouched clique 2 should keep its grouping")
	}
	if !sameComm(res, "A", "B") || !sameComm(res, "B", "C") {
		t.Error("re-clustered clique 1 should still be one community")
	}
	if got := res.Nodes["E"].Betweenness; got != 0.42 {
		t.Errorf("betweenness for untouched node E = %v, want carried-forward 0.42", got)
	}
}

// TestIncrementalDeletion: a node absent from the current graph drops out of the
// results and the partition stays valid.
func TestIncrementalDeletion(t *testing.T) {
	nodes, edges := twoCliques()
	prev := RunIncremental(nil, nodes, edges, map[string]bool{"A": true}, 1.0, rand.New(rand.NewSource(4)))

	// Remove F and its edges; "touch" F (as a delete would) and its old neighbors.
	var curNodes []model.Node
	for _, n := range nodes {
		if n.ID != "F" {
			curNodes = append(curNodes, n)
		}
	}
	var curEdges []model.Edge
	for _, ed := range edges {
		if ed.Src != "F" && ed.Dst != "F" {
			curEdges = append(curEdges, ed)
		}
	}
	res := RunIncremental(prev, curNodes, curEdges, map[string]bool{"F": true, "D": true, "E": true}, 1.0, rand.New(rand.NewSource(5)))
	validatePartition(t, res, curNodes)
	if _, ok := res.Nodes["F"]; ok {
		t.Error("deleted node F should not appear in results")
	}
	if !sameComm(res, "D", "E") {
		t.Error("D and E should remain together after F's removal")
	}
}

// TestIncrementalNewNode: a brand-new node not present in prev gets a community.
func TestIncrementalNewNode(t *testing.T) {
	nodes, edges := twoCliques()
	prev := RunIncremental(nil, nodes, edges, map[string]bool{"A": true}, 1.0, rand.New(rand.NewSource(6)))

	curNodes := append([]model.Node{}, nodes...)
	curNodes = append(curNodes, makeNode("G", "G")) // isolated new node
	res := RunIncremental(prev, curNodes, edges, map[string]bool{"G": true}, 1.0, rand.New(rand.NewSource(7)))
	validatePartition(t, res, curNodes)
	if _, ok := res.Nodes["G"]; !ok {
		t.Fatal("new node G missing from results")
	}
	if res.Nodes["G"].Betweenness != 0 {
		t.Errorf("new node G betweenness = %v, want 0", res.Nodes["G"].Betweenness)
	}
	// G is isolated, so it must be in its own community, distinct from both cliques.
	if sameComm(res, "G", "A") || sameComm(res, "G", "D") {
		t.Error("isolated new node G should be its own community")
	}
}

// TestFullMatchesSeparatePasses pins Full to Run plus metrics.Compute, the passes `chiton cluster` combines.
func TestFullMatchesSeparatePasses(t *testing.T) {
	nodes, edges := twoCliques()
	nodes = append(nodes, makeNode("A", "A2")) // a duplicate ID, so the later entry wins
	edges = append(edges, makeEdge("A", "A"), makeEdge("B", "A"), makeEdge("A", "gone"))

	res := Full(nodes, edges, 1.0, rand.New(rand.NewSource(3)))
	comm := Run(nodes, edges, 1.0, rand.New(rand.NewSource(3)))
	met := metrics.Compute(nodes, edges)
	if res.NodeCount != len(nodes) || res.CommunityCount != 2 {
		t.Errorf("NodeCount, CommunityCount = %d, %d; want %d, 2", res.NodeCount, res.CommunityCount, len(nodes))
	}
	for _, n := range nodes {
		want := NodeResult{Label: n.Label, SourceURI: n.SourceURI, CommunityID: comm[n.ID],
			Betweenness: met[n.ID].Betweenness, Degree: met[n.ID].Degree}
		if n.ID == "A" {
			want.Label = "A2"
		}
		if got := res.Nodes[n.ID]; got != want {
			t.Errorf("Full node %s = %+v, want %+v", n.ID, got, want)
		}
	}
}

// TestNeighborhood checks 2-hop expansion over a path graph.
func TestNeighborhood(t *testing.T) {
	// Path: a-b-c-d-e
	adj := map[string][]string{
		"a": {"b"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"c", "e"}, "e": {"d"},
	}
	hood := neighborhood(adj, map[string]bool{"a": true}, 2)
	for _, want := range []string{"a", "b", "c"} {
		if !hood[want] {
			t.Errorf("expected %q in 2-hop neighborhood of a", want)
		}
	}
	if hood["d"] || hood["e"] {
		t.Errorf("d/e are >2 hops from a and should be excluded: %v", hood)
	}
}
