package metrics

import (
	"fmt"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func node(id string) model.Node {
	return model.Node{ID: id, Label: id, Kind: model.KindSymbol, Provenance: model.ProvExtracted, Confidence: 1.0}
}

func edge(src, dst string) model.Edge {
	return model.Edge{
		ID: model.NewEdgeID(src, dst, model.EdgeCalls), Src: src, Dst: dst,
		Kind: model.EdgeCalls, Provenance: model.ProvExtracted, Confidence: 1.0,
	}
}

// TestPathGraph verifies betweenness on the path A-B-C. B lies on the only path from A to C, so it must beat
// the endpoints A and C, which have zero betweenness.
func TestPathGraph(t *testing.T) {
	nodes := []model.Node{node("A"), node("B"), node("C")}
	edges := []model.Edge{edge("A", "B"), edge("B", "C")}

	m := Compute(nodes, edges)
	if m == nil {
		t.Fatal("Compute returned nil")
	}

	if m["B"].Betweenness <= m["A"].Betweenness {
		t.Errorf("B betweenness (%f) should be > A betweenness (%f)", m["B"].Betweenness, m["A"].Betweenness)
	}
	if m["B"].Betweenness <= m["C"].Betweenness {
		t.Errorf("B betweenness (%f) should be > C betweenness (%f)", m["B"].Betweenness, m["C"].Betweenness)
	}
	if m["A"].Betweenness != 0 {
		t.Errorf("A (endpoint) betweenness should be 0, got %f", m["A"].Betweenness)
	}
	if m["C"].Betweenness != 0 {
		t.Errorf("C (endpoint) betweenness should be 0, got %f", m["C"].Betweenness)
	}
}

// TestDegree verifies degree counts on a star graph (hub with 3 spokes).
func TestDegree(t *testing.T) {
	nodes := []model.Node{node("hub"), node("s1"), node("s2"), node("s3")}
	edges := []model.Edge{edge("hub", "s1"), edge("hub", "s2"), edge("hub", "s3")}

	m := Compute(nodes, edges)
	if m["hub"].Degree != 3 {
		t.Errorf("hub degree: expected 3, got %d", m["hub"].Degree)
	}
	for _, s := range []string{"s1", "s2", "s3"} {
		if m[s].Degree != 1 {
			t.Errorf("%s degree: expected 1, got %d", s, m[s].Degree)
		}
	}
}

// TestIsolatedNode verifies that isolated nodes get degree 0 and betweenness 0.
func TestIsolatedNode(t *testing.T) {
	nodes := []model.Node{node("alone")}
	m := Compute(nodes, nil)
	if m["alone"].Degree != 0 {
		t.Errorf("isolated node degree: expected 0, got %d", m["alone"].Degree)
	}
	if m["alone"].Betweenness != 0 {
		t.Errorf("isolated node betweenness: expected 0, got %f", m["alone"].Betweenness)
	}
}

// TestBetweennessNormalized verifies the normalized range is [0, 1].
func TestBetweennessNormalized(t *testing.T) {
	// Long path: A-B-C-D-E. C has maximum betweenness.
	nodes := []model.Node{node("A"), node("B"), node("C"), node("D"), node("E")}
	edges := []model.Edge{
		edge("A", "B"), edge("B", "C"), edge("C", "D"), edge("D", "E"),
	}
	m := Compute(nodes, edges)
	for id, nm := range m {
		if nm.Betweenness < 0 || nm.Betweenness > 1 {
			t.Errorf("node %s betweenness %f out of [0,1]", id, nm.Betweenness)
		}
	}
}

// TestBetweennessSampledStaysInRange forces sampling on a star whose hub lies on every shortest path, so the
// n/k extrapolation would overshoot 1 without the clamp in normalize.
func TestBetweennessSampledStaysInRange(t *testing.T) {
	const n = samplingThreshold + 1000 // > threshold → sampled path
	nodes := make([]model.Node, n)
	nodes[0] = node("hub")
	edges := make([]model.Edge, 0, n-1)
	for i := 1; i < n; i++ {
		id := fmt.Sprintf("s%d", i)
		nodes[i] = node(id)
		edges = append(edges, edge("hub", id))
	}
	m := Compute(nodes, edges)
	for id, nm := range m {
		if nm.Betweenness < 0 || nm.Betweenness > 1 {
			t.Errorf("node %s betweenness %f out of [0,1]", id, nm.Betweenness)
		}
	}
}
