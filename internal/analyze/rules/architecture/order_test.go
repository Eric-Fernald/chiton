package architecture

import (
	"context"
	"reflect"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// targets lists each finding's TargetNodeID in order.
func targets(fs []analyze.Finding) []string {
	out := make([]string, len(fs))
	for i, f := range fs {
		out[i] = f.TargetNodeID
	}
	return out
}

// TestGraphRulesReportInStableOrder runs each map-driven rule repeatedly; Go randomises map iteration, so
// any order leak shows up as a mismatch.
func TestGraphRulesReportInStableOrder(t *testing.T) {
	var god analyze.GraphTarget
	for _, f := range []string{"F5", "F3", "F9", "F1", "F7"} {
		god.Nodes = append(god.Nodes, file(f, f+".go"))
		for i := 0; i <= GodModuleSymbolThreshold; i++ {
			id := f + "s" + itoa(i)
			god.Nodes = append(god.Nodes, sym(id, "x", "func"))
			god.Edges = append(god.Edges, edge(f, id, model.EdgeContains))
		}
	}
	var cyc analyze.GraphTarget
	for _, p := range [][2]string{{"P", "Q"}, {"C", "D"}, {"X", "Y"}, {"A", "B"}, {"M", "N"}} {
		cyc.Nodes = append(cyc.Nodes, file(p[0], p[0]+".ts"), file(p[1], p[1]+".ts"))
		cyc.Edges = append(cyc.Edges, edge(p[0], p[1], model.EdgeImports), edge(p[1], p[0], model.EdgeImports))
	}
	cases := []struct {
		name string
		run  func() []analyze.Finding
		want []string
	}{
		{"god-module", func() []analyze.Finding { return GodModule{}.MatchGraph(context.Background(), god) },
			[]string{"F1", "F3", "F5", "F7", "F9"}},
		{"circular-import", func() []analyze.Finding { return CircularImport{}.MatchGraph(context.Background(), cyc) },
			[]string{"A", "C", "M", "P", "X"}},
	}
	for _, tc := range cases {
		for range 20 {
			if got := targets(tc.run()); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("%s order = %v, want %v", tc.name, got, tc.want)
			}
		}
	}
}
