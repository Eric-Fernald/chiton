package architecture

import (
	"context"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func file(id, label string) model.Node {
	return model.Node{ID: id, Kind: model.KindFile, Label: label, SourceURI: "file://" + label}
}

func sym(id, label, langKind string) model.Node {
	return model.Node{
		ID:    id,
		Kind:  model.KindSymbol,
		Label: label,
		Attrs: map[string]any{"lang_kind": langKind},
	}
}

func edge(src, dst string, kind model.EdgeKind) model.Edge {
	return model.Edge{
		ID:   model.NewEdgeID(src, dst, kind),
		Src:  src,
		Dst:  dst,
		Kind: kind,
	}
}

func TestCircularImport_DetectsTwoNodeCycle(t *testing.T) {
	a, b := file("A", "a.go"), file("B", "b.go")
	tg := analyze.GraphTarget{
		Nodes: []model.Node{a, b},
		Edges: []model.Edge{
			edge("A", "B", model.EdgeImports),
			edge("B", "A", model.EdgeImports),
		},
	}
	got := (CircularImport{}).MatchGraph(context.Background(), tg)
	if len(got) != 1 {
		t.Fatalf("expected 1 cycle, got %d (%+v)", len(got), got)
	}
}

func TestCircularImport_IgnoresExternalPackages(t *testing.T) {
	a := file("A", "a.go")
	ext := model.Node{
		ID:    "EXT",
		Kind:  model.KindSymbol,
		Label: "fmt",
		Attrs: map[string]any{"external": true},
	}
	tg := analyze.GraphTarget{
		Nodes: []model.Node{a, ext},
		Edges: []model.Edge{edge("A", "EXT", model.EdgeImports)},
	}
	if got := (CircularImport{}).MatchGraph(context.Background(), tg); len(got) != 0 {
		t.Fatalf("expected no findings, got %d", len(got))
	}
}

func TestCircularImport_NoCycle(t *testing.T) {
	a, b, c := file("A", "a.go"), file("B", "b.go"), file("C", "c.go")
	tg := analyze.GraphTarget{
		Nodes: []model.Node{a, b, c},
		Edges: []model.Edge{
			edge("A", "B", model.EdgeImports),
			edge("B", "C", model.EdgeImports),
		},
	}
	if got := (CircularImport{}).MatchGraph(context.Background(), tg); len(got) != 0 {
		t.Fatalf("expected no findings, got %d (%+v)", len(got), got)
	}
}

func TestGodModule(t *testing.T) {
	big := file("F", "big.go")
	small := file("F2", "small.go")
	nodes := []model.Node{big, small}
	var edges []model.Edge
	for i := 0; i <= GodModuleSymbolThreshold; i++ {
		nodes = append(nodes, sym("sB"+itoa(i), "x", "func"))
		edges = append(edges, edge("F", "sB"+itoa(i), model.EdgeContains))
	}
	// small file: only a few symbols
	for i := 0; i < 3; i++ {
		nodes = append(nodes, sym("sS"+itoa(i), "y", "func"))
		edges = append(edges, edge("F2", "sS"+itoa(i), model.EdgeContains))
	}

	got := (GodModule{}).MatchGraph(context.Background(), analyze.GraphTarget{Nodes: nodes, Edges: edges})
	if len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d", len(got))
	}
	if got[0].TargetNodeID != "F" {
		t.Fatalf("expected finding on F, got %q", got[0].TargetNodeID)
	}
}

func TestDeadSymbol(t *testing.T) {
	// "used" has incoming CALLS; "deadFn" does not; "Exported" is exported;
	// "T" is a type and should be skipped.
	used := sym("used", "used", "func")
	dead := sym("dead", "deadFn", "func")
	exported := sym("Exp", "Exported", "func")
	typ := sym("T1", "MyType", "type")
	used.SourceURI = "file://app.go"
	dead.SourceURI = "file://app.go"
	exported.SourceURI = "file://app.go"
	typ.SourceURI = "file://app.go"

	tg := analyze.GraphTarget{
		Nodes: []model.Node{used, dead, exported, typ},
		Edges: []model.Edge{
			edge("caller", "used", model.EdgeCalls),
		},
	}
	got := (DeadSymbol{}).MatchGraph(context.Background(), tg)
	if len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d (%+v)", len(got), got)
	}
	if got[0].TargetNodeID != "dead" {
		t.Fatalf("expected finding on 'dead', got %q", got[0].TargetNodeID)
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	digits := "0123456789"
	out := ""
	for i > 0 {
		out = string(digits[i%10]) + out
		i /= 10
	}
	return out
}
