package report

import (
	"bytes"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func sym(id, label string) model.Node {
	return model.Node{ID: id, Kind: model.KindSymbol, Label: label, SourceURI: "file://x.go"}
}

func TestBuildViewCapsAndDropsDanglingEdges(t *testing.T) {
	nodes := []model.Node{sym("a", "A"), sym("b", "B"), sym("c", "C")}
	edges := []model.Edge{
		{ID: "e1", Src: "a", Dst: "b", Kind: model.EdgeCalls}, // a,b each degree 1
		{ID: "e2", Src: "a", Dst: "c", Kind: model.EdgeCalls}, // a degree 2, c degree 1
	}
	// Cap to 2 nodes: a (deg 2) and b/c (deg 1, tie broken by ID → b).
	views, kept := buildView(nodes, edges, nil, 2)
	if len(views) != 2 {
		t.Fatalf("views: got %d, want 2", len(views))
	}
	if views[0].ID != "a" {
		t.Errorf("highest-degree node should sort first: got %q", views[0].ID)
	}
	// Only the a→b edge survives; a→c dangles because c was dropped.
	if len(kept) != 1 || kept[0].ID != "e1" {
		t.Fatalf("kept edges: got %+v, want only e1 (e2 dangles to dropped node)", kept)
	}
}

func TestGenerateMermaid(t *testing.T) {
	nodes := []model.Node{sym("a", "A"), sym("b", `B"quote`)}
	edges := []model.Edge{{ID: "e1", Src: "a", Dst: "b", Kind: model.EdgeCalls}}
	var buf bytes.Buffer
	if err := GenerateMermaid(&buf, nodes, edges, nil, DefaultVisual); err != nil {
		t.Fatalf("mermaid: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "flowchart LR") {
		t.Error("missing flowchart header")
	}
	if !strings.Contains(out, "-->|calls|") {
		t.Error("missing lowercased edge label")
	}
	if strings.Contains(out, `B"quote`) {
		t.Error("unescaped double quote leaked into Mermaid label")
	}
	if !strings.Contains(out, "&quot;") {
		t.Error("double quote should be escaped to &quot;")
	}
}

func TestGenerateHTMLNoUntrustedInnerHTMLSinks(t *testing.T) {
	// Regression: node label/uri/kind are ingested, untrusted strings. The
	// legend and detail panel must build the DOM via textContent/createTextNode,
	// never innerHTML — otherwise a label like "<img src=x onerror=…>" runs.
	evil := []model.Node{sym("a", "<img src=x onerror=alert(1)>"), sym("b", "B")}
	edges := []model.Edge{{ID: "e1", Src: "a", Dst: "b", Kind: model.EdgeCalls}}
	var buf bytes.Buffer
	if err := GenerateHTML(&buf, evil, edges, nil, DefaultVisual); err != nil {
		t.Fatalf("html: %v", err)
	}
	out := buf.String()
	for _, sink := range []string{"detail.innerHTML", "row.innerHTML"} {
		if strings.Contains(out, sink) {
			t.Errorf("found unsafe innerHTML sink %q — untrusted node data must use textContent/createTextNode", sink)
		}
	}
	if !strings.Contains(out, "createTextNode") {
		t.Error("expected safe DOM construction (createTextNode) in the rendered page")
	}
}

func TestGenerateHTMLEscapesScriptClose(t *testing.T) {
	edges := []model.Edge{{ID: "e1", Src: "a", Dst: "b", Kind: model.EdgeCalls}}

	// Baseline: how many </script> tags the template emits with benign data.
	var base bytes.Buffer
	if err := GenerateHTML(&base, []model.Node{sym("a", "A"), sym("b", "B")}, edges, nil, DefaultVisual); err != nil {
		t.Fatalf("html (benign): %v", err)
	}
	baseCount := strings.Count(base.String(), "</script>")

	// A node label that, injected verbatim, would close the <script> block early.
	evil := []model.Node{sym("a", "</script><b>pwn"), sym("b", "B")}
	var buf bytes.Buffer
	if err := GenerateHTML(&buf, evil, edges, nil, DefaultVisual); err != nil {
		t.Fatalf("html (evil): %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "rawNodes") {
		t.Error("missing embedded node data")
	}
	if got := strings.Count(out, "</script>"); got != baseCount {
		t.Errorf("malicious label introduced a new </script> (%d vs baseline %d) — script-break escape failed", got, baseCount)
	}
	if strings.Contains(out, "</script><b>pwn") {
		t.Error("malicious label appears verbatim in output — not escaped")
	}
}
