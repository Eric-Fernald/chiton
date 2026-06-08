package report

import (
	"bytes"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func sampleDashboardData() DashboardData {
	return DashboardData{
		Title: "demo",
		Nodes: []model.Node{sym("a", "A"), sym("b", "B")},
		Edges: []model.Edge{{ID: "e1", Src: "a", Dst: "b", Kind: model.EdgeCalls}},
		Findings: []DashboardFinding{
			{ID: "f1", RuleID: "security/hardcoded-secret", Category: "security", Severity: "critical",
				Message: "secret in source", TargetURI: "file://x.go", Line: 12, Community: 0, NodeID: "a",
				Provenance: "EXTRACTED"},
			{ID: "f2", RuleID: "quality/todo-comment", Category: "quality", Severity: "info",
				Message: "TODO left in code", TargetURI: "file://y.go", Line: 3, Community: 1, NodeID: "b",
				Provenance: "INFERRED"},
		},
	}
}

func TestGenerateDashboardEmbedsGraphAndFindings(t *testing.T) {
	var buf bytes.Buffer
	if err := GenerateDashboard(&buf, sampleDashboardData()); err != nil {
		t.Fatalf("dashboard: %v", err)
	}
	out := buf.String()

	for _, want := range []string{
		"var rawNodes =", "var rawEdges =", "var rawFindings =",
		`"rule":"security/hardcoded-secret"`,
		`"sev":"critical"`,
		"id=\"f-sev\"", "id=\"f-cat\"", "id=\"f-comm\"", "id=\"f-prov\"", "id=\"f-file\"",
		"data-tab=\"graph\"", "data-tab=\"findings\"",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("dashboard output missing %q", want)
		}
	}
	// Header summary counts are substituted (2 findings, 1 edge).
	if !strings.Contains(out, "1 edges") || !strings.Contains(out, "2 findings") {
		t.Errorf("summary counts not substituted into header")
	}
}

func TestGenerateDashboardEmptyFindings(t *testing.T) {
	data := sampleDashboardData()
	data.Findings = nil
	var buf bytes.Buffer
	if err := GenerateDashboard(&buf, data); err != nil {
		t.Fatalf("dashboard (no findings): %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "var rawFindings = []") {
		t.Errorf("empty findings should embed an empty array, got:\n%s", out[:min(len(out), 400)])
	}
	if !strings.Contains(out, "0 findings") {
		t.Errorf("header should report 0 findings")
	}
	if !strings.Contains(out, "No findings match") {
		t.Errorf("missing empty-state text")
	}
}

func TestGenerateDashboardEscapesScriptClose(t *testing.T) {
	// Baseline </script> count with benign data.
	var base bytes.Buffer
	if err := GenerateDashboard(&base, sampleDashboardData()); err != nil {
		t.Fatalf("dashboard (benign): %v", err)
	}
	baseCount := strings.Count(base.String(), "</script>")

	// A finding message that would close the <script> block if injected raw.
	data := sampleDashboardData()
	data.Findings[0].Message = "</script><img src=x onerror=alert(1)>"
	data.Nodes[0].Label = "</script><b>pwn"
	var buf bytes.Buffer
	if err := GenerateDashboard(&buf, data); err != nil {
		t.Fatalf("dashboard (evil): %v", err)
	}
	out := buf.String()
	if got := strings.Count(out, "</script>"); got != baseCount {
		t.Errorf("malicious content introduced a new </script> (%d vs baseline %d) — escape failed", got, baseCount)
	}
	if strings.Contains(out, "</script><img") || strings.Contains(out, "</script><b>pwn") {
		t.Error("malicious content appears verbatim — not escaped")
	}
}

func TestGenerateDashboardCapsGraphNodes(t *testing.T) {
	data := sampleDashboardData()
	data.Nodes = []model.Node{sym("a", "A"), sym("b", "B"), sym("c", "C")}
	data.Edges = []model.Edge{
		{ID: "e1", Src: "a", Dst: "b", Kind: model.EdgeCalls},
		{ID: "e2", Src: "a", Dst: "c", Kind: model.EdgeCalls},
	}
	data.MaxNodes = 2
	var buf bytes.Buffer
	if err := GenerateDashboard(&buf, data); err != nil {
		t.Fatalf("dashboard: %v", err)
	}
	out := buf.String()
	// a (deg 2) and b (deg 1, ID tie-break) survive; c is trimmed.
	if !strings.Contains(out, `"id":"a"`) || !strings.Contains(out, `"id":"b"`) {
		t.Error("expected highest-degree nodes a and b to survive the cap")
	}
	if strings.Contains(out, `"id":"c"`) {
		t.Error("node c should have been trimmed by --max-nodes")
	}
}
