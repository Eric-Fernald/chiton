package report

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
)

// TestGenerateIsDeterministic builds communities whose members tie on betweenness and share a source file,
// the case the old SourceURI-only tie-break left to map order, and checks every run renders the same report.
func TestGenerateIsDeterministic(t *testing.T) {
	res := &cluster.Results{GeneratedAt: time.Unix(0, 0).UTC(), NodeCount: 24, CommunityCount: 2, Nodes: map[string]cluster.NodeResult{}}
	for i := range 24 {
		res.Nodes[fmt.Sprint("id", i)] = cluster.NodeResult{
			Label: fmt.Sprint("sym", i%7), SourceURI: "file:///x.go", CommunityID: i % 2, Degree: 1,
		}
	}
	opts := Options{MaxGodNodes: 5, MaxPerCommunity: 12}
	var first bytes.Buffer
	if err := Generate(&first, res, nil, opts); err != nil {
		t.Fatal(err)
	}
	for range 20 {
		var again bytes.Buffer
		if err := Generate(&again, res, nil, opts); err != nil {
			t.Fatal(err)
		}
		if again.String() != first.String() {
			t.Fatalf("report changed between runs:\n%s\n---\n%s", first.String(), again.String())
		}
	}
	if !strings.Contains(first.String(), "### Community 0 (12 nodes)\n\n- `sym0` (file:///x.go)\n- `sym0` (file:///x.go)\n- `sym1`") {
		t.Errorf("community members are not ranked by label:\n%s", first.String())
	}
}
