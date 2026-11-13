package badger

import (
	"context"
	"fmt"
	"path/filepath"
	"slices"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func idsOf(nodes []model.Node) []string {
	out := make([]string, len(nodes))
	for i, n := range nodes {
		out[i] = n.ID
	}
	return out
}

func TestReadOnlySearchNodesMatchesScan(t *testing.T) {
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "graph.badger")
	rw, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	var nodes []model.Node
	for i, l := range []string{"Fetch", "fetchAll", "FETCH_ALL", "Straße", "ÜBER", "", "prefetcher", "other"} {
		nodes = append(nodes, model.Node{ID: fmt.Sprintf("n%02d", i), Kind: model.KindSymbol, Label: l})
	}
	if err := rw.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	type query struct {
		q     string
		limit int
	}
	want := map[query][]string{}
	for _, q := range []string{"", "fetch", "FETCH", "straße", "über", "zzz", "_"} {
		for _, limit := range []int{0, 1, 2} {
			got, err := rw.SearchNodes(ctx, q, limit)
			if err != nil {
				t.Fatal(err)
			}
			want[query{q, limit}] = idsOf(got)
		}
	}
	if err := rw.Close(); err != nil {
		t.Fatal(err)
	}

	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ro.Close()
	for k, w := range want {
		got, err := ro.SearchNodes(ctx, k.q, k.limit)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(idsOf(got), w) {
			t.Errorf("read-only SearchNodes(%q, %d) = %v, want the scan's %v", k.q, k.limit, idsOf(got), w)
		}
	}
}

// TestSearchNodesLimitKeepsExactLabel checks a limit can't cut off the exact label behind earlier substring matches.
func TestSearchNodesLimitKeepsExactLabel(t *testing.T) {
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "graph.badger")
	rw, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	nodes := []model.Node{
		{ID: "n00", Kind: model.KindSymbol, Label: "NewEngine"},
		{ID: "n01", Kind: model.KindSymbol, Label: "NewStore"},
		{ID: "n02", Kind: model.KindSymbol, Label: "renew"},
		{ID: "n03", Kind: model.KindSymbol, Label: "New"},
		{ID: "n04", Kind: model.KindSymbol, Label: "new"},
	}
	if err := rw.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	check := func(s *Store, name string) {
		t.Helper()
		for limit, want := range map[int][]string{
			1: {"n03"},
			2: {"n03", "n04"},
			3: {"n03", "n04", "n00"},
			0: {"n03", "n04", "n00", "n01", "n02"},
		} {
			got, err := s.SearchNodes(ctx, "New", limit)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(idsOf(got), want) {
				t.Errorf("%s SearchNodes(New, %d) = %v, want exact labels first: %v", name, limit, idsOf(got), want)
			}
		}
	}
	check(rw, "writable")
	if err := rw.Close(); err != nil {
		t.Fatal(err)
	}
	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ro.Close()
	check(ro, "read-only")
}

func TestReadOnlySearchNodesSkipsRecordDecoding(t *testing.T) {
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "graph.badger")
	rw, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	nodes := make([]model.Node, 2000)
	for i := range nodes {
		nodes[i] = model.Node{ID: fmt.Sprintf("n%05d", i), Kind: model.KindSymbol, Label: fmt.Sprintf("sym%05d", i)}
	}
	nodes[1999].Label = "needle"
	if err := rw.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	if err := rw.Close(); err != nil {
		t.Fatal(err)
	}
	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ro.Close()
	if got, err := ro.SearchNodes(ctx, "NEEDLE", 1); err != nil || len(got) != 1 || got[0].ID != "n01999" {
		t.Fatalf("SearchNodes(NEEDLE) = %v, %v", idsOf(got), err)
	}
	// A scan decodes 2000 labels (at least one allocation each); the index loads one record.
	allocs := testing.AllocsPerRun(10, func() { _, _ = ro.SearchNodes(ctx, "needle", 1) })
	if allocs > 500 {
		t.Fatalf("read-only SearchNodes allocated %.0f times per call, want an indexed lookup (< 500)", allocs)
	}
}
