package ingest

import (
	"context"
	"errors"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

// noBulkStore fails the whole-graph reads, so a test proves a re-cluster never makes them.
type noBulkStore struct{ *badgerstore.Store }

func (noBulkStore) AllNodes(context.Context) ([]model.Node, error) {
	return nil, errors.New("AllNodes called")
}

func (noBulkStore) AllEdges(context.Context) ([]model.Edge, error) {
	return nil, errors.New("AllEdges called")
}

// seedSidecar writes a full cluster pass over the harness store, as `chiton cluster` would.
func seedSidecar(t *testing.T, h *harness) {
	t.Helper()
	ctx := context.Background()
	nodes, _ := h.gs.AllNodes(ctx)
	edges, _ := h.gs.AllEdges(ctx)
	if err := cluster.WriteSidecar(cluster.SidecarPath(h.eng.cfg.GraphPath), cluster.Full(nodes, edges, 1.0, nil), nil); err != nil {
		t.Fatal(err)
	}
}

// TestReclusterReadsOnlyRegion checks an edit re-clusters from the store's neighbour reads alone, drops the
// replaced symbol and the pruned finding, and adds the new symbol.
func TestReclusterReadsOnlyRegion(t *testing.T) {
	h := newHarness(t, false, true)
	ctx := context.Background()
	p := h.write(t, "a.go", "package src\n\nimport \"fmt\"\n\n// TODO: tidy\nfunc Old() { fmt.Println() }\n")
	h.write(t, "b.go", "package src\n\nimport \"fmt\"\n\nfunc B() { fmt.Println() }\n")
	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	seedSidecar(t, h)
	views, _ := analyze.List(ctx, h.gs, analyze.ListOptions{Scope: model.FileURI(p)})
	if len(views) == 0 {
		t.Fatal("setup: a.go should have a TODO finding")
	}
	todoID := views[0].ID

	eng := New(noBulkStore{h.gs}, h.cache, h.eng.cfg, nil, analyze.New(h.gs, nil), func(string, ...any) {})
	h.write(t, "a.go", "package src\n\nimport \"fmt\"\n\nfunc New() { fmt.Println() }\n")
	sum, err := eng.SyncChanges(ctx, []Change{{Path: p, Op: OpUpsert}})
	if err != nil || !sum.Reclustered {
		t.Fatalf("SyncChanges = %+v, %v; want a re-cluster without whole-graph reads", sum, err)
	}
	res, err := cluster.LoadSidecar(cluster.SidecarPath(h.eng.cfg.GraphPath), nil)
	if err != nil {
		t.Fatal(err)
	}
	uri := model.FileURI(p)
	if _, ok := res.Nodes[model.NewNodeID(uri, "Old")]; ok {
		t.Error("replaced symbol Old is still in the sidecar")
	}
	if _, ok := res.Nodes[todoID]; ok {
		t.Error("pruned TODO finding is still in the sidecar")
	}
	if _, ok := res.Nodes[model.NewNodeID(uri, "New")]; !ok {
		t.Error("new symbol New is missing from the sidecar")
	}
	if res.NodeCount != len(res.Nodes) {
		t.Errorf("NodeCount %d != len(Nodes) %d", res.NodeCount, len(res.Nodes))
	}
}
