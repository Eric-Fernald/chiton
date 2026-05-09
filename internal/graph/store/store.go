package store

import (
	"context"
	"io"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// Subgraph is a set of nodes and edges returned by neighborhood or path queries.
type Subgraph struct {
	Nodes []model.Node `json:"nodes"`
	Edges []model.Edge `json:"edges"`
}

// Path is an ordered sequence of nodes forming the shortest path between two nodes.
type Path struct {
	Nodes []model.Node `json:"nodes"`
	Edges []model.Edge `json:"edges"`
}

// GraphStore is the central abstraction over the underlying graph database.
// All packages depend on this interface, never on a concrete implementation,
// so the backing store (Kuzu, Badger, etc.) can be swapped without touching callers.
type GraphStore interface {
	// UpsertNodes inserts or updates nodes by ID. Implementations must be
	// idempotent and batch-friendly (callers may pass thousands at once).
	UpsertNodes(ctx context.Context, nodes []model.Node) error

	// UpsertEdges inserts or updates edges by ID.
	UpsertEdges(ctx context.Context, edges []model.Edge) error

	// GetNode returns a single node by its stable ID, or (nil, nil) if not found.
	GetNode(ctx context.Context, id string) (*model.Node, error)

	// Neighbors returns all nodes reachable from id within depth hops, optionally
	// filtered to the given edge kinds. Pass nil edgeKinds to traverse all kinds.
	Neighbors(ctx context.Context, id string, depth int, edgeKinds []model.EdgeKind) (Subgraph, error)

	// ShortestPath returns the shortest undirected path between src and dst.
	// Returns an empty Path (no error) when no path exists.
	ShortestPath(ctx context.Context, src, dst string) (Path, error)

	// DeleteBySourceURI removes all nodes and edges whose source_uri matches uri.
	// Returns the number of nodes deleted.
	DeleteBySourceURI(ctx context.Context, uri string) (int, error)

	// SearchNodes returns all nodes whose label contains query (case-insensitive
	// substring match). Pass limit=0 for no cap; callers typically pass 10.
	SearchNodes(ctx context.Context, query string, limit int) ([]model.Node, error)

	// Snapshot writes a self-contained JSON export of the entire graph to w.
	// Intended for portability, git-diffable snapshots, and cross-repo merging.
	Snapshot(ctx context.Context, w io.Writer) error

	// Close flushes and releases all resources held by the store.
	Close() error
}
