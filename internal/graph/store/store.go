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

// Embedding pairs a node ID with its embedding vector.
type Embedding struct {
	NodeID string    `json:"node_id"`
	Vec    []float32 `json:"vec"`
}

// EmbeddingStore is an optional GraphStore capability that persists node embeddings outside the node JSON,
// keeping traversals lean; callers type-assert and degrade gracefully when it is absent.
type EmbeddingStore interface {
	// UpsertEmbeddings inserts or replaces embeddings by node ID.
	UpsertEmbeddings(ctx context.Context, embs []Embedding) error
	// AllEmbeddings returns every stored embedding. Used to rebuild the
	// in-memory ANN index at load.
	AllEmbeddings(ctx context.Context) ([]Embedding, error)
}

// NodeDeleter is an optional capability for deleting nodes by ID with point lookups, O(K) instead of
// DeleteBySourceURI's full scan; callers type-assert and fall back to DeleteBySourceURI without it.
type NodeDeleter interface {
	// DeleteByIDs deletes the given nodes plus their embeddings and adjacent
	// edges. Missing IDs are skipped. Returns the number actually deleted.
	DeleteByIDs(ctx context.Context, ids []string) (int, error)
}

// ReadOnlyAware is an optional capability letting callers detect a read-only store and fail writes early
// with an actionable message instead of a raw backend error.
type ReadOnlyAware interface {
	ReadOnly() bool
}

// NodeScanner is an optional capability: a filtered AllNodes that skips decoding non-matching nodes.
// Callers use the NodesMatching helper, which falls back to AllNodes without it.
type NodeScanner interface {
	// NodesMatching returns the nodes f matches, in AllNodes order.
	NodesMatching(ctx context.Context, f NodeFilter) ([]model.Node, error)
}

// EmbeddingIDLister is an optional EmbeddingStore capability listing embedded node IDs without decoding
// vectors; callers use the EmbeddingIDs helper.
type EmbeddingIDLister interface {
	EmbeddingIDs(ctx context.Context) ([]string, error)
}

// GraphStore is the central abstraction over the graph database; packages depend on it, never on a
// concrete backend, so the store can be swapped without touching callers.
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

	// AllNodes returns every node in the graph. Used by bulk operations such as
	// clustering and metrics computation.
	AllNodes(ctx context.Context) ([]model.Node, error)

	// AllEdges returns every edge in the graph. Used by bulk operations such as
	// clustering and metrics computation.
	AllEdges(ctx context.Context) ([]model.Edge, error)

	// Snapshot writes a self-contained JSON export of the entire graph to w.
	// Intended for portability, git-diffable snapshots, and cross-repo merging.
	Snapshot(ctx context.Context, w io.Writer) error

	// Close flushes and releases all resources held by the store.
	Close() error
}
