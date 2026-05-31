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

// EmbeddingStore is an optional capability a GraphStore may implement to
// persist node embeddings. It is kept separate from GraphStore so backends
// without vector support (e.g. the Kuzu stub) and test fakes are unaffected;
// callers type-assert and degrade gracefully when it is absent.
//
// Embeddings are stored in their own keyspace, out of the node JSON, so graph
// traversals (Neighbors, AllNodes, SearchNodes) stay lean.
type EmbeddingStore interface {
	// UpsertEmbeddings inserts or replaces embeddings by node ID.
	UpsertEmbeddings(ctx context.Context, embs []Embedding) error
	// AllEmbeddings returns every stored embedding. Used to rebuild the
	// in-memory ANN index at load.
	AllEmbeddings(ctx context.Context) ([]Embedding, error)
}

// NodeDeleter is an optional capability for batch node deletion by ID. It does
// point lookups keyed by ID rather than the full-scan DeleteBySourceURI, so
// callers that already hold the IDs (e.g. pruning resolved findings) can delete
// K nodes in O(K) instead of O(K·N). Callers type-assert and fall back to
// DeleteBySourceURI when a backend does not implement it.
type NodeDeleter interface {
	// DeleteByIDs deletes the given nodes plus their embeddings and adjacent
	// edges. Missing IDs are skipped. Returns the number actually deleted.
	DeleteByIDs(ctx context.Context, ids []string) (int, error)
}

// ReadOnlyAware is an optional capability letting callers detect a read-only
// store and fail writes early with an actionable message instead of surfacing a
// raw backend error.
type ReadOnlyAware interface {
	ReadOnly() bool
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

	// AllNodes returns every node in the graph. Used by bulk operations such as
	// clustering and metrics computation.
	AllNodes(ctx context.Context) ([]model.Node, error)

	// AllEdges returns every edge in the graph. Used by bulk operations such as
	// clustering and metrics computation.
	AllEdges(ctx context.Context) ([]model.Edge, error)

	// GodNodes returns the top-limit nodes ranked by betweenness read from
	// attrs["betweenness"], breaking ties by node ID for a stable order. If
	// communityID is non-nil, only nodes in that community are returned.
	//
	// Note: `chiton cluster` writes betweenness to the cluster sidecar (consumed
	// by the live god_nodes tool), NOT into node attrs — so a caller needing real
	// betweenness ranking must populate attrs["betweenness"] first; otherwise
	// every node scores 0 and results fall back to the stable ID order.
	GodNodes(ctx context.Context, communityID *int, limit int) ([]model.Node, error)

	// Snapshot writes a self-contained JSON export of the entire graph to w.
	// Intended for portability, git-diffable snapshots, and cross-repo merging.
	Snapshot(ctx context.Context, w io.Writer) error

	// Close flushes and releases all resources held by the store.
	Close() error
}
