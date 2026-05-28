// Package vector provides an approximate-nearest-neighbour index over node
// embeddings, behind an interface so the backend can be swapped.
//
// v1 ships Flat: a pure-Go, dependency-free, exact cosine index. It is O(n·d)
// per query, which is comfortable into the low hundreds of thousands of
// vectors — the corpora chiton targets in v1. A future HNSW or sqlite-vec
// backend can implement the same Index interface, exactly the way the Kuzu
// store is deferred behind store.GraphStore.
package vector

// Item is one indexed vector keyed by its node ID.
type Item struct {
	ID  string
	Vec []float32
}

// Result is one search hit: a node ID and its similarity score (cosine,
// in [-1, 1]; higher is closer).
type Result struct {
	ID    string
	Score float32
}

// Index is an approximate-nearest-neighbour index over node embeddings.
// Implementations are not required to be safe for concurrent mutation; chiton
// builds the index once at load and only reads it while serving.
type Index interface {
	// Add inserts or replaces vectors by ID.
	Add(items []Item)
	// Search returns up to k results ranked by descending cosine similarity.
	Search(query []float32, k int) []Result
	// Delete removes vectors by ID. Unknown IDs are ignored.
	Delete(ids []string)
	// Len reports how many vectors are indexed.
	Len() int
}
