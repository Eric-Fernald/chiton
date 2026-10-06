// Package vector provides a nearest-neighbour index over node embeddings behind a swappable Index interface.
// Flat is exact and O(n·d) per query: about 3 ms at 10k 768-dimensional vectors and 30 ms at 100k. HNSW
// is approximate and roughly logarithmic, well under a millisecond at 100k, but takes seconds to minutes to
// build. NewIndex picks between them; see docs/BENCHMARK.md.
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

// Index is a nearest-neighbour index over node embeddings. Implementations need not be safe for
// concurrent mutation, because chiton builds the index once at load and only reads it while serving.
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
