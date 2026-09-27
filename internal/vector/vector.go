// Package vector provides a nearest-neighbour index over node embeddings behind a swappable Index interface.
// Flat, the exact pure-Go backend, is O(n·d) per query and fast enough for a few hundred thousand vectors.
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
