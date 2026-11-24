package vector

import (
	"math"
	"math/rand"
	"slices"

	"github.com/Eric-Fernald/chiton/internal/vector/hnswlib"
)

// Defaults for HNSWConfig's zero fields, chosen from the recall sweep in docs/BENCHMARK.md: queries stay
// well under a millisecond at 100k vectors, so ef_search favours recall on hard data over speed. Build
// time grows about linearly with ef_construction, and 100 lost 2.5 recall points to 200 only on hard data.
const (
	DefaultHNSWM              = 16
	DefaultHNSWEfSearch       = 128
	DefaultHNSWEfConstruction = 100
)

// HNSWConfig tunes an HNSW index. Zero fields take the Default* values.
type HNSWConfig struct {
	Dim            int   // vector length to accept; 0 adopts the first valid vector's length
	M              int   // maximum links per node (2·M on layer 0)
	EfSearch       int   // candidate list size per query
	EfConstruction int   // candidate list size while inserting
	Seed           int64 // level-generation seed, fixed so builds are reproducible
}

// HNSW is an approximate cosine index over an hnswlib graph. Search only reads the graph, so concurrent
// searches are safe once Add and Delete have stopped; neither may run alongside Search.
type HNSW struct {
	g       *hnswlib.Graph[string]
	dim     int
	skipped int
}

// NewHNSW returns an empty HNSW index.
func NewHNSW(cfg HNSWConfig) *HNSW {
	g := hnswlib.New[string]()
	// Vectors are normalized on the way in, so the dot product alone gives cosine distance.
	g.Metric = hnswlib.UnitCosine
	g.M = cmpOr(cfg.M, DefaultHNSWM)
	g.EfSearch = cmpOr(cfg.EfSearch, DefaultHNSWEfSearch)
	g.EfConstruction = cmpOr(cfg.EfConstruction, DefaultHNSWEfConstruction)
	g.Rand = rand.New(rand.NewSource(cfg.Seed))
	return &HNSW{g: g, dim: cfg.Dim}
}

// Add inserts or replaces vectors by ID, linking a large batch on every CPU. Vectors that are empty,
// all-zero, non-finite, or of another dimension are counted in Skipped and never reach the graph, which
// panics on a dimension mismatch.
func (h *HNSW) Add(items []Item) {
	ids, vecs := make([]string, 0, len(items)), make([][]float32, 0, len(items))
	for _, it := range items {
		v, ok := h.prepare(it.Vec)
		if !ok {
			h.skipped++
			continue
		}
		ids, vecs = append(ids, it.ID), append(vecs, v)
	}
	h.g.InsertBatch(ids, vecs)
}

// prepare validates vec against the index dimension and returns a unit-length copy.
func (h *HNSW) prepare(vec []float32) ([]float32, bool) {
	if len(vec) == 0 || (h.dim != 0 && len(vec) != h.dim) || !finite(vec) {
		return nil, false
	}
	v := normalized(vec)
	if isZero(v) {
		return nil, false
	}
	if h.dim == 0 {
		h.dim = len(v)
	}
	return v, true
}

// Search returns up to k approximate nearest neighbours ranked like Flat: descending cosine, ties by
// ascending ID. A zero, non-finite, or dimension-mismatched query, or k <= 0, yields no results.
func (h *HNSW) Search(query []float32, k int) []Result {
	if k <= 0 || h.g.Len() == 0 || len(query) != h.dim || !finite(query) {
		return nil
	}
	q := normalized(query)
	if isZero(q) {
		return nil
	}
	hits := h.g.Nearest(q, k)
	if len(hits) == 0 {
		return nil
	}
	out := make([]Result, len(hits))
	for i, hit := range hits {
		out[i] = Result{ID: hit.Key, Score: 1 - hit.Dist}
	}
	slices.SortFunc(out, rankOrder)
	return out
}

// Delete removes vectors by ID. Unknown IDs are ignored.
func (h *HNSW) Delete(ids []string) {
	for _, id := range ids {
		h.g.Remove(id)
	}
}

// Len reports how many vectors are indexed.
func (h *HNSW) Len() int { return h.g.Len() }

// Dim reports the vector length the index accepts, or 0 before the first valid vector.
func (h *HNSW) Dim() int { return h.dim }

// Skipped reports how many vectors Add rejected.
func (h *HNSW) Skipped() int { return h.skipped }

func finite(v []float32) bool {
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return false
		}
	}
	return true
}

// cmpOr returns v, or def when v is not positive.
func cmpOr(v, def int) int {
	if v > 0 {
		return v
	}
	return def
}
