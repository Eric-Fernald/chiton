package vector

import (
	"math"
	"sort"
)

// Flat is an exact cosine-similarity index backed by an in-memory slice.
// Vectors are L2-normalized on insert, so a search reduces to a dot product.
// Not safe for concurrent mutation; build once, then read.
type Flat struct {
	ids  []string
	vecs [][]float32    // parallel to ids; each normalized
	pos  map[string]int // id → index into ids/vecs
}

// NewFlat returns an empty Flat index.
func NewFlat() *Flat {
	return &Flat{pos: make(map[string]int)}
}

// Add inserts or replaces vectors by ID. Zero-length vectors are skipped.
func (f *Flat) Add(items []Item) {
	for _, it := range items {
		if len(it.Vec) == 0 {
			continue
		}
		v := normalized(it.Vec)
		if idx, ok := f.pos[it.ID]; ok {
			f.vecs[idx] = v
			continue
		}
		f.pos[it.ID] = len(f.ids)
		f.ids = append(f.ids, it.ID)
		f.vecs = append(f.vecs, v)
	}
}

// Search returns up to k results ranked by descending cosine similarity.
// A zero or dimension-mismatched query, or k <= 0, yields no results.
func (f *Flat) Search(query []float32, k int) []Result {
	if k <= 0 || len(f.ids) == 0 {
		return nil
	}
	q := normalized(query)
	if len(q) == 0 || isZero(q) {
		return nil
	}

	results := make([]Result, 0, len(f.ids))
	for i, v := range f.vecs {
		if len(v) != len(q) {
			continue // dimension mismatch (e.g. mixed embedders); skip
		}
		results = append(results, Result{ID: f.ids[i], Score: dot(q, v)})
	}
	if len(results) == 0 {
		return nil
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].Score != results[j].Score {
			return results[i].Score > results[j].Score
		}
		return results[i].ID < results[j].ID // stable tie-break
	})
	if len(results) > k {
		results = results[:k]
	}
	return results
}

// Delete removes vectors by ID using swap-with-last to stay O(1) per delete.
func (f *Flat) Delete(ids []string) {
	for _, id := range ids {
		idx, ok := f.pos[id]
		if !ok {
			continue
		}
		last := len(f.ids) - 1
		if idx != last {
			f.ids[idx] = f.ids[last]
			f.vecs[idx] = f.vecs[last]
			f.pos[f.ids[idx]] = idx
		}
		f.ids = f.ids[:last]
		f.vecs = f.vecs[:last]
		delete(f.pos, id)
	}
}

// Len reports how many vectors are indexed.
func (f *Flat) Len() int { return len(f.ids) }

// normalized returns a unit-L2 copy of v. A zero vector returns a zero copy.
func normalized(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	out := make([]float32, len(v))
	if sum == 0 {
		return out
	}
	inv := float32(1 / math.Sqrt(sum))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

// isZero reports whether every component of v is zero (a zero vector has no
// direction, so cosine similarity against it is meaningless).
func isZero(v []float32) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
