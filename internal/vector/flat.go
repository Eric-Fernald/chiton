package vector

import (
	"cmp"
	"math"
	"slices"
	"strings"
)

// Flat is an exact cosine index; vectors are L2-normalized on insert so a search is a dot product.
// Not safe for concurrent mutation; build once, then read.
type Flat struct {
	ids  []string
	vecs [][]float32    // parallel to ids; each normalized
	pos  map[string]int // id → index into ids/vecs
}

// NewFlat returns an empty Flat index.
func NewFlat() *Flat { return NewFlatSized(0) }

// NewFlatSized returns an empty Flat index with room for n vectors, so a caller that knows its count never regrows.
func NewFlatSized(n int) *Flat {
	return &Flat{ids: make([]string, 0, n), vecs: make([][]float32, 0, n), pos: make(map[string]int, n)}
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

// Search returns up to k results ranked by descending cosine similarity, ties broken by ascending ID.
// A zero or dimension-mismatched query, or k <= 0, yields no results.
func (f *Flat) Search(query []float32, k int) []Result {
	if k <= 0 || len(f.ids) == 0 {
		return nil
	}
	q := normalized(query)
	if len(q) == 0 || isZero(q) {
		return nil
	}

	// top is a min-heap of the best k hits seen so far (root = worst kept), so a scan is O(n log k).
	top := make([]Result, 0, min(k, len(f.ids)))
	for i, v := range f.vecs {
		if len(v) != len(q) {
			continue // dimension mismatch (e.g. mixed embedders); skip
		}
		r := Result{ID: f.ids[i], Score: dot(q, v)}
		switch {
		case len(top) < k:
			top = append(top, r)
			siftUp(top, len(top)-1)
		case worse(top[0], r):
			top[0] = r
			siftDown(top, 0)
		}
	}
	if len(top) == 0 {
		return nil
	}

	slices.SortFunc(top, rankOrder)
	return top
}

// rankOrder is Search's result order: descending score, ties broken by ascending ID.
func rankOrder(a, b Result) int {
	if c := cmp.Compare(b.Score, a.Score); c != 0 {
		return c
	}
	return strings.Compare(a.ID, b.ID)
}

// worse reports whether a ranks below b in Search order.
func worse(a, b Result) bool { return rankOrder(a, b) > 0 }

func siftUp(h []Result, i int) {
	for i > 0 {
		p := (i - 1) / 2
		if !worse(h[i], h[p]) {
			return
		}
		h[i], h[p] = h[p], h[i]
		i = p
	}
}

func siftDown(h []Result, i int) {
	for {
		c := 2*i + 1
		if c >= len(h) {
			return
		}
		if r := c + 1; r < len(h) && worse(h[r], h[c]) {
			c = r
		}
		if !worse(h[c], h[i]) {
			return
		}
		h[i], h[c] = h[c], h[i]
		i = c
	}
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
