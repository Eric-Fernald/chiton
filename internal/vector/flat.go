package vector

import (
	"cmp"
	"math"
	"slices"
	"strings"
)

// Flat is an exact cosine index. It L2-normalizes vectors on insert, so a search is a dot product. Vectors live in
// one slab of dim-wide rows. Not safe for concurrent mutation; build once, then read.
type Flat struct {
	dim  int // row width, fixed by the first vector added
	ids  []string
	vecs []float32      // row i, normalized, is vecs[i*dim:(i+1)*dim], parallel to ids
	pos  map[string]int // id → row
}

// NewFlat returns an empty Flat index.
func NewFlat() *Flat { return NewFlatSized(0) }

// NewFlatSized returns an empty Flat index presized for n vectors, so adding n of them never regrows it.
func NewFlatSized(n int) *Flat {
	return &Flat{ids: make([]string, 0, n), pos: make(map[string]int, n)}
}

// Add inserts or replaces vectors by ID. It skips zero-length vectors and any whose length differs from the first
// vector added, since a cosine score across dimensions is meaningless.
func (f *Flat) Add(items []Item) {
	for _, it := range items {
		if len(it.Vec) == 0 {
			continue
		}
		if f.dim == 0 {
			f.dim = len(it.Vec)
			f.vecs = make([]float32, 0, cap(f.ids)*f.dim)
		}
		if len(it.Vec) != f.dim {
			continue
		}
		idx, ok := f.pos[it.ID]
		if !ok {
			idx = len(f.ids)
			f.pos[it.ID] = idx
			f.ids = append(f.ids, it.ID)
			f.vecs = slices.Grow(f.vecs, f.dim)[:len(f.vecs)+f.dim]
		}
		normalizeInto(f.row(idx), it.Vec)
	}
}

// row returns vector i, capped so an append can't spill into row i+1.
func (f *Flat) row(i int) []float32 { return f.vecs[i*f.dim : (i+1)*f.dim : (i+1)*f.dim] }

// Search returns up to k results ranked by descending cosine similarity, ties broken by ascending ID.
// A zero or dimension-mismatched query, or k <= 0, yields no results.
func (f *Flat) Search(query []float32, k int) []Result {
	if k <= 0 || len(f.ids) == 0 {
		return nil
	}
	q := normalized(query)
	if len(q) != f.dim || isZero(q) {
		return nil // an empty, zero or dimension-mismatched query
	}

	// top is a min-heap of the best k hits seen so far, rooted at the worst one kept, so a scan is O(n log k).
	top := make([]Result, 0, min(k, len(f.ids)))
	for i, id := range f.ids {
		r := Result{ID: id, Score: dot(q, f.row(i))}
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

// rankOrder compares results in Search order, by descending score and then ascending ID.
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
			copy(f.row(idx), f.row(last))
			f.pos[f.ids[idx]] = idx
		}
		f.ids = f.ids[:last]
		f.vecs = f.vecs[:last*f.dim]
		delete(f.pos, id)
	}
}

// Len reports how many vectors are indexed.
func (f *Flat) Len() int { return len(f.ids) }

// normalized returns a unit-L2 copy of v. A zero vector returns a zero copy.
func normalized(v []float32) []float32 {
	out := make([]float32, len(v))
	normalizeInto(out, v)
	return out
}

// normalizeInto writes v scaled to unit L2 length into dst (len(v) long), or zeros for a zero vector.
func normalizeInto(dst, v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		clear(dst)
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i, x := range v {
		dst[i] = x * inv
	}
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
