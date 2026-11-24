package vector

import "sync/atomic"

// Swappable is an Index whose backend can be replaced while searches run. serve answers from Flat while
// HNSW builds in the background, then swaps HNSW in.
type Swappable struct {
	cur atomic.Pointer[box]
}

type box struct{ idx Index }

// NewSwappable returns a Swappable serving from idx.
func NewSwappable(idx Index) *Swappable {
	s := &Swappable{}
	s.Swap(idx)
	return s
}

// Swap makes idx the backend. Searches already running finish on the old one.
func (s *Swappable) Swap(idx Index) { s.cur.Store(&box{idx}) }

// Current returns the backend in use.
func (s *Swappable) Current() Index { return s.cur.Load().idx }

func (s *Swappable) Search(query []float32, k int) []Result { return s.Current().Search(query, k) }

// Add and Delete change the current backend, under the same rules as calling it directly: not alongside
// Search, and not alongside a Swap that could discard the change.
func (s *Swappable) Add(items []Item)    { s.Current().Add(items) }
func (s *Swappable) Delete(ids []string) { s.Current().Delete(ids) }
func (s *Swappable) Len() int            { return s.Current().Len() }
