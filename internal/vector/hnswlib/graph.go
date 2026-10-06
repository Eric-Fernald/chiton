// Package hnswlib is a Hierarchical Navigable Small World graph for approximate nearest-neighbour search,
// written from the paper: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
// using Hierarchical Navigable Small World graphs" (2018). Insertion follows Algorithm 1, layer search
// Algorithm 2, and neighbour selection the heuristic of Algorithm 4. Nodes are dense int32 ids with slice
// adjacency, layer 0 allows 2·M links, and deletes are tombstones that keep routing searches.
package hnswlib

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sync"
)

// Hit is one search result.
type Hit[K cmp.Ordered] struct {
	Key  K
	Dist float32
}

// Graph is an HNSW graph keyed by K. Set the public parameters before the first Insert. Nearest only
// reads the graph, so concurrent searches are safe; Insert and Remove must not run alongside anything.
type Graph[K cmp.Ordered] struct {
	// Metric is the distance the graph is built on.
	Metric Metric
	// Rand draws node levels. A fixed seed makes builds reproducible.
	Rand *rand.Rand
	// M is the maximum number of links per node above layer 0. Layer 0 allows 2·M.
	M int
	// LevelMult scales node levels: a node's top level is floor(-ln(U)·LevelMult). The paper's choice,
	// used when this is zero, is 1/ln(M).
	LevelMult float64
	// EfSearch is the candidate list size for queries.
	EfSearch int
	// EfConstruction is the candidate list size while inserting.
	EfConstruction int

	keys    []K
	vecs    [][]float32
	links   [][][]int32 // links[node][level] lists neighbour ids
	dead    []bool      // tombstones still route searches but never appear in results
	index   map[K]int32 // key to the live node holding it
	entry   int32
	top     int // level of the entry point
	visited sync.Pool
}

// New returns an empty graph: cosine distance, M 16, EfSearch 64, EfConstruction 128, seed 1.
func New[K cmp.Ordered]() *Graph[K] {
	return &Graph[K]{
		Metric:         Cosine,
		Rand:           rand.New(rand.NewSource(1)),
		M:              16,
		EfSearch:       64,
		EfConstruction: 128,
	}
}

func (g *Graph[K]) dist(a, b []float32) float32 { return g.Metric.Distance(a, b) }

func (g *Graph[K]) maxLinks(level int) int {
	if level == 0 {
		return 2 * g.M
	}
	return g.M
}

func (g *Graph[K]) drawLevel() int {
	mult := g.LevelMult
	if mult <= 0 {
		mult = 1 / math.Log(float64(max(g.M, 2)))
	}
	return int(math.Floor(-math.Log(1-g.Rand.Float64()) * mult))
}

// Len returns the number of live nodes.
func (g *Graph[K]) Len() int { return len(g.index) }

// Dim returns the vector length the graph holds, or 0 when it is empty.
func (g *Graph[K]) Dim() int {
	if len(g.index) == 0 {
		return 0
	}
	return len(g.vecs[g.entry])
}

// Vector returns the vector stored under key.
func (g *Graph[K]) Vector(key K) ([]float32, bool) {
	id, ok := g.index[key]
	if !ok {
		return nil, false
	}
	return g.vecs[id], true
}

// Insert adds vec under key, replacing any vector already there. The graph keeps vec without copying it.
// It panics if vec's length differs from the graph's.
func (g *Graph[K]) Insert(key K, vec []float32) {
	if d := g.Dim(); d != 0 && d != len(vec) {
		panic(fmt.Sprintf("hnswlib: inserting a %d-dimensional vector into a %d-dimensional graph", len(vec), d))
	}
	if g.M <= 0 {
		g.M = 16
	}
	if !g.Metric.valid() {
		g.Metric = Cosine
	}
	if g.Rand == nil {
		g.Rand = rand.New(rand.NewSource(1))
	}
	if g.index == nil {
		g.index = map[K]int32{}
	}
	g.Remove(key)

	level := g.drawLevel()
	id := int32(len(g.keys))
	g.keys = append(g.keys, key)
	g.vecs = append(g.vecs, vec)
	g.dead = append(g.dead, false)
	lists := make([][]int32, level+1)
	for l := range lists {
		lists[l] = make([]int32, 0, g.maxLinks(l)+1)
	}
	g.links = append(g.links, lists)
	empty := len(g.index) == 0
	g.index[key] = id
	if empty {
		g.entry, g.top = id, level
		return
	}

	ep := g.entry
	for l := g.top; l > level; l-- {
		ep = g.descend(vec, ep, l)
	}
	eps := []candidate{{ep, g.dist(vec, g.vecs[ep])}}
	ef := max(g.EfConstruction, g.M)
	for l := min(level, g.top); l >= 0; l-- {
		found := g.searchLayer(vec, eps, ef, l)
		if len(found) == 0 {
			continue // only tombstones nearby; keep the previous entry points
		}
		chosen := g.pickLinks(found, g.M)
		g.links[id][l] = append(g.links[id][l], chosen...)
		for _, nb := range chosen {
			g.link(nb, id, l)
		}
		eps = found
	}
	if level > g.top {
		g.entry, g.top = id, level
	}
}

// link adds an edge from node to nb at level. When node's list overflows it is re-chosen with pickLinks.
func (g *Graph[K]) link(node, nb int32, level int) {
	list := append(g.links[node][level], nb)
	if len(list) <= g.maxLinks(level) {
		g.links[node][level] = list
		return
	}
	cands := make([]candidate, len(list))
	for i, c := range list {
		cands[i] = candidate{c, g.dist(g.vecs[node], g.vecs[c])}
	}
	slices.SortFunc(cands, candidate.compare)
	g.links[node][level] = append(list[:0], g.pickLinks(cands, g.maxLinks(level))...)
}

// pickLinks chooses up to m links from cands, which must be sorted nearest first. A candidate is kept only
// if it is closer to the base node than to every link already kept, so links fan out in different
// directions instead of piling into one cluster; that is what keeps the graph navigable.
func (g *Graph[K]) pickLinks(cands []candidate, m int) []int32 {
	out := make([]int32, 0, m)
	for _, c := range cands {
		if len(out) == m {
			break
		}
		diverse := true
		for _, kept := range out {
			if g.dist(g.vecs[c.id], g.vecs[kept]) < c.dist {
				diverse = false
				break
			}
		}
		if diverse {
			out = append(out, c.id)
		}
	}
	return out
}

// descend walks one level greedily from ep toward q until no neighbour is closer. Tombstones still route.
func (g *Graph[K]) descend(q []float32, ep int32, level int) int32 {
	cur, curDist := ep, g.dist(q, g.vecs[ep])
	for moved := true; moved; {
		moved = false
		for _, nb := range g.links[cur][level] {
			if d := g.dist(q, g.vecs[nb]); d < curDist {
				cur, curDist, moved = nb, d, true
			}
		}
	}
	return cur
}

// searchLayer runs a best-first search at one level and returns up to ef live nodes, nearest first.
// Tombstones are expanded so the search can pass through them, but are never returned.
func (g *Graph[K]) searchLayer(q []float32, eps []candidate, ef, level int) []candidate {
	seen := g.acquireSeen()
	defer g.visited.Put(seen)

	var frontier nearestFirst
	var best farthestFirst
	for _, ep := range eps {
		if seen.mark(ep.id) {
			continue
		}
		frontier.push(ep)
		if !g.dead[ep.id] {
			best.push(ep)
		}
	}
	for len(best) > ef {
		best.pop()
	}
	for len(frontier) > 0 {
		c := frontier.pop()
		if len(best) >= ef && c.dist > best[0].dist {
			break
		}
		for _, nb := range g.links[c.id][level] {
			if seen.mark(nb) {
				continue
			}
			d := g.dist(q, g.vecs[nb])
			if len(best) >= ef && d >= best[0].dist {
				continue
			}
			frontier.push(candidate{nb, d})
			if !g.dead[nb] {
				best.push(candidate{nb, d})
				if len(best) > ef {
					best.pop()
				}
			}
		}
	}
	out := []candidate(best)
	slices.SortFunc(out, candidate.compare)
	return out
}

// Nearest returns up to k live nodes closest to q, nearest first, ties by key. It panics if q's length
// differs from the graph's.
func (g *Graph[K]) Nearest(q []float32, k int) []Hit[K] {
	if len(g.index) == 0 || k <= 0 {
		return nil
	}
	if d := g.Dim(); d != len(q) {
		panic(fmt.Sprintf("hnswlib: searching a %d-dimensional graph with a %d-dimensional query", d, len(q)))
	}
	ep := g.entry
	for l := g.top; l > 0; l-- {
		ep = g.descend(q, ep, l)
	}
	found := g.searchLayer(q, []candidate{{ep, g.dist(q, g.vecs[ep])}}, max(g.EfSearch, k), 0)
	hits := make([]Hit[K], len(found))
	for i, c := range found {
		hits[i] = Hit[K]{Key: g.keys[c.id], Dist: c.dist}
	}
	slices.SortFunc(hits, func(a, b Hit[K]) int {
		if c := cmp.Compare(a.Dist, b.Dist); c != 0 {
			return c
		}
		return cmp.Compare(a.Key, b.Key)
	})
	return hits[:min(k, len(hits))]
}

// Remove tombstones the node under key and reports whether one was there. The node keeps routing searches
// so the graph stays connected, but is never returned again. Removing the last live node empties the graph.
func (g *Graph[K]) Remove(key K) bool {
	id, ok := g.index[key]
	if !ok {
		return false
	}
	delete(g.index, key)
	g.dead[id] = true
	switch {
	case len(g.index) == 0:
		g.clear()
	case id == g.entry:
		g.electEntry()
	}
	return true
}

// electEntry makes the live node with the highest level, lowest id on ties, the entry point.
func (g *Graph[K]) electEntry() {
	g.entry, g.top = -1, -1
	for id, dead := range g.dead {
		if !dead && len(g.links[id])-1 > g.top {
			g.entry, g.top = int32(id), len(g.links[id])-1
		}
	}
}

func (g *Graph[K]) clear() {
	g.keys, g.vecs, g.links, g.dead = nil, nil, nil, nil
	g.index = map[K]int32{}
	g.entry, g.top = 0, 0
}

type candidate struct {
	id   int32
	dist float32
}

func (a candidate) compare(b candidate) int {
	if c := cmp.Compare(a.dist, b.dist); c != 0 {
		return c
	}
	return cmp.Compare(a.id, b.id)
}

// nearestFirst is a binary min-heap on distance.
type nearestFirst []candidate

func (h *nearestFirst) push(c candidate) {
	s := append(*h, c)
	for i := len(s) - 1; i > 0; {
		p := (i - 1) / 2
		if s[p].dist <= s[i].dist {
			break
		}
		s[i], s[p] = s[p], s[i]
		i = p
	}
	*h = s
}

func (h *nearestFirst) pop() candidate {
	s := *h
	top, n := s[0], len(s)-1
	s[0] = s[n]
	s = s[:n]
	for i := 0; ; {
		c := 2*i + 1
		if c >= n {
			break
		}
		if c+1 < n && s[c+1].dist < s[c].dist {
			c++
		}
		if s[i].dist <= s[c].dist {
			break
		}
		s[i], s[c] = s[c], s[i]
		i = c
	}
	*h = s
	return top
}

// farthestFirst is a binary max-heap on distance, so the worst kept result is evicted first.
type farthestFirst []candidate

func (h *farthestFirst) push(c candidate) {
	s := append(*h, c)
	for i := len(s) - 1; i > 0; {
		p := (i - 1) / 2
		if s[p].dist >= s[i].dist {
			break
		}
		s[i], s[p] = s[p], s[i]
		i = p
	}
	*h = s
}

func (h *farthestFirst) pop() candidate {
	s := *h
	top, n := s[0], len(s)-1
	s[0] = s[n]
	s = s[:n]
	for i := 0; ; {
		c := 2*i + 1
		if c >= n {
			break
		}
		if c+1 < n && s[c+1].dist > s[c].dist {
			c++
		}
		if s[i].dist >= s[c].dist {
			break
		}
		s[i], s[c] = s[c], s[i]
		i = c
	}
	*h = s
	return top
}

// seenSet records node ids reached in one search. Bumping epoch empties it without clearing the slice,
// so pooled sets are reused cheaply.
type seenSet struct {
	epochs []uint32
	epoch  uint32
}

func (g *Graph[K]) acquireSeen() *seenSet {
	s, _ := g.visited.Get().(*seenSet)
	if s == nil {
		s = &seenSet{}
	}
	if len(s.epochs) < len(g.keys) {
		s.epochs = make([]uint32, len(g.keys)+len(g.keys)/4)
		s.epoch = 0
	}
	if s.epoch++; s.epoch == 0 { // wrapped after 2^32 searches: start over
		clear(s.epochs)
		s.epoch = 1
	}
	return s
}

// mark records id and reports whether it had already been recorded.
func (s *seenSet) mark(id int32) bool {
	if s.epochs[id] == s.epoch {
		return true
	}
	s.epochs[id] = s.epoch
	return false
}
