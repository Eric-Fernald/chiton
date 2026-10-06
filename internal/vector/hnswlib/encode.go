package hnswlib

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"slices"
)

// The stream is a magic string and version, the parameters, then every node (tombstones included, since
// they route searches) as key, tombstone flag, vector, and per-level links. Integers are uvarints and
// floats little-endian IEEE 754 bits. Keys must be strings or signed integers; the header records which.
const (
	magic         = "chiton-hnsw"
	topoMagic     = "chiton-links" // the same stream with every vector left out; not prefixed by magic
	formatVersion = 1

	// Bounds a stream must respect. They keep a corrupt or hostile stream from forcing huge allocations,
	// and from yielding a graph whose later Insert would.
	maxVectorLen = 1 << 20
	maxKeyLen    = 1 << 20
	maxLevels    = 64
	maxM         = 1 << 10
	maxEf        = 1 << 16
	maxLevelMult = 2 // 1/ln(2), the largest the M >= 2 default can produce, is about 1.44
)

// WriteTo writes the graph to w. It implements io.WriterTo.
func (g *Graph[K]) WriteTo(w io.Writer) (int64, error) { return g.write(w, true) }

// WriteTopology writes the graph as WriteTo does but without its vectors, for a caller that stores them
// elsewhere and hands them back to ReadTopology. It refuses a graph with tombstones, whose keys may since
// have been given new vectors.
func (g *Graph[K]) WriteTopology(w io.Writer) (int64, error) {
	if slices.Contains(g.dead, true) {
		return 0, errors.New("hnswlib: cannot write the topology of a graph with removed nodes")
	}
	return g.write(w, false)
}

func (g *Graph[K]) write(w io.Writer, withVecs bool) (int64, error) {
	cw := &countingWriter{w: w}
	e := encoder{w: bufio.NewWriter(cw)}
	kind, err := keyKind[K]()
	if err != nil {
		return 0, err
	}
	if withVecs {
		e.raw([]byte(magic))
	} else {
		e.raw([]byte(topoMagic))
	}
	e.uint(formatVersion)
	e.uint(kind)
	e.uint(uint64(g.Metric))
	e.uint(uint64(g.M))
	e.float64(g.LevelMult)
	e.uint(uint64(g.EfSearch))
	e.uint(uint64(g.EfConstruction))
	e.uint(uint64(len(g.keys)))
	e.uint(uint64(g.entry))
	e.uint(uint64(g.top))
	for id := range g.keys {
		e.key(g.keys[id])
		e.bool(g.dead[id])
		if withVecs {
			e.floats(g.vecs[id])
		}
		e.uint(uint64(len(g.links[id])))
		for _, list := range g.links[id] {
			e.uint(uint64(len(list)))
			for _, nb := range list {
				e.uint(uint64(nb))
			}
		}
	}
	if e.err == nil {
		e.err = e.w.Flush()
	}
	return cw.n, e.err
}

// ReadFrom replaces the graph, parameters included, with one written by WriteTo. On error the graph is
// unchanged. It implements io.ReaderFrom.
func (g *Graph[K]) ReadFrom(r io.Reader) (int64, error) {
	return g.read(r, magic, func(d *decoder, _ K) []float32 { return d.floats() })
}

// ReadTopology replaces the graph, parameters included, with one written by WriteTopology, taking each
// node's vector from vec, which reports false for a key it has none for. The graph keeps those vectors
// without copying them. On error the graph is unchanged.
func (g *Graph[K]) ReadTopology(r io.Reader, vec func(K) ([]float32, bool)) (int64, error) {
	return g.read(r, topoMagic, func(d *decoder, key K) []float32 {
		v, ok := vec(key)
		if !ok && d.err == nil {
			d.err = fmt.Errorf("hnswlib: no vector for key %v", key)
		}
		return v
	})
}

// read decodes a stream that starts with want, getting each node's vector from nodeVec.
func (g *Graph[K]) read(r io.Reader, want string, nodeVec func(d *decoder, key K) []float32) (int64, error) {
	cr := &countingReader{r: r}
	d := decoder{r: bufio.NewReader(cr)}
	if got := string(d.raw(len(want))); d.err == nil && got != want {
		return cr.n, errors.New("hnswlib: not a graph stream")
	}
	if v := d.uint(); d.err == nil && v != formatVersion {
		return cr.n, fmt.Errorf("hnswlib: unsupported format version %d", v)
	}
	wantKind, err := keyKind[K]()
	if err != nil {
		return cr.n, err
	}
	if kind := d.uint(); d.err == nil && kind != wantKind {
		return cr.n, fmt.Errorf("hnswlib: stream holds key kind %d, graph uses %d", kind, wantKind)
	}
	metric := Metric(d.uint())
	m, mult := d.uint(), d.float64()
	efSearch, efConstruction := d.uint(), d.uint()
	n, entry, top := d.uint(), d.uint(), d.uint()
	if d.err == nil && (!metric.valid() || m == 0 || m > maxM || !(mult >= 0 && mult <= maxLevelMult) ||
		efSearch > maxEf || efConstruction > maxEf || n > math.MaxInt32 || (n > 0 && entry >= n) || top >= maxLevels) {
		d.err = errors.New("hnswlib: corrupt graph header")
	}

	var (
		keys  []K
		vecs  [][]float32
		dead  []bool
		links [][][]int32
		index = map[K]int32{}
		dim   int
	)
	for id := uint64(0); id < n && d.err == nil; id++ {
		key := decodeKey[K](&d)
		isDead := d.bool()
		if isDead && want == topoMagic && d.err == nil {
			d.err = fmt.Errorf("hnswlib: topology node %d is removed", id)
		}
		var vec []float32
		if d.err == nil {
			vec = nodeVec(&d, key)
		}
		levels := d.uint()
		switch {
		case d.err != nil:
		case len(vec) == 0 || (dim != 0 && len(vec) != dim):
			d.err = fmt.Errorf("hnswlib: node %d has a %d-dimensional vector", id, len(vec))
		case levels == 0 || levels > maxLevels:
			d.err = fmt.Errorf("hnswlib: node %d has %d levels", id, levels)
		case !isDead:
			if _, dup := index[key]; dup {
				d.err = fmt.Errorf("hnswlib: key %v is live twice", key)
			}
		}
		dim = len(vec)
		lists := make([][]int32, 0, min(levels, maxLevels))
		for l := uint64(0); l < levels && d.err == nil; l++ {
			limit := m
			if l == 0 {
				limit = 2 * m
			}
			count := d.uint()
			if d.err == nil && count > limit {
				d.err = fmt.Errorf("hnswlib: node %d has %d links on level %d", id, count, l)
				break
			}
			list := make([]int32, count)
			for i := range list {
				nb := d.uint()
				if d.err == nil && (nb >= n || nb == id) {
					d.err = fmt.Errorf("hnswlib: node %d has an invalid link to %d", id, nb)
				}
				list[i] = int32(nb)
			}
			lists = append(lists, list)
		}
		keys, vecs, dead, links = append(keys, key), append(vecs, vec), append(dead, isDead), append(links, lists)
		if !isDead && d.err == nil {
			index[key] = int32(id)
		}
	}
	// Every link must point at a node that exists on that level, or a search would index past its lists.
	for id := 0; id < len(links) && d.err == nil; id++ {
		for l, list := range links[id] {
			for _, nb := range list {
				if len(links[nb]) <= l {
					d.err = fmt.Errorf("hnswlib: node %d links to %d on level %d, which %d lacks", id, nb, l, nb)
				}
			}
		}
	}
	if d.err != nil {
		if errors.Is(d.err, io.EOF) {
			d.err = io.ErrUnexpectedEOF
		}
		return cr.n, d.err
	}

	g.Metric, g.M, g.LevelMult = metric, int(m), mult
	g.EfSearch, g.EfConstruction = int(efSearch), int(efConstruction)
	g.keys, g.vecs, g.dead, g.links, g.index = keys, vecs, dead, links, index
	g.entry, g.top = int32(entry), int(top)
	switch {
	case len(index) == 0:
		g.clear()
	case dead[entry] || len(links[entry])-1 != int(top):
		g.electEntry()
	}
	if g.Rand == nil {
		g.Rand = rand.New(rand.NewSource(1))
	}
	return cr.n, nil
}

type encoder struct {
	w   *bufio.Writer
	err error
	buf [binary.MaxVarintLen64]byte
}

func (e *encoder) raw(b []byte) {
	if e.err == nil {
		_, e.err = e.w.Write(b)
	}
}

func (e *encoder) uint(x uint64) { e.raw(e.buf[:binary.PutUvarint(e.buf[:], x)]) }
func (e *encoder) int(x int64)   { e.raw(e.buf[:binary.PutVarint(e.buf[:], x)]) }

func (e *encoder) bool(b bool) {
	if b {
		e.uint(1)
	} else {
		e.uint(0)
	}
}

func (e *encoder) float64(f float64) {
	binary.LittleEndian.PutUint64(e.buf[:8], math.Float64bits(f))
	e.raw(e.buf[:8])
}

func (e *encoder) floats(v []float32) {
	e.uint(uint64(len(v)))
	for _, f := range v {
		binary.LittleEndian.PutUint32(e.buf[:4], math.Float32bits(f))
		e.raw(e.buf[:4])
	}
}

func (e *encoder) key(k any) {
	switch k := k.(type) {
	case string:
		e.uint(uint64(len(k)))
		if e.err == nil {
			_, e.err = e.w.WriteString(k)
		}
	case int:
		e.int(int64(k))
	case int32:
		e.int(int64(k))
	case int64:
		e.int(k)
	default:
		if e.err == nil {
			e.err = fmt.Errorf("hnswlib: cannot encode key type %T", k)
		}
	}
}

type decoder struct {
	r   *bufio.Reader
	err error
}

func (d *decoder) raw(n int) []byte {
	if d.err != nil {
		return nil
	}
	b := make([]byte, n)
	_, d.err = io.ReadFull(d.r, b)
	return b
}

func (d *decoder) uint() uint64 {
	if d.err != nil {
		return 0
	}
	x, err := binary.ReadUvarint(d.r)
	if err != nil { // on overflow ReadUvarint also returns a partial value; never let it escape
		d.err = err
		return 0
	}
	return x
}

func (d *decoder) int() int64 {
	if d.err != nil {
		return 0
	}
	x, err := binary.ReadVarint(d.r)
	if err != nil {
		d.err = err
		return 0
	}
	return x
}

func (d *decoder) bool() bool { return d.uint() != 0 }

func (d *decoder) float64() float64 {
	b := d.raw(8)
	if d.err != nil {
		return 0
	}
	return math.Float64frombits(binary.LittleEndian.Uint64(b))
}

func (d *decoder) floats() []float32 {
	n := d.uint()
	if d.err == nil && n > maxVectorLen {
		d.err = fmt.Errorf("hnswlib: vector length %d", n)
	}
	b := d.raw(int(n) * 4)
	if d.err != nil {
		return nil
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

func decodeKey[K any](d *decoder) K {
	var key K
	switch p := any(&key).(type) {
	case *string:
		n := d.uint()
		if d.err == nil && n > maxKeyLen {
			d.err = fmt.Errorf("hnswlib: key length %d", n)
		}
		*p = string(d.raw(int(n)))
	case *int:
		*p = int(d.int())
	case *int32:
		*p = int32(d.int())
	case *int64:
		*p = d.int()
	default:
		if d.err == nil {
			d.err = fmt.Errorf("hnswlib: cannot decode key type %T", key)
		}
	}
	return key
}

type countingWriter struct {
	w io.Writer
	n int64
}

func (c *countingWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += int64(n)
	return n, err
}

type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// keyKind numbers K's encoding for the stream header: 1 for strings, 2 for signed integers.
func keyKind[K any]() (uint64, error) {
	var key K
	switch any(key).(type) {
	case string:
		return 1, nil
	case int, int32, int64:
		return 2, nil
	}
	return 0, fmt.Errorf("hnswlib: cannot encode key type %T", key)
}
