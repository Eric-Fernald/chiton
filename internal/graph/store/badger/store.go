// Package badger implements GraphStore using BadgerDB (pure Go, no CGo).
// It is chiton's working backend; the kuzu package is only a stub.
package badger

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	badgerdb "github.com/dgraph-io/badger/v4"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// Key prefixes. Node IDs and edge IDs are 32-char lowercase hex strings,
// so prefix bytes ('d','e','n','s','v') never collide with key content.
const (
	pfxNode = "n:" // n:{nodeID}            → json(Node)
	pfxEdge = "e:" // e:{edgeID}            → json(Edge)
	pfxSrc  = "s:" // s:{src}:{kind}:{dst}  → edgeID   (forward adjacency)
	pfxDst  = "d:" // d:{dst}:{kind}:{src}  → edgeID   (reverse adjacency)
	pfxEmb  = "v:" // v:{nodeID}            → packed little-endian []float32
)

// Store is a BadgerDB-backed GraphStore.
type Store struct {
	db       *badgerdb.DB
	readOnly bool
}

// Compile-time checks that Store satisfies the store interfaces.
var (
	_ store.GraphStore        = (*Store)(nil)
	_ store.EmbeddingStore    = (*Store)(nil)
	_ store.NodeDeleter       = (*Store)(nil)
	_ store.ReadOnlyAware     = (*Store)(nil)
	_ store.NodeScanner       = (*Store)(nil)
	_ store.NodeGetter        = (*Store)(nil)
	_ store.EmbeddingIDLister = (*Store)(nil)
)

// ReadOnly reports whether the store was opened read-only, so callers can fail writes early
// with a clear message instead of a raw Badger error.
func (s *Store) ReadOnly() bool { return s.readOnly }

// Options configures store opens. The zero value is a plaintext store with
// default Badger settings — historic callers can keep using Open/OpenReadOnly.
type Options struct {
	// EncryptionKey, when non-nil, enables Badger's at-rest encryption; it must be 16, 24, or 32 bytes.
	EncryptionKey []byte
}

// Open opens or creates a BadgerDB store at path with read-write access.
func Open(path string) (*Store, error) { return OpenWith(path, Options{}) }

// OpenReadOnly opens an existing BadgerDB store in read-only mode without
// acquiring the directory lock, so chiton build can run concurrently.
func OpenReadOnly(path string) (*Store, error) { return OpenReadOnlyWith(path, Options{}) }

// OpenWith opens (or creates) the store with explicit options. When
// opts.EncryptionKey is set, Badger encryption is enabled.
func OpenWith(path string, opts Options) (*Store, error) {
	bopts := badgerdb.DefaultOptions(path).WithLogger(nil)
	bopts = applyEncryption(bopts, opts.EncryptionKey)
	db, err := badgerdb.Open(bopts)
	if err != nil {
		return nil, fmt.Errorf("open badger at %s: %w", path, err)
	}
	return &Store{db: db}, nil
}

// OpenReadOnlyWith is the read-only variant of OpenWith.
func OpenReadOnlyWith(path string, opts Options) (*Store, error) {
	bopts := badgerdb.DefaultOptions(path).WithLogger(nil).WithReadOnly(true).WithBypassLockGuard(true)
	bopts = applyEncryption(bopts, opts.EncryptionKey)
	db, err := badgerdb.Open(bopts)
	if err != nil {
		return nil, fmt.Errorf("open badger at %s: %w", path, err)
	}
	return &Store{db: db, readOnly: true}, nil
}

// applyEncryption enables Badger encryption when key is non-empty; Badger then
// requires a non-zero IndexCacheSize.
func applyEncryption(bopts badgerdb.Options, key []byte) badgerdb.Options {
	if len(key) == 0 {
		return bopts
	}
	return bopts.
		WithEncryptionKey(key).
		WithIndexCacheSize(64 << 20)
}

func (s *Store) Close() error { return s.db.Close() }

// --- key builders ---

func nodeKey(id string) []byte { return []byte(pfxNode + id) }
func edgeKey(id string) []byte { return []byte(pfxEdge + id) }
func embKey(id string) []byte  { return []byte(pfxEmb + id) }

func srcKey(e *model.Edge) []byte {
	return []byte(pfxSrc + e.Src + ":" + string(e.Kind) + ":" + e.Dst)
}

func dstKey(e *model.Edge) []byte {
	return []byte(pfxDst + e.Dst + ":" + string(e.Kind) + ":" + e.Src)
}

// --- transaction helpers ---

// errStopScan, returned from a scan callback, ends the scan early without error.
var errStopScan = errors.New("stop scan")

// prefixIterator iterates the keys under pfx without prefetching values: they are small and inline in the
// LSM tree, so prefetching would only add overhead.
func prefixIterator(txn *badgerdb.Txn, pfx string) *badgerdb.Iterator {
	opts := badgerdb.DefaultIteratorOptions
	opts.Prefix = []byte(pfx)
	opts.PrefetchValues = false
	return txn.NewIterator(opts)
}

// scan calls fn, in key order, with the key suffix and value of every entry under pfx; both are valid only
// during the call.
func scan(txn *badgerdb.Txn, pfx string, fn func(key, val []byte) error) error {
	it := prefixIterator(txn, pfx)
	defer it.Close()
	for it.Rewind(); it.Valid(); it.Next() {
		item := it.Item()
		key := item.Key()[len(pfx):]
		if err := item.Value(func(v []byte) error { return fn(key, v) }); err != nil {
			if errors.Is(err, errStopScan) {
				return nil
			}
			return err
		}
	}
	return nil
}

// scanAll decodes every JSON value under pfx, in key order.
func scanAll[T any](txn *badgerdb.Txn, pfx string) ([]T, error) {
	var out []T
	err := scan(txn, pfx, func(_, v []byte) error {
		var x T
		if err := json.Unmarshal(v, &x); err != nil {
			return err
		}
		out = append(out, x)
		return nil
	})
	return out, err
}

// loadJSON decodes the value at key, returning (nil, nil) when the key is absent.
func loadJSON[T any](txn *badgerdb.Txn, key []byte) (*T, error) {
	item, err := txn.Get(key)
	if errors.Is(err, badgerdb.ErrKeyNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var x T
	if err := item.Value(func(v []byte) error { return json.Unmarshal(v, &x) }); err != nil {
		return nil, err
	}
	return &x, nil
}

func loadNode(txn *badgerdb.Txn, id string) (*model.Node, error) {
	return loadJSON[model.Node](txn, nodeKey(id))
}

// appendLoaded appends the records stored under pfx+id for each id to dst, skipping absent ones.
func appendLoaded[T any](dst []T, txn *badgerdb.Txn, pfx string, ids []string) ([]T, error) {
	for _, id := range ids {
		x, err := loadJSON[T](txn, []byte(pfx+id))
		if err != nil {
			return dst, err
		}
		if x != nil {
			dst = append(dst, *x)
		}
	}
	return dst, nil
}

// adjacentEdgeKeys returns, deduplicated by edge ID, the edges touching nodeID as parsed from its adjacency
// keys (ID, Src, Dst and Kind only): outgoing first, each direction in key order. A nil kinds keeps every kind.
func adjacentEdgeKeys(txn *badgerdb.Txn, nodeID string, kinds map[model.EdgeKind]bool) ([]model.Edge, error) {
	var edges []model.Edge
	seen := map[string]bool{}
	for _, out := range [...]bool{true, false} {
		pfx := pfxDst + nodeID + ":"
		if out {
			pfx = pfxSrc + nodeID + ":"
		}
		// The key suffix is {kind}:{other endpoint}. Split at the last ':': node IDs contain none (the prefix
		// scan relies on that), but an imported edge kind may.
		if err := scan(txn, pfx, func(k, v []byte) error {
			i := bytes.LastIndexByte(k, ':')
			if i < 0 {
				return nil // malformed adjacency key
			}
			kind, other := string(k[:i]), string(k[i+1:])
			if (kinds != nil && !kinds[model.EdgeKind(kind)]) || seen[string(v)] {
				return nil
			}
			seen[string(v)] = true
			e := model.Edge{ID: string(v), Src: nodeID, Dst: other, Kind: model.EdgeKind(kind)}
			if !out {
				e.Src, e.Dst = other, nodeID
			}
			edges = append(edges, e)
			return nil
		}); err != nil {
			return nil, err
		}
	}
	return edges, nil
}

// adjacentEdges loads the edges adjacent to nodeID in adjacentEdgeKeys order, skipping missing records.
func adjacentEdges(txn *badgerdb.Txn, nodeID string, kinds map[model.EdgeKind]bool) ([]model.Edge, error) {
	keyed, err := adjacentEdgeKeys(txn, nodeID, kinds)
	if err != nil {
		return nil, err
	}
	ids := make([]string, len(keyed))
	for i, e := range keyed {
		ids[i] = e.ID
	}
	return appendLoaded[model.Edge](nil, txn, pfxEdge, ids)
}

// deleteNode queues deletion of node id, its embedding, and every adjacent edge with its adjacency keys.
func deleteNode(txn *badgerdb.Txn, wb *badgerdb.WriteBatch, id string) error {
	adj, err := adjacentEdgeKeys(txn, id, nil)
	if err != nil {
		return err
	}
	keys := make([][]byte, 0, 2+3*len(adj))
	keys = append(keys, nodeKey(id), embKey(id))
	for i := range adj {
		keys = append(keys, edgeKey(adj[i].ID), srcKey(&adj[i]), dstKey(&adj[i]))
	}
	for _, k := range keys {
		if err := wb.Delete(k); err != nil {
			return err
		}
	}
	return nil
}

// batch runs fill against a new write batch, flushing it on success and discarding it on error.
func (s *Store) batch(fill func(wb *badgerdb.WriteBatch) error) error {
	wb := s.db.NewWriteBatch()
	if err := fill(wb); err != nil {
		wb.Cancel()
		return err
	}
	return wb.Flush()
}

// --- GraphStore implementation ---

func (s *Store) UpsertNodes(_ context.Context, nodes []model.Node) error {
	now := time.Now()
	return s.batch(func(wb *badgerdb.WriteBatch) error {
		for i := range nodes {
			n := nodes[i]
			if n.CreatedAt.IsZero() {
				n.CreatedAt = now
			}
			n.UpdatedAt = now
			data, err := json.Marshal(n)
			if err != nil {
				return fmt.Errorf("marshal node %s: %w", n.ID, err)
			}
			if err := wb.Set(nodeKey(n.ID), data); err != nil {
				return err
			}
		}
		return nil
	})
}

func (s *Store) UpsertEdges(_ context.Context, edges []model.Edge) error {
	now := time.Now()
	return s.batch(func(wb *badgerdb.WriteBatch) error {
		for i := range edges {
			e := edges[i]
			if e.CreatedAt.IsZero() {
				e.CreatedAt = now
			}
			data, err := json.Marshal(e)
			if err != nil {
				return fmt.Errorf("marshal edge %s: %w", e.ID, err)
			}
			eid := []byte(e.ID)
			if err := wb.Set(edgeKey(e.ID), data); err != nil {
				return err
			}
			if err := wb.Set(srcKey(&e), eid); err != nil {
				return err
			}
			if err := wb.Set(dstKey(&e), eid); err != nil {
				return err
			}
		}
		return nil
	})
}

func (s *Store) GetNode(_ context.Context, id string) (*model.Node, error) {
	var result *model.Node
	err := s.db.View(func(txn *badgerdb.Txn) error {
		n, err := loadNode(txn, id)
		result = n
		return err
	})
	return result, err
}

// GetNodes loads the nodes with the given IDs that exist, in ids order, in one read transaction.
// Implements store.NodeGetter.
func (s *Store) GetNodes(_ context.Context, ids []string) ([]model.Node, error) {
	var nodes []model.Node
	err := s.db.View(func(txn *badgerdb.Txn) error {
		var err error
		nodes, err = appendLoaded(nodes, txn, pfxNode, ids)
		return err
	})
	return nodes, err
}

// Neighbors returns all nodes reachable from id within depth hops,
// traversing edges in both directions. Results are deduplicated.
func (s *Store) Neighbors(_ context.Context, id string, depth int, edgeKinds []model.EdgeKind) (store.Subgraph, error) {
	var kindFilter map[model.EdgeKind]bool // nil traverses every kind
	if len(edgeKinds) > 0 {
		kindFilter = make(map[model.EdgeKind]bool, len(edgeKinds))
		for _, k := range edgeKinds {
			kindFilter[k] = true
		}
	}

	var sub store.Subgraph
	err := s.db.View(func(txn *badgerdb.Txn) error {
		seed, err := loadNode(txn, id)
		if err != nil {
			return err
		}
		if seed != nil {
			sub.Nodes = append(sub.Nodes, *seed)
		}

		visited := map[string]bool{id: true}
		seenEdges := map[string]bool{}
		frontier := []string{id}
		for d := 0; d < depth && len(frontier) > 0; d++ {
			var nextFrontier []string
			for _, nodeID := range frontier {
				adjEdges, err := adjacentEdges(txn, nodeID, kindFilter)
				if err != nil {
					return err
				}
				for _, e := range adjEdges {
					if !seenEdges[e.ID] {
						seenEdges[e.ID] = true
						sub.Edges = append(sub.Edges, e)
					}
					neighbor := e.Dst
					if neighbor == nodeID {
						neighbor = e.Src
					}
					if !visited[neighbor] {
						visited[neighbor] = true
						nextFrontier = append(nextFrontier, neighbor)
					}
				}
			}

			if sub.Nodes, err = appendLoaded(sub.Nodes, txn, pfxNode, nextFrontier); err != nil {
				return err
			}
			frontier = nextFrontier
		}
		return nil
	})
	if err != nil {
		return store.Subgraph{}, err
	}
	return sub, nil
}

// ShortestPath returns the shortest undirected path between src and dst via BFS.
// Returns an empty Path (no error) when no path exists.
func (s *Store) ShortestPath(_ context.Context, src, dst string) (store.Path, error) {
	var path store.Path
	err := s.db.View(func(txn *badgerdb.Txn) error {
		if src == dst {
			n, err := loadNode(txn, src)
			if err == nil && n != nil {
				path.Nodes = []model.Node{*n}
			}
			return err
		}

		type step struct {
			parent string
			edgeID string
		}
		seen := map[string]step{src: {}}
		queue := []string{src}
		found := false
		for len(queue) > 0 && !found {
			curr := queue[0]
			queue = queue[1:]
			adjEdges, err := adjacentEdges(txn, curr, nil)
			if err != nil {
				return err
			}
			for _, e := range adjEdges {
				neighbor := e.Dst
				if neighbor == curr {
					neighbor = e.Src
				}
				if _, ok := seen[neighbor]; ok {
					continue
				}
				seen[neighbor] = step{parent: curr, edgeID: e.ID}
				if neighbor == dst {
					found = true
					break
				}
				queue = append(queue, neighbor)
			}
		}
		if !found {
			return nil
		}

		// Reconstruct the path backwards from dst.
		var nodeIDs, edgeIDs []string
		for curr := dst; curr != src; curr = seen[curr].parent {
			nodeIDs = append(nodeIDs, curr)
			edgeIDs = append(edgeIDs, seen[curr].edgeID)
		}
		nodeIDs = append(nodeIDs, src)
		slices.Reverse(nodeIDs)
		slices.Reverse(edgeIDs)

		var err error
		if path.Nodes, err = appendLoaded(path.Nodes, txn, pfxNode, nodeIDs); err != nil {
			return err
		}
		path.Edges, err = appendLoaded(path.Edges, txn, pfxEdge, edgeIDs)
		return err
	})
	return path, err
}

// stringProbe returns s's JSON encoding, which appears verbatim in any node record with a string field or attr
// equal to s, or nil when escaping could differ across Go versions (control chars, U+FFFD, invalid UTF-8).
func stringProbe(s string) []byte {
	if strings.ContainsRune(s, utf8.RuneError) || strings.ContainsFunc(s, func(r rune) bool { return r < 0x20 }) {
		return nil
	}
	b, err := json.Marshal(s)
	if err != nil {
		return nil
	}
	return b
}

// nodeProbe prefilters raw node records for a store.NodeFilter without false negatives: a matching record
// contains every probe in all and, when anyOf is non-empty, at least one probe in anyOf.
type nodeProbe struct{ all, anyOf [][]byte }

func newNodeProbe(f store.NodeFilter) nodeProbe {
	var p nodeProbe
	for _, k := range f.Kinds {
		b := stringProbe(string(k))
		if b == nil {
			p.anyOf = nil // one unprobeable kind disables the any-of check
			break
		}
		p.anyOf = append(p.anyOf, b)
	}
	vals := []string{f.SourceURI}
	for _, v := range f.Attrs {
		vals = append(vals, v)
	}
	for _, v := range vals {
		// Empty SourceURI doesn't constrain, and an empty attr value also matches an absent attr: no probe.
		if v == "" {
			continue
		}
		if b := stringProbe(v); b != nil {
			p.all = append(p.all, b)
		}
	}
	return p
}

func (p nodeProbe) candidate(rec []byte) bool {
	for _, b := range p.all {
		if !bytes.Contains(rec, b) {
			return false
		}
	}
	if len(p.anyOf) == 0 {
		return true
	}
	return slices.ContainsFunc(p.anyOf, func(b []byte) bool { return bytes.Contains(rec, b) })
}

// NodesMatching returns the nodes matching f in AllNodes order, decoding only records that pass the raw-bytes
// probe. Implements store.NodeScanner.
func (s *Store) NodesMatching(_ context.Context, f store.NodeFilter) ([]model.Node, error) {
	probe := newNodeProbe(f)
	var nodes []model.Node
	err := s.db.View(func(txn *badgerdb.Txn) error {
		return scan(txn, pfxNode, func(_, v []byte) error {
			if !probe.candidate(v) {
				return nil
			}
			var n model.Node
			if err := json.Unmarshal(v, &n); err != nil {
				return err
			}
			if f.Match(n) {
				nodes = append(nodes, n)
			}
			return nil
		})
	})
	return nodes, err
}

// DeleteBySourceURI removes all nodes whose source_uri matches uri and
// all edges adjacent to those nodes. Returns the number of nodes deleted.
func (s *Store) DeleteBySourceURI(_ context.Context, uri string) (int, error) {
	probe := stringProbe(uri)
	deleted := 0
	err := s.batch(func(wb *badgerdb.WriteBatch) error {
		return s.db.View(func(txn *badgerdb.Txn) error {
			var ids []string
			if err := scan(txn, pfxNode, func(_, v []byte) error {
				if probe != nil && !bytes.Contains(v, probe) {
					return nil
				}
				var n struct {
					ID        string `json:"id"`
					SourceURI string `json:"source_uri"`
				}
				if err := json.Unmarshal(v, &n); err != nil {
					return err
				}
				if n.SourceURI == uri {
					ids = append(ids, n.ID)
				}
				return nil
			}); err != nil {
				return err
			}
			for _, id := range ids {
				if err := deleteNode(txn, wb, id); err != nil {
					return err
				}
			}
			deleted = len(ids)
			return nil
		})
	})
	return deleted, err
}

// DeleteByIDs deletes nodes by ID plus their embeddings and adjacent edges, skipping missing IDs.
// Point lookups keep it O(K) in the number of IDs, unlike DeleteBySourceURI's full scan.
func (s *Store) DeleteByIDs(_ context.Context, ids []string) (int, error) {
	if len(ids) == 0 {
		return 0, nil
	}
	deleted := 0
	err := s.batch(func(wb *badgerdb.WriteBatch) error {
		return s.db.View(func(txn *badgerdb.Txn) error {
			n := 0
			for _, id := range ids {
				if _, err := txn.Get(nodeKey(id)); err != nil {
					if errors.Is(err, badgerdb.ErrKeyNotFound) {
						continue
					}
					return err
				}
				if err := deleteNode(txn, wb, id); err != nil {
					return err
				}
				n++
			}
			deleted = n
			return nil
		})
	})
	return deleted, err
}

// SearchNodes returns all nodes whose label contains query (case-insensitive).
func (s *Store) SearchNodes(_ context.Context, query string, limit int) ([]model.Node, error) {
	q := strings.ToLower(query)
	var results []model.Node
	err := s.db.View(func(txn *badgerdb.Txn) error {
		return scan(txn, pfxNode, func(_, v []byte) error {
			// Decode just the label first; most nodes miss, so the full decode is skipped for them.
			var probe struct {
				Label string `json:"label"`
			}
			if err := json.Unmarshal(v, &probe); err != nil {
				return err
			}
			if !strings.Contains(strings.ToLower(probe.Label), q) {
				return nil
			}
			var n model.Node
			if err := json.Unmarshal(v, &n); err != nil {
				return err
			}
			results = append(results, n)
			if limit > 0 && len(results) >= limit {
				return errStopScan
			}
			return nil
		})
	})
	return results, err
}

// AllNodes returns every node in the graph by scanning the node key space.
func (s *Store) AllNodes(_ context.Context) ([]model.Node, error) {
	var nodes []model.Node
	err := s.db.View(func(txn *badgerdb.Txn) error {
		var err error
		nodes, err = scanAll[model.Node](txn, pfxNode)
		return err
	})
	return nodes, err
}

// AllEdges returns every edge in the graph by scanning the edge key space.
func (s *Store) AllEdges(_ context.Context) ([]model.Edge, error) {
	var edges []model.Edge
	err := s.db.View(func(txn *badgerdb.Txn) error {
		var err error
		edges, err = scanAll[model.Edge](txn, pfxEdge)
		return err
	})
	return edges, err
}

// UpsertEmbeddings stores vectors packed (little-endian float32) in the v: keyspace, out of the node JSON,
// so graph traversals stay lean. Implements store.EmbeddingStore.
func (s *Store) UpsertEmbeddings(_ context.Context, embs []store.Embedding) error {
	return s.batch(func(wb *badgerdb.WriteBatch) error {
		for _, e := range embs {
			if e.NodeID == "" {
				continue
			}
			if err := wb.Set(embKey(e.NodeID), encodeVec(e.Vec)); err != nil {
				return err
			}
		}
		return nil
	})
}

// EmbeddingIDs returns the node ID of every stored embedding in key order, reading keys only.
// Implements store.EmbeddingIDLister.
func (s *Store) EmbeddingIDs(_ context.Context) ([]string, error) {
	var ids []string
	err := s.db.View(func(txn *badgerdb.Txn) error {
		it := prefixIterator(txn, pfxEmb)
		defer it.Close()
		for it.Rewind(); it.Valid(); it.Next() {
			ids = append(ids, string(it.Item().Key()[len(pfxEmb):]))
		}
		return nil
	})
	return ids, err
}

// AllEmbeddings returns every stored embedding by scanning the v: keyspace.
// Implements store.EmbeddingStore.
func (s *Store) AllEmbeddings(_ context.Context) ([]store.Embedding, error) {
	var embs []store.Embedding
	err := s.db.View(func(txn *badgerdb.Txn) error {
		return scan(txn, pfxEmb, func(k, v []byte) error {
			embs = append(embs, store.Embedding{NodeID: string(k), Vec: decodeVec(v)})
			return nil
		})
	})
	return embs, err
}

// encodeVec packs a float32 slice as little-endian bytes (4 bytes per value).
func encodeVec(vec []float32) []byte {
	b := make([]byte, 4*len(vec))
	for i, f := range vec {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

// decodeVec reverses encodeVec. A length not divisible by 4 yields nil.
func decodeVec(b []byte) []float32 {
	if len(b)%4 != 0 {
		return nil
	}
	vec := make([]float32, len(b)/4)
	for i := range vec {
		vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return vec
}

// Snapshot writes a self-contained JSON export of all nodes and edges.
func (s *Store) Snapshot(_ context.Context, w io.Writer) error {
	type snap struct {
		Nodes []model.Node `json:"nodes"`
		Edges []model.Edge `json:"edges"`
	}
	var out snap

	if err := s.db.View(func(txn *badgerdb.Txn) error {
		// Embeddings are attached to their nodes, keeping the export self-contained and portable.
		embByID := map[string][]float32{}
		if err := scan(txn, pfxEmb, func(k, v []byte) error {
			embByID[string(k)] = decodeVec(v)
			return nil
		}); err != nil {
			return err
		}

		var err error
		if out.Nodes, err = scanAll[model.Node](txn, pfxNode); err != nil {
			return err
		}
		for i := range out.Nodes {
			if vec, ok := embByID[out.Nodes[i].ID]; ok {
				out.Nodes[i].Embedding = vec
			}
		}
		out.Edges, err = scanAll[model.Edge](txn, pfxEdge)
		return err
	}); err != nil {
		return err
	}

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}
