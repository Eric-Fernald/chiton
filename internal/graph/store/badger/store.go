// Package badger implements GraphStore using BadgerDB (pure Go, no CGo).
// Used as the fallback when Kuzu CGo bindings are unavailable.
package badger

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strings"
	"time"

	badgerdb "github.com/dgraph-io/badger/v4"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// Key prefixes. Node IDs and edge IDs are 32-char lowercase hex strings,
// so prefix bytes ('d','e','n','s') never collide with key content.
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
	_ store.GraphStore     = (*Store)(nil)
	_ store.EmbeddingStore = (*Store)(nil)
	_ store.NodeDeleter    = (*Store)(nil)
	_ store.ReadOnlyAware  = (*Store)(nil)
)

// ReadOnly reports whether the store was opened read-only. Callers use this to
// fail write operations early with an actionable message instead of surfacing a
// raw Badger "read-only transaction" error.
func (s *Store) ReadOnly() bool { return s.readOnly }

// Options configures store opens. The zero value is a plaintext store with
// default Badger settings — historic callers can keep using Open/OpenReadOnly.
type Options struct {
	// EncryptionKey, when non-nil, enables Badger's at-rest encryption.
	// Length must be 16, 24, or 32 bytes. Chiton's keyfile derives 32 bytes
	// (AES-256) by default.
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

// applyEncryption layers Badger encryption options on top of bopts when key
// is non-nil. Badger requires a non-zero IndexCacheSize whenever encryption
// is enabled; 64 MiB is a sane default for chiton's typical graph size.
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

func loadNode(txn *badgerdb.Txn, id string) (*model.Node, error) {
	item, err := txn.Get(nodeKey(id))
	if err == badgerdb.ErrKeyNotFound {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var n model.Node
	if err := item.Value(func(v []byte) error { return json.Unmarshal(v, &n) }); err != nil {
		return nil, err
	}
	return &n, nil
}

func loadEdge(txn *badgerdb.Txn, id string) (*model.Edge, error) {
	item, err := txn.Get(edgeKey(id))
	if err == badgerdb.ErrKeyNotFound {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var e model.Edge
	if err := item.Value(func(v []byte) error { return json.Unmarshal(v, &e) }); err != nil {
		return nil, err
	}
	return &e, nil
}

// adjacentEdges returns all edges where nodeID appears as src or dst.
// Each iterator is closed before the next is opened (Badger requirement).
func adjacentEdges(txn *badgerdb.Txn, nodeID string) ([]*model.Edge, error) {
	seen := map[string]bool{}
	var edges []*model.Edge

	for _, prefix := range []string{pfxSrc + nodeID + ":", pfxDst + nodeID + ":"} {
		pfx := []byte(prefix)
		it := txn.NewIterator(badgerdb.DefaultIteratorOptions)
		for it.Seek(pfx); it.ValidForPrefix(pfx); it.Next() {
			var edgeID string
			if err := it.Item().Value(func(v []byte) error {
				edgeID = string(v)
				return nil
			}); err != nil {
				it.Close()
				return nil, err
			}
			if seen[edgeID] {
				continue
			}
			seen[edgeID] = true
			e, err := loadEdge(txn, edgeID)
			if err != nil {
				it.Close()
				return nil, err
			}
			if e != nil {
				edges = append(edges, e)
			}
		}
		it.Close()
	}
	return edges, nil
}

// --- GraphStore implementation ---

func (s *Store) UpsertNodes(_ context.Context, nodes []model.Node) error {
	wb := s.db.NewWriteBatch()
	now := time.Now()
	for i := range nodes {
		n := nodes[i]
		if n.CreatedAt.IsZero() {
			n.CreatedAt = now
		}
		n.UpdatedAt = now
		data, err := json.Marshal(n)
		if err != nil {
			wb.Cancel()
			return fmt.Errorf("marshal node %s: %w", n.ID, err)
		}
		if err := wb.Set(nodeKey(n.ID), data); err != nil {
			wb.Cancel()
			return err
		}
	}
	return wb.Flush()
}

func (s *Store) UpsertEdges(_ context.Context, edges []model.Edge) error {
	wb := s.db.NewWriteBatch()
	now := time.Now()
	for i := range edges {
		e := edges[i]
		if e.CreatedAt.IsZero() {
			e.CreatedAt = now
		}
		data, err := json.Marshal(e)
		if err != nil {
			wb.Cancel()
			return fmt.Errorf("marshal edge %s: %w", e.ID, err)
		}
		eid := []byte(e.ID)
		if err := wb.Set(edgeKey(e.ID), data); err != nil {
			wb.Cancel()
			return err
		}
		if err := wb.Set(srcKey(&e), eid); err != nil {
			wb.Cancel()
			return err
		}
		if err := wb.Set(dstKey(&e), eid); err != nil {
			wb.Cancel()
			return err
		}
	}
	return wb.Flush()
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

// Neighbors returns all nodes reachable from id within depth hops,
// traversing edges in both directions. Results are deduplicated.
func (s *Store) Neighbors(_ context.Context, id string, depth int, edgeKinds []model.EdgeKind) (store.Subgraph, error) {
	kindFilter := make(map[model.EdgeKind]bool, len(edgeKinds))
	for _, k := range edgeKinds {
		kindFilter[k] = true
	}
	filterByKind := len(kindFilter) > 0

	visited := map[string]bool{id: true}
	frontier := []string{id}
	seenEdges := map[string]bool{}

	var resultNodes []model.Node
	var resultEdges []model.Edge

	seed, err := s.GetNode(context.TODO(), id)
	if err != nil {
		return store.Subgraph{}, err
	}
	if seed != nil {
		resultNodes = append(resultNodes, *seed)
	}

	for d := 0; d < depth && len(frontier) > 0; d++ {
		var nextFrontier []string

		if err := s.db.View(func(txn *badgerdb.Txn) error {
			for _, nodeID := range frontier {
				adjEdges, err := adjacentEdges(txn, nodeID)
				if err != nil {
					return err
				}
				for _, e := range adjEdges {
					if filterByKind && !kindFilter[e.Kind] {
						continue
					}
					if !seenEdges[e.ID] {
						seenEdges[e.ID] = true
						resultEdges = append(resultEdges, *e)
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
			return nil
		}); err != nil {
			return store.Subgraph{}, err
		}

		for _, nid := range nextFrontier {
			n, err := s.GetNode(context.TODO(), nid)
			if err != nil {
				return store.Subgraph{}, err
			}
			if n != nil {
				resultNodes = append(resultNodes, *n)
			}
		}
		frontier = nextFrontier
	}

	return store.Subgraph{Nodes: resultNodes, Edges: resultEdges}, nil
}

// ShortestPath returns the shortest undirected path between src and dst via BFS.
// Returns an empty Path (no error) when no path exists.
func (s *Store) ShortestPath(_ context.Context, src, dst string) (store.Path, error) {
	if src == dst {
		n, err := s.GetNode(context.TODO(), src)
		if err != nil || n == nil {
			return store.Path{}, err
		}
		return store.Path{Nodes: []model.Node{*n}}, nil
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

		if err := s.db.View(func(txn *badgerdb.Txn) error {
			adjEdges, err := adjacentEdges(txn, curr)
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
					return nil
				}
				queue = append(queue, neighbor)
			}
			return nil
		}); err != nil {
			return store.Path{}, err
		}
	}

	if !found {
		return store.Path{}, nil
	}

	// Reconstruct path backwards from dst.
	var nodeIDs, edgeIDs []string
	for curr := dst; curr != src; {
		v := seen[curr]
		nodeIDs = append([]string{curr}, nodeIDs...)
		edgeIDs = append([]string{v.edgeID}, edgeIDs...)
		curr = v.parent
	}
	nodeIDs = append([]string{src}, nodeIDs...)

	var nodes []model.Node
	var edges []model.Edge
	err := s.db.View(func(txn *badgerdb.Txn) error {
		for _, nid := range nodeIDs {
			n, err := loadNode(txn, nid)
			if err != nil {
				return err
			}
			if n != nil {
				nodes = append(nodes, *n)
			}
		}
		for _, eid := range edgeIDs {
			e, err := loadEdge(txn, eid)
			if err != nil {
				return err
			}
			if e != nil {
				edges = append(edges, *e)
			}
		}
		return nil
	})
	return store.Path{Nodes: nodes, Edges: edges}, err
}

// DeleteBySourceURI removes all nodes whose source_uri matches uri and
// all edges adjacent to those nodes. Returns the number of nodes deleted.
func (s *Store) DeleteBySourceURI(_ context.Context, uri string) (int, error) {
	var toDelete []model.Node
	if err := s.db.View(func(txn *badgerdb.Txn) error {
		pfx := []byte(pfxNode)
		it := txn.NewIterator(badgerdb.DefaultIteratorOptions)
		defer it.Close()
		for it.Seek(pfx); it.ValidForPrefix(pfx); it.Next() {
			var n model.Node
			if err := it.Item().Value(func(v []byte) error {
				return json.Unmarshal(v, &n)
			}); err != nil {
				return err
			}
			if n.SourceURI == uri {
				toDelete = append(toDelete, n)
			}
		}
		return nil
	}); err != nil {
		return 0, err
	}

	wb := s.db.NewWriteBatch()
	for _, n := range toDelete {
		if err := wb.Delete(nodeKey(n.ID)); err != nil {
			wb.Cancel()
			return 0, err
		}
		if err := wb.Delete(embKey(n.ID)); err != nil {
			wb.Cancel()
			return 0, err
		}
		if err := s.db.View(func(txn *badgerdb.Txn) error {
			adjEdges, err := adjacentEdges(txn, n.ID)
			if err != nil {
				return err
			}
			for _, e := range adjEdges {
				_ = wb.Delete(edgeKey(e.ID))
				_ = wb.Delete(srcKey(e))
				_ = wb.Delete(dstKey(e))
			}
			return nil
		}); err != nil {
			wb.Cancel()
			return 0, err
		}
	}
	return len(toDelete), wb.Flush()
}

// DeleteByIDs deletes nodes by their IDs, along with their embeddings and
// adjacent edges. Unlike DeleteBySourceURI it does point lookups keyed by ID
// (no full node-space scan per ID), so pruning K findings costs O(K) not
// O(K·N). IDs that do not exist are skipped. Returns the number deleted.
func (s *Store) DeleteByIDs(_ context.Context, ids []string) (int, error) {
	if len(ids) == 0 {
		return 0, nil
	}
	wb := s.db.NewWriteBatch()
	deleted := 0
	for _, id := range ids {
		var present bool
		if err := s.db.View(func(txn *badgerdb.Txn) error {
			if _, err := txn.Get(nodeKey(id)); err != nil {
				if err == badgerdb.ErrKeyNotFound {
					return nil
				}
				return err
			}
			present = true
			adjEdges, err := adjacentEdges(txn, id)
			if err != nil {
				return err
			}
			for _, e := range adjEdges {
				_ = wb.Delete(edgeKey(e.ID))
				_ = wb.Delete(srcKey(e))
				_ = wb.Delete(dstKey(e))
			}
			return nil
		}); err != nil {
			wb.Cancel()
			return 0, err
		}
		if !present {
			continue
		}
		if err := wb.Delete(nodeKey(id)); err != nil {
			wb.Cancel()
			return 0, err
		}
		if err := wb.Delete(embKey(id)); err != nil {
			wb.Cancel()
			return 0, err
		}
		deleted++
	}
	return deleted, wb.Flush()
}

// SearchNodes returns all nodes whose label contains query (case-insensitive).
func (s *Store) SearchNodes(_ context.Context, query string, limit int) ([]model.Node, error) {
	q := strings.ToLower(query)
	var results []model.Node
	err := s.db.View(func(txn *badgerdb.Txn) error {
		pfx := []byte(pfxNode)
		it := txn.NewIterator(badgerdb.DefaultIteratorOptions)
		defer it.Close()
		for it.Seek(pfx); it.ValidForPrefix(pfx); it.Next() {
			var n model.Node
			if err := it.Item().Value(func(v []byte) error {
				return json.Unmarshal(v, &n)
			}); err != nil {
				return err
			}
			if strings.Contains(strings.ToLower(n.Label), q) {
				results = append(results, n)
				if limit > 0 && len(results) >= limit {
					return nil
				}
			}
		}
		return nil
	})
	return results, err
}

// AllNodes returns every node in the graph by scanning the node key space.
func (s *Store) AllNodes(_ context.Context) ([]model.Node, error) {
	var nodes []model.Node
	err := s.db.View(func(txn *badgerdb.Txn) error {
		pfx := []byte(pfxNode)
		it := txn.NewIterator(badgerdb.DefaultIteratorOptions)
		defer it.Close()
		for it.Seek(pfx); it.ValidForPrefix(pfx); it.Next() {
			var n model.Node
			if err := it.Item().Value(func(v []byte) error {
				return json.Unmarshal(v, &n)
			}); err != nil {
				return err
			}
			nodes = append(nodes, n)
		}
		return nil
	})
	return nodes, err
}

// AllEdges returns every edge in the graph by scanning the edge key space.
func (s *Store) AllEdges(_ context.Context) ([]model.Edge, error) {
	var edges []model.Edge
	err := s.db.View(func(txn *badgerdb.Txn) error {
		pfx := []byte(pfxEdge)
		it := txn.NewIterator(badgerdb.DefaultIteratorOptions)
		defer it.Close()
		for it.Seek(pfx); it.ValidForPrefix(pfx); it.Next() {
			var e model.Edge
			if err := it.Item().Value(func(v []byte) error {
				return json.Unmarshal(v, &e)
			}); err != nil {
				return err
			}
			edges = append(edges, e)
		}
		return nil
	})
	return edges, err
}

// UpsertEmbeddings inserts or replaces node embeddings in the v: keyspace.
// Vectors are stored packed (little-endian float32) out of the node JSON so
// graph traversals stay lean. Implements store.EmbeddingStore.
func (s *Store) UpsertEmbeddings(_ context.Context, embs []store.Embedding) error {
	wb := s.db.NewWriteBatch()
	for _, e := range embs {
		if e.NodeID == "" {
			continue
		}
		if err := wb.Set(embKey(e.NodeID), encodeVec(e.Vec)); err != nil {
			wb.Cancel()
			return err
		}
	}
	return wb.Flush()
}

// AllEmbeddings returns every stored embedding by scanning the v: keyspace.
// Implements store.EmbeddingStore.
func (s *Store) AllEmbeddings(_ context.Context) ([]store.Embedding, error) {
	var embs []store.Embedding
	err := s.db.View(func(txn *badgerdb.Txn) error {
		pfx := []byte(pfxEmb)
		it := txn.NewIterator(badgerdb.DefaultIteratorOptions)
		defer it.Close()
		for it.Seek(pfx); it.ValidForPrefix(pfx); it.Next() {
			item := it.Item()
			id := string(item.Key()[len(pfxEmb):])
			var vec []float32
			if err := item.Value(func(v []byte) error {
				vec = decodeVec(v)
				return nil
			}); err != nil {
				return err
			}
			embs = append(embs, store.Embedding{NodeID: id, Vec: vec})
		}
		return nil
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

// GodNodes returns the top-limit nodes ranked by attrs["betweenness"] descending,
// with a stable node-ID tie-break. If communityID is non-nil, only nodes whose
// CommunityID matches are included.
//
// Note: `chiton cluster` writes betweenness (and community) to the cluster
// sidecar, which the live god_nodes MCP tool overlays — it does NOT populate
// attrs["betweenness"] here. So unless a caller has written that attr, every node
// scores 0 and results fall back to the stable ID order. See store.GodNodes.
func (s *Store) GodNodes(_ context.Context, communityID *int, limit int) ([]model.Node, error) {
	var candidates []model.Node
	err := s.db.View(func(txn *badgerdb.Txn) error {
		pfx := []byte(pfxNode)
		it := txn.NewIterator(badgerdb.DefaultIteratorOptions)
		defer it.Close()
		for it.Seek(pfx); it.ValidForPrefix(pfx); it.Next() {
			var n model.Node
			if err := it.Item().Value(func(v []byte) error {
				return json.Unmarshal(v, &n)
			}); err != nil {
				return err
			}
			if communityID != nil && (n.CommunityID == nil || *n.CommunityID != *communityID) {
				continue
			}
			candidates = append(candidates, n)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	// Sort by betweenness descending (stored in attrs["betweenness"]).
	betweenness := func(n model.Node) float64 {
		if n.Attrs == nil {
			return 0
		}
		switch v := n.Attrs["betweenness"].(type) {
		case float64:
			return v
		case float32:
			return float64(v)
		}
		return 0
	}
	sortByBetweenness(candidates, betweenness)

	if limit > 0 && len(candidates) > limit {
		candidates = candidates[:limit]
	}
	return candidates, nil
}

// sortByBetweenness sorts nodes by betweenness descending using a simple
// insertion sort (top-k lists are typically small; avoids importing sort).
// Ties break by node ID ascending so the order is well-defined even when many
// nodes share a score (e.g. the common all-zero case before betweenness is set).
func sortByBetweenness(nodes []model.Node, score func(model.Node) float64) {
	for i := 1; i < len(nodes); i++ {
		key := nodes[i]
		ks := score(key)
		j := i - 1
		for j >= 0 && (score(nodes[j]) < ks || (score(nodes[j]) == ks && nodes[j].ID > key.ID)) {
			nodes[j+1] = nodes[j]
			j--
		}
		nodes[j+1] = key
	}
}

// Snapshot writes a self-contained JSON export of all nodes and edges.
func (s *Store) Snapshot(_ context.Context, w io.Writer) error {
	type snap struct {
		Nodes []model.Node `json:"nodes"`
		Edges []model.Edge `json:"edges"`
	}
	var out snap

	if err := s.db.View(func(txn *badgerdb.Txn) error {
		it := txn.NewIterator(badgerdb.DefaultIteratorOptions)
		defer it.Close()

		// Load embeddings first so they can be attached to their nodes,
		// keeping the export self-contained and portable.
		embByID := map[string][]float32{}
		ep := []byte(pfxEmb)
		for it.Seek(ep); it.ValidForPrefix(ep); it.Next() {
			item := it.Item()
			id := string(item.Key()[len(pfxEmb):])
			if err := item.Value(func(v []byte) error {
				embByID[id] = decodeVec(v)
				return nil
			}); err != nil {
				return err
			}
		}

		pfx := []byte(pfxNode)
		for it.Seek(pfx); it.ValidForPrefix(pfx); it.Next() {
			var n model.Node
			if err := it.Item().Value(func(v []byte) error {
				return json.Unmarshal(v, &n)
			}); err != nil {
				return err
			}
			if vec, ok := embByID[n.ID]; ok {
				n.Embedding = vec
			}
			out.Nodes = append(out.Nodes, n)
		}

		pfx = []byte(pfxEdge)
		for it.Seek(pfx); it.ValidForPrefix(pfx); it.Next() {
			var e model.Edge
			if err := it.Item().Value(func(v []byte) error {
				return json.Unmarshal(v, &e)
			}); err != nil {
				return err
			}
			out.Edges = append(out.Edges, e)
		}
		return nil
	}); err != nil {
		return err
	}

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}
