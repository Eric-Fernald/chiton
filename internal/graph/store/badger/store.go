// Package badger implements GraphStore using BadgerDB (pure Go, no CGo).
// Used as the fallback when Kuzu CGo bindings are unavailable.
package badger

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
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
)

// Store is a BadgerDB-backed GraphStore.
type Store struct {
	db *badgerdb.DB
}

// Open opens or creates a BadgerDB store at path.
func Open(path string) (*Store, error) {
	opts := badgerdb.DefaultOptions(path).WithLogger(nil)
	db, err := badgerdb.Open(opts)
	if err != nil {
		return nil, fmt.Errorf("open badger at %s: %w", path, err)
	}
	return &Store{db: db}, nil
}

func (s *Store) Close() error { return s.db.Close() }

// --- key builders ---

func nodeKey(id string) []byte { return []byte(pfxNode + id) }
func edgeKey(id string) []byte { return []byte(pfxEdge + id) }

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

	seed, err := s.GetNode(nil, id)
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
			n, err := s.GetNode(nil, nid)
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
		n, err := s.GetNode(nil, src)
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

		pfx := []byte(pfxNode)
		for it.Seek(pfx); it.ValidForPrefix(pfx); it.Next() {
			var n model.Node
			if err := it.Item().Value(func(v []byte) error {
				return json.Unmarshal(v, &n)
			}); err != nil {
				return err
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
