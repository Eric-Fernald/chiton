// Package badger implements GraphStore using BadgerDB (pure Go, no CGo).
// It is chiton's working backend. The kuzu package is only a stub.
package badger

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	badgerdb "github.com/dgraph-io/badger/v4"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// Key prefixes. Node IDs and edge IDs are 32-char lowercase hex strings, so the prefix bytes
// ('d','e','k','m','n','s','u','v') never collide with key content (metadata keys are in schema.go).
const (
	pfxNode = "n:" // n:{nodeID}                        → json(Node)
	pfxEdge = "e:" // e:{edgeID}                        → json(Edge)
	pfxSrc  = "s:" // s:{src}:{kind}:{dst}              → edgeID   (forward adjacency)
	pfxDst  = "d:" // d:{dst}:{kind}:{src}              → edgeID   (reverse adjacency)
	pfxEmb  = "v:" // v:{nodeID}                        → embMagic header + packed little-endian []float32 (legacy: bare vector)
	pfxURI  = "u:" // u:{hash(source_uri)}\x00{nodeID}  → empty    (source_uri index)
	pfxKind = "k:" // k:{kind}\x00{nodeID}              → empty    (kind index)
)

// Store is a BadgerDB-backed GraphStore.
type Store struct {
	db       *badgerdb.DB
	readOnly bool
	key      []byte
	lock     *storeLock                   // chiton.lock; nil for a reader once holdReaderLock lets it go
	labels   func() ([]labelEntry, error) // read-only stores: lowercased labels in key order, loaded on first search
	epoch    string                       // random identity minted with the store; see Epoch
}

// Compile-time checks that Store satisfies the store interfaces.
var (
	_ store.GraphStore          = (*Store)(nil)
	_ store.EmbeddingStore      = (*Store)(nil)
	_ store.NodeDeleter         = (*Store)(nil)
	_ store.ReadOnlyAware       = (*Store)(nil)
	_ store.NodeScanner         = (*Store)(nil)
	_ store.NodeGetter          = (*Store)(nil)
	_ store.EmbeddingIDLister   = (*Store)(nil)
	_ store.EmbeddingMetaLister = (*Store)(nil)
	_ store.EmbeddingDeleter    = (*Store)(nil)
	_ store.SourceReplacer      = (*Store)(nil)
	_ store.KindCounter         = (*Store)(nil)
	_ store.Syncer              = (*Store)(nil)
)

// ReadOnly reports whether the store was opened read-only, so callers can fail writes early
// with a clear message instead of a raw Badger error.
func (s *Store) ReadOnly() bool { return s.readOnly }

// EncryptionKey returns the key the store was opened with, or nil for a plaintext store, so files kept
// beside the store can be sealed to match it.
func (s *Store) EncryptionKey() []byte { return s.key }

// Options configures store opens. The zero value is a plaintext store with
// default Badger settings — historic callers can keep using Open/OpenReadOnly.
type Options struct {
	// EncryptionKey, when non-nil, enables Badger's at-rest encryption. It must be 16, 24, or 32 bytes.
	EncryptionKey []byte
}

// Open opens or creates a BadgerDB store at path with read-write access.
func Open(path string) (*Store, error) { return OpenWith(path, Options{}) }

// OpenReadOnly opens an existing store read-only. Readers never block each other; see OpenReadOnlyWith.
func OpenReadOnly(path string) (*Store, error) { return OpenReadOnlyWith(path, Options{}) }

// OpenWith opens (or creates) the store with explicit options, enabling Badger encryption when
// opts.EncryptionKey is set. It returns ErrStoreBusy while another chiton process holds the store, stamps a
// fresh store with the key-layout version, migrates an older layout, and refuses a newer one with ErrSchemaTooNew.
func OpenWith(path string, opts Options) (*Store, error) {
	if err := os.MkdirAll(path, 0o700); err != nil {
		return nil, fmt.Errorf("open badger at %s: %w", path, err)
	}
	lock, err := acquireLock(path, true)
	if err != nil {
		return nil, err
	}
	db, err := badgerdb.Open(badgerOptions(path, opts, false))
	if err != nil {
		lock.release()
		return nil, fmt.Errorf("open badger at %s: %w", path, err)
	}
	s := &Store{db: db, key: opts.EncryptionKey, lock: lock}
	if err := s.prepare(); err != nil {
		s.abandon()
		return nil, err
	}
	return s, nil
}

// OpenReadOnlyWith is the read-only variant of OpenWith. When a crashed writer left a log that a read-only open
// can't replay, or the store has an older key layout, it replays the log and migrates the layout with one
// read-write open and close, then retries.
func OpenReadOnlyWith(path string, opts Options) (*Store, error) {
	s, err := openReadOnly(path, opts)
	if !errors.Is(err, errNeedsWriterPass) {
		return s, err
	}
	rw, rwErr := OpenWith(path, opts)
	if rwErr != nil {
		return nil, fmt.Errorf("%v: %w", err, rwErr)
	}
	if err := rw.Close(); err != nil {
		return nil, err
	}
	return openReadOnly(path, opts)
}

// errNeedsWriterPass marks a read-only open that one read-write open and close would fix.
var errNeedsWriterPass = errors.New("graph store needs a one-time read-write pass")

// openReadOnly opens path read-only under a shared chiton.lock, kept until Close only where holdReaderLock.
func openReadOnly(path string, opts Options) (*Store, error) {
	lock, err := acquireLock(path, false)
	if err != nil {
		return nil, err
	}
	// Check before Badger opens: its failed read-only open leaks the log's mapping, which blocks repair on Windows.
	if hasWriteLog(path) {
		lock.release()
		return nil, fmt.Errorf("%w (a writer did not close cleanly)", errNeedsWriterPass)
	}
	db, err := badgerdb.Open(badgerOptions(path, opts, true))
	if err != nil {
		lock.release()
		if isTruncateNeeded(err) {
			return nil, fmt.Errorf("open badger at %s: a writer started or crashed during the open; retry: %w", path, err)
		}
		return nil, fmt.Errorf("open badger at %s: %w", path, err)
	}
	s := &Store{db: db, readOnly: true, key: opts.EncryptionKey, lock: lock}
	if err := s.prepare(); err != nil {
		s.abandon()
		return nil, err
	}
	if !holdReaderLock {
		s.lock.release()
		s.lock = nil
	}
	return s, nil
}

// prepare readies a just-opened store: it checks, and when writable stamps or migrates, the key layout, loads
// the store epoch, and gives a read-only store its lazy label index.
func (s *Store) prepare() error {
	if err := prepareSchema(s.db, s.readOnly); err != nil {
		return err
	}
	var err error
	if s.epoch, err = loadEpoch(s.db, s.readOnly); err != nil {
		return err
	}
	if s.readOnly {
		s.labels = sync.OnceValues(func() ([]labelEntry, error) { return loadLabels(s.db) })
	}
	return nil
}

// Epoch returns the store's random identity, minted the first time the store is opened writable, so a sidecar
// can tell that the store was replaced underneath it. It is empty only for a store opened read-only that no
// writer has minted one for yet.
func (s *Store) Epoch() string { return s.epoch }

// Sync makes every write so far durable; it is a no-op on a read-only store. Implements store.Syncer.
func (s *Store) Sync() error { return s.db.Sync() }

// abandon closes a store whose open failed after Badger opened, dropping close errors so the caller reports
// the open's.
func (s *Store) abandon() {
	s.db.Close()
	s.lock.release()
}

// valueLogFileSize caps each value-log file. Badger preallocates twice this while a writer is open, and chiton's
// values stay inline in the LSM tree, so the 1 GiB default only left 2 GiB files behind after a crash.
const valueLogFileSize = 64 << 20

// badgerOptions returns the Badger options for path. Read-only opens bypass Badger's directory lock, which
// Windows can't take shared; chiton.lock coordinates readers and writers instead.
func badgerOptions(path string, opts Options, readOnly bool) badgerdb.Options {
	bopts := badgerdb.DefaultOptions(path).WithLogger(nil).WithValueLogFileSize(valueLogFileSize)
	if readOnly {
		bopts = bopts.WithReadOnly(true).WithBypassLockGuard(true)
	}
	return applyEncryption(bopts, opts.EncryptionKey)
}

// isTruncateNeeded reports Badger's "Log truncate required" error. Outside debug builds Badger wraps it with
// %+v, not %w, so errors.Is can't see it.
func isTruncateNeeded(err error) bool {
	return err != nil && strings.Contains(err.Error(), badgerdb.ErrTruncateNeeded.Error())
}

// applyEncryption enables Badger encryption when key is non-empty, plus the non-zero IndexCacheSize that
// Badger then requires.
func applyEncryption(bopts badgerdb.Options, key []byte) badgerdb.Options {
	if len(key) == 0 {
		return bopts
	}
	return bopts.
		WithEncryptionKey(key).
		WithIndexCacheSize(64 << 20)
}

// collectValueLog runs bounded value-log GC before a writable store closes. Values rarely reach the log, so
// this usually stops at once with ErrNoRewrite.
func (s *Store) collectValueLog() {
	for range 4 {
		if s.db.RunValueLogGC(0.5) != nil {
			return // ErrNoRewrite (nothing to reclaim) or ErrRejected
		}
	}
}

// Close collects and seals a writable store (see sealed), then releases it. GC runs first because its
// rewrites are commits the seal must cover.
func (s *Store) Close() error {
	var sealErr error
	if !s.readOnly {
		s.collectValueLog()
		sealErr = seal(s.db)
	}
	return errors.Join(sealErr, s.db.Close(), s.lock.release())
}

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

// prefixIterator iterates the keys under pfx without prefetching values. Values are small and inline in
// the LSM tree, so prefetching would only add overhead.
func prefixIterator(txn *badgerdb.Txn, pfx string) *badgerdb.Iterator {
	opts := badgerdb.DefaultIteratorOptions
	opts.Prefix = []byte(pfx)
	opts.PrefetchValues = false
	return txn.NewIterator(opts)
}

// scan calls fn, in key order, with the key suffix and value of every entry under pfx. Both are valid only
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

// adjacentEdgeKeys parses nodeID's adjacency keys into its edges (only ID, Src, Dst, and Kind), deduplicated
// by edge ID, with outgoing edges first and each direction in key order. A nil kinds keeps every kind.
func adjacentEdgeKeys(txn *badgerdb.Txn, nodeID string, kinds map[model.EdgeKind]bool) ([]model.Edge, error) {
	var edges []model.Edge
	seen := map[string]bool{}
	for _, out := range [...]bool{true, false} {
		pfx := pfxDst + nodeID + ":"
		if out {
			pfx = pfxSrc + nodeID + ":"
		}
		// The key suffix is {kind}:{other endpoint}. Split at the last ':', because node IDs contain none (the
		// prefix scan relies on that) but an imported edge kind may.
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

// deleteEdge queues deletion of edge e's record and both adjacency keys.
func deleteEdge(wb *badgerdb.WriteBatch, e *model.Edge) error {
	for _, k := range [][]byte{edgeKey(e.ID), srcKey(e), dstKey(e)} {
		if err := wb.Delete(k); err != nil {
			return err
		}
	}
	return nil
}

// deleteNode queues deletion of node id (whose stored indexed fields are x): its record, embedding, index keys,
// and every adjacent edge with its adjacency keys.
func deleteNode(txn *badgerdb.Txn, wb *badgerdb.WriteBatch, id string, x indexed) error {
	adj, err := adjacentEdgeKeys(txn, id, nil)
	if err != nil {
		return err
	}
	for _, k := range [][]byte{nodeKey(id), embKey(id), uriKey(x.SourceURI, id), kindKey(x.Kind, id)} {
		if err := wb.Delete(k); err != nil {
			return err
		}
	}
	for i := range adj {
		if err := deleteEdge(wb, &adj[i]); err != nil {
			return err
		}
	}
	return nil
}

// putNodes queues nodes and their index keys into wb, stamping CreatedAt when unset and UpdatedAt always, and
// drops the keys a changed kind or source_uri left behind. txn is the batch's read view, which still shows the
// records the nodes replace.
func putNodes(txn *badgerdb.Txn, wb *badgerdb.WriteBatch, nodes []model.Node, now time.Time) error {
	written := map[string]indexed{} // this batch's earlier writes, which txn can't see
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
		prev, had := written[n.ID]
		if !had {
			old, err := loadJSON[indexed](txn, nodeKey(n.ID))
			if err != nil {
				return err
			}
			if old != nil {
				prev, had = *old, true
			}
		}
		cur := indexedOf(n)
		written[n.ID] = cur
		if err := setIndex(wb, n.ID, cur); err != nil {
			return err
		}
		if err := wb.Set(nodeKey(n.ID), data); err != nil {
			return err
		}
		if had {
			if err := dropStaleIndex(wb, n.ID, prev, cur); err != nil {
				return err
			}
		}
	}
	return nil
}

// putEdges queues edges and their adjacency keys into wb, stamping CreatedAt when unset.
func putEdges(wb *badgerdb.WriteBatch, edges []model.Edge, now time.Time) error {
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
}

// writeBatch runs fill against a new write batch on db, flushing it on success and discarding it on error.
func writeBatch(db *badgerdb.DB, fill func(wb *badgerdb.WriteBatch) error) error {
	wb := db.NewWriteBatch()
	if err := fill(wb); err != nil {
		wb.Cancel()
		return err
	}
	return wb.Flush()
}

func (s *Store) batch(fill func(wb *badgerdb.WriteBatch) error) error { return writeBatch(s.db, fill) }

// --- GraphStore implementation ---

func (s *Store) UpsertNodes(_ context.Context, nodes []model.Node) error {
	now := time.Now()
	return s.batch(func(wb *badgerdb.WriteBatch) error {
		return s.db.View(func(txn *badgerdb.Txn) error { return putNodes(txn, wb, nodes, now) })
	})
}

func (s *Store) UpsertEdges(_ context.Context, edges []model.Edge) error {
	now := time.Now()
	return s.batch(func(wb *badgerdb.WriteBatch) error { return putEdges(wb, edges, now) })
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

// ShortestPath returns the shortest undirected path between src and dst via BFS over adjacency keys, loading
// only the path's nodes and edges. Returns an empty Path (no error) when no path exists.
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
			adj, err := adjacentEdgeKeys(txn, curr, nil)
			if err != nil {
				return err
			}
			for _, e := range adj {
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

// attrProbe returns raw-bytes probes for f's attrs; a record matching f contains every one. Empty values
// (which also match an absent attr) and values whose escaping could vary get none.
func attrProbe(f store.NodeFilter) [][]byte {
	var probes [][]byte
	for _, v := range f.Attrs {
		if b := stringProbe(v); v != "" && b != nil {
			probes = append(probes, b)
		}
	}
	return probes
}

// NodesMatching returns the nodes matching f in AllNodes order. A source_uri or kind constraint loads only the
// index's candidates; otherwise it scans, decoding records that pass the attr probe. Implements store.NodeScanner.
func (s *Store) NodesMatching(_ context.Context, f store.NodeFilter) ([]model.Node, error) {
	var nodes []model.Node
	keep := func(n model.Node) {
		if f.Match(n) {
			nodes = append(nodes, n)
		}
	}
	err := s.db.View(func(txn *badgerdb.Txn) error {
		ids, ok, err := candidateIDs(txn, f)
		if err != nil {
			return err
		}
		if ok {
			for _, id := range ids {
				n, err := loadNode(txn, id)
				if err != nil {
					return err
				}
				if n != nil {
					keep(*n)
				}
			}
			return nil
		}
		probes := attrProbe(f)
		return scan(txn, pfxNode, func(_, v []byte) error {
			if slices.ContainsFunc(probes, func(p []byte) bool { return !bytes.Contains(v, p) }) {
				return nil
			}
			var n model.Node
			if err := json.Unmarshal(v, &n); err != nil {
				return err
			}
			keep(n)
			return nil
		})
	})
	return nodes, err
}

// sourceIDs returns the IDs the source_uri index lists under each of uris, keyed by URI. A digest collision
// could list a foreign node, so callers check the loaded record's source_uri.
func sourceIDs(txn *badgerdb.Txn, uris []string) (map[string][]string, error) {
	out := make(map[string][]string, len(uris))
	for _, u := range uris {
		if _, done := out[u]; done {
			continue
		}
		ids, err := keySuffixes(txn, uriPrefix(u))
		if err != nil {
			return nil, err
		}
		out[u] = ids
	}
	return out, nil
}

// DeleteBySourceURI removes the nodes whose source_uri is uri, found through the source_uri index, and all
// edges adjacent to them. Returns the number of nodes deleted.
func (s *Store) DeleteBySourceURI(_ context.Context, uri string) (int, error) {
	deleted := 0
	err := s.batch(func(wb *badgerdb.WriteBatch) error {
		return s.db.View(func(txn *badgerdb.Txn) error {
			byURI, err := sourceIDs(txn, []string{uri})
			if err != nil {
				return err
			}
			for _, id := range byURI[uri] {
				x, err := loadJSON[indexed](txn, nodeKey(id))
				if err != nil {
					return err
				}
				if x == nil || x.SourceURI != uri {
					continue // stale index key or a digest collision
				}
				if err := deleteNode(txn, wb, id, *x); err != nil {
					return err
				}
				deleted++
			}
			return nil
		})
	})
	return deleted, err
}

// ReplaceSources applies each update as its source's complete new extraction in one write batch. Implements
// store.SourceReplacer; see replaceSource for what survives.
func (s *Store) ReplaceSources(_ context.Context, updates []store.SourceUpdate) ([]int, error) {
	deleted := make([]int, len(updates))
	now := time.Now()
	err := s.batch(func(wb *badgerdb.WriteBatch) error {
		return s.db.View(func(txn *badgerdb.Txn) error {
			var uris []string
			for _, u := range updates {
				if !u.Fresh {
					uris = append(uris, u.URI)
				}
			}
			prior, err := sourceIDs(txn, uris)
			if err != nil {
				return err
			}
			var nodes []model.Node
			var edges []model.Edge
			for i, u := range updates {
				n, kept, err := replaceSource(txn, wb, u, prior[u.URI])
				if err != nil {
					return err
				}
				deleted[i] = n
				nodes = append(nodes, kept...)
				edges = append(edges, u.Edges...)
			}
			// Every delete is queued before any write, so a later Set for the same key wins.
			if err := putNodes(txn, wb, nodes, now); err != nil {
				return err
			}
			return putEdges(wb, edges, now)
		})
	})
	return deleted, err
}

// replaceSource queues deletion of prior nodes u drops (with every edge) and of owned edges u no longer emits,
// and returns u.Nodes with unchanged nodes' extraction_status carried over. Changed nodes lose their vector.
func replaceSource(txn *badgerdb.Txn, wb *badgerdb.WriteBatch, u store.SourceUpdate, prior []string) (int, []model.Node, error) {
	nodes := u.Nodes
	if len(prior) == 0 {
		return 0, nodes, nil
	}
	index := make(map[string]int, len(nodes))
	for i := range nodes {
		index[nodes[i].ID] = i
	}
	keepEdge := make(map[string]bool, len(u.Edges))
	for i := range u.Edges {
		keepEdge[u.Edges[i].ID] = true
	}
	deleted, cloned := 0, false
	for _, id := range prior {
		old, err := loadNode(txn, id)
		if err != nil {
			return 0, nil, err
		}
		if old == nil || old.SourceURI != u.URI {
			continue // stale index key or a digest collision
		}
		i, ok := index[id]
		if !ok {
			if err := deleteNode(txn, wb, id, indexed{Kind: old.Kind, SourceURI: old.SourceURI}); err != nil {
				return 0, nil, err
			}
			deleted++
			continue
		}
		if err := dropDisownedEdges(txn, wb, id, keepEdge); err != nil {
			return 0, nil, err
		}
		if !sameContent(old, &nodes[i]) {
			// Its embedded text may have changed, so drop the vector for `chiton embed` to redo.
			if err := wb.Delete(embKey(id)); err != nil {
				return 0, nil, err
			}
			continue
		}
		st := model.AttrString(old.Attrs, statusAttr)
		if st == "" || model.AttrString(nodes[i].Attrs, statusAttr) != "" {
			continue
		}
		if !cloned {
			nodes, cloned = slices.Clone(nodes), true
		}
		attrs := maps.Clone(nodes[i].Attrs)
		if attrs == nil {
			attrs = map[string]any{}
		}
		attrs[statusAttr] = st
		nodes[i].Attrs = attrs
	}
	return deleted, nodes, nil
}

// dropDisownedEdges queues deletion of edges leaving node id that keep omits and that id's source owns. A source
// owns its nodes' outgoing EXTRACTED edges; linkage, concept, and finding edges are INFERRED or start elsewhere.
func dropDisownedEdges(txn *badgerdb.Txn, wb *badgerdb.WriteBatch, id string, keep map[string]bool) error {
	adj, err := adjacentEdgeKeys(txn, id, nil)
	if err != nil {
		return err
	}
	for i := range adj {
		if adj[i].Src != id || keep[adj[i].ID] {
			continue
		}
		e, err := loadJSON[model.Edge](txn, edgeKey(adj[i].ID))
		if err != nil {
			return err
		}
		if e != nil && e.Provenance != model.ProvExtracted {
			continue
		}
		if err := deleteEdge(wb, &adj[i]); err != nil {
			return err
		}
	}
	return nil
}

// statusAttr is the attr the subagent flows set once they have processed a node. It is state, not content.
const statusAttr = "extraction_status"

// sameContent reports whether a and b agree on kind, label, and every attr but statusAttr: the inputs to a
// node's embedded text and to its subagent task.
func sameContent(a, b *model.Node) bool {
	return a.Kind == b.Kind && a.Label == b.Label && bytes.Equal(contentAttrs(a.Attrs), contentAttrs(b.Attrs))
}

// contentAttrs encodes attrs minus statusAttr as JSON. Sorted keys and uniform number encoding make a decoded
// stored map compare equal to a fresh one.
func contentAttrs(attrs map[string]any) []byte {
	m := make(map[string]any, len(attrs))
	for k, v := range attrs {
		if k != statusAttr {
			m[k] = v
		}
	}
	b, _ := json.Marshal(m)
	return b
}

// DeleteByIDs deletes nodes by ID plus their embeddings, adjacent edges, and index keys, skipping missing IDs.
// Point lookups keep it O(K) in the number of IDs.
func (s *Store) DeleteByIDs(_ context.Context, ids []string) (int, error) {
	if len(ids) == 0 {
		return 0, nil
	}
	deleted := 0
	err := s.batch(func(wb *badgerdb.WriteBatch) error {
		return s.db.View(func(txn *badgerdb.Txn) error {
			n := 0
			for _, id := range ids {
				x, err := loadJSON[indexed](txn, nodeKey(id))
				if err != nil {
					return err
				}
				if x == nil {
					continue
				}
				if err := deleteNode(txn, wb, id, *x); err != nil {
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

// labelEntry is one node in a read-only store's label index.
type labelEntry struct{ lower, id string }

// labelOnly decodes just a node record's label.
type labelOnly struct {
	Label string `json:"label"`
}

// SearchNodes returns the nodes whose label contains query (case-insensitive), in key order. A read-only
// store never changes, so it matches against a label index built on first use; a writable one scans.
func (s *Store) SearchNodes(ctx context.Context, query string, limit int) ([]model.Node, error) {
	if s.labels == nil {
		return s.scanLabels(query, limit)
	}
	idx, err := s.labels()
	if err != nil {
		return nil, err
	}
	q := strings.ToLower(query)
	var ids []string
	for _, e := range idx {
		if !strings.Contains(e.lower, q) {
			continue
		}
		ids = append(ids, e.id)
		if limit > 0 && len(ids) >= limit {
			break
		}
	}
	return s.GetNodes(ctx, ids)
}

// scanLabels is SearchNodes over the live key space: it decodes each label, then only matching records in full.
func (s *Store) scanLabels(query string, limit int) ([]model.Node, error) {
	q := strings.ToLower(query)
	var results []model.Node
	err := s.db.View(func(txn *badgerdb.Txn) error {
		return scan(txn, pfxNode, func(_, v []byte) error {
			var probe labelOnly
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

// loadLabels reads every node's lowercased label, in key order, decoding only the label.
func loadLabels(db *badgerdb.DB) ([]labelEntry, error) {
	var idx []labelEntry
	err := db.View(func(txn *badgerdb.Txn) error {
		return scan(txn, pfxNode, func(k, v []byte) error {
			var probe labelOnly
			if err := json.Unmarshal(v, &probe); err != nil {
				return err
			}
			idx = append(idx, labelEntry{lower: strings.ToLower(probe.Label), id: string(k)})
			return nil
		})
	})
	return idx, err
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

// CountByKind tallies nodes from the kind index and edges from their forward adjacency keys, reading keys only.
// Implements store.KindCounter.
func (s *Store) CountByKind(_ context.Context) (map[model.NodeKind]int, map[model.EdgeKind]int, error) {
	nodes, edges := map[model.NodeKind]int{}, map[model.EdgeKind]int{}
	err := s.db.View(func(txn *badgerdb.Txn) error {
		if err := eachKey(txn, pfxKind, func(k []byte) error {
			if i := bytes.LastIndexByte(k, 0); i >= 0 {
				nodes[model.NodeKind(k[:i])]++
			}
			return nil
		}); err != nil {
			return err
		}
		return eachKey(txn, pfxSrc, func(k []byte) error {
			// k is {src}:{kind}:{dst}. Node IDs hold no ':', but an imported edge kind may.
			if i, j := bytes.IndexByte(k, ':'), bytes.LastIndexByte(k, ':'); i >= 0 && j > i {
				edges[model.EdgeKind(k[i+1:j])]++
			}
			return nil
		})
	})
	return nodes, edges, err
}

// UpsertEmbeddings stores each vector with its model and text hash in the v: keyspace, out of the node JSON, so
// graph traversals stay lean. Implements store.EmbeddingStore.
func (s *Store) UpsertEmbeddings(_ context.Context, embs []store.Embedding) error {
	return s.batch(func(wb *badgerdb.WriteBatch) error {
		for _, e := range embs {
			if e.NodeID == "" {
				continue
			}
			val, err := encodeEmb(e)
			if err != nil {
				return err
			}
			if err := wb.Set(embKey(e.NodeID), val); err != nil {
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
	err := s.db.View(func(txn *badgerdb.Txn) (err error) {
		ids, err = keySuffixes(txn, pfxEmb)
		return err
	})
	return ids, err
}

// AllEmbeddings returns every stored embedding by scanning the v: keyspace. Implements store.EmbeddingStore.
func (s *Store) AllEmbeddings(_ context.Context) ([]store.Embedding, error) {
	var embs []store.Embedding
	err := s.db.View(func(txn *badgerdb.Txn) error {
		return scan(txn, pfxEmb, func(k, v []byte) error {
			embs = append(embs, decodeEmb(string(k), v))
			return nil
		})
	})
	return embs, err
}

// EmbeddingMetas lists every stored embedding's metadata in key order, parsing headers only.
// Implements store.EmbeddingMetaLister.
func (s *Store) EmbeddingMetas(_ context.Context) ([]store.EmbeddingMeta, error) {
	var metas []store.EmbeddingMeta
	err := s.db.View(func(txn *badgerdb.Txn) error {
		return scan(txn, pfxEmb, func(k, v []byte) error {
			m := store.EmbeddingMeta{NodeID: string(k)}
			if modelID, sha, off, ok := embHeader(v); ok {
				m.Model, m.TextSHA, m.Dim = modelID, sha, (len(v)-off)/4
			}
			metas = append(metas, m)
			return nil
		})
	})
	return metas, err
}

// DeleteEmbeddings removes the vectors of ids, keeping their nodes. Implements store.EmbeddingDeleter.
func (s *Store) DeleteEmbeddings(_ context.Context, ids []string) error {
	return s.batch(func(wb *badgerdb.WriteBatch) error {
		for _, id := range ids {
			if err := wb.Delete(embKey(id)); err != nil {
				return err
			}
		}
		return nil
	})
}

// embMagic opens a tagged embedding value. As a float32 it is a NaN, which no stored vector starts with, so a
// value without it is a legacy bare vector whose model is unknown.
const embMagic uint32 = 0x7FC04301

// encodeEmb packs e as embMagic, the length-prefixed model id and raw text SHA, zero padding to a 4-byte boundary,
// then the vector as little-endian float32s.
func encodeEmb(e store.Embedding) ([]byte, error) {
	sha, err := hex.DecodeString(e.TextSHA)
	if err != nil || len(e.Model) > 255 || len(sha) > 255 {
		return nil, fmt.Errorf("embedding %s: model id over 255 bytes or text SHA not short hex", e.NodeID)
	}
	hdr := (4 + 1 + len(e.Model) + 1 + len(sha) + 3) &^ 3
	b := make([]byte, hdr+4*len(e.Vec))
	binary.LittleEndian.PutUint32(b, embMagic)
	b[4] = byte(len(e.Model))
	n := 5 + copy(b[5:], e.Model)
	b[n] = byte(len(sha))
	copy(b[n+1:], sha)
	putVec(b[hdr:], e.Vec)
	return b, nil
}

// embHeader parses an embedding value's header: its model id, hex text SHA and the vector's byte offset. A legacy
// bare vector has none (off 0). ok is false for a malformed value.
func embHeader(b []byte) (modelID, textSHA string, off int, ok bool) {
	if len(b) < 4 || binary.LittleEndian.Uint32(b) != embMagic {
		return "", "", 0, len(b)%4 == 0
	}
	p := 4
	if p >= len(b) {
		return "", "", 0, false
	}
	ml := int(b[p])
	p++
	if p+ml >= len(b) {
		return "", "", 0, false
	}
	modelID = string(b[p : p+ml])
	p += ml
	sl := int(b[p])
	p++
	if p+sl > len(b) {
		return "", "", 0, false
	}
	textSHA = hex.EncodeToString(b[p : p+sl])
	off = (p + sl + 3) &^ 3
	if off > len(b) || (len(b)-off)%4 != 0 {
		return "", "", 0, false
	}
	return modelID, textSHA, off, true
}

// decodeEmb reverses encodeEmb and also reads legacy bare vectors (model ""). A malformed value yields a nil Vec.
func decodeEmb(id string, b []byte) store.Embedding {
	modelID, sha, off, ok := embHeader(b)
	if !ok {
		return store.Embedding{NodeID: id}
	}
	return store.Embedding{NodeID: id, Vec: decodeVec(b[off:]), Model: modelID, TextSHA: sha}
}

// putVec writes vec into b as little-endian float32s (4 bytes per value).
func putVec(b []byte, vec []float32) {
	for i, f := range vec {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
}

// decodeVec reads little-endian float32s, as putVec writes them. A length not divisible by 4 yields nil.
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
		// Attach embeddings to their nodes so the export is self-contained and portable.
		embByID := map[string][]float32{}
		if err := scan(txn, pfxEmb, func(k, v []byte) error {
			embByID[string(k)] = decodeEmb(string(k), v).Vec
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
