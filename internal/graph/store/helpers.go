package store

import (
	"context"
	"slices"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// IsReadOnly reports whether gs declares itself read-only (see ReadOnlyAware).
func IsReadOnly(gs GraphStore) bool {
	ro, ok := gs.(ReadOnlyAware)
	return ok && ro.ReadOnly()
}

// FindNode resolves a label substring, plus an optional source-file hint, to one node via PickByFile.
// found is false when nothing matches, and err is the raw search error.
func FindNode(ctx context.Context, gs GraphStore, label, file string) (node model.Node, found bool, err error) {
	limit := 1
	if file != "" {
		limit = 20
	}
	nodes, err := gs.SearchNodes(ctx, label, limit)
	if err != nil || len(nodes) == 0 {
		return model.Node{}, false, err
	}
	return PickByFile(nodes, file), true, nil
}

// PickByFile picks a node whose source_uri contains the file hint, preferring symbols over file
// nodes (a file hint almost always targets a symbol in that file), and falls back to nodes[0].
func PickByFile(nodes []model.Node, file string) model.Node {
	if file == "" {
		return nodes[0]
	}
	lower := strings.ToLower(file)
	var firstMatch *model.Node
	for i := range nodes {
		n := &nodes[i]
		if strings.Contains(strings.ToLower(n.SourceURI), lower) {
			if firstMatch == nil {
				firstMatch = n
			}
			if n.Kind == model.KindSymbol {
				return *n
			}
		}
	}
	if firstMatch != nil {
		return *firstMatch
	}
	return nodes[0]
}

// NodeFilter selects nodes for NodesMatching. Zero-valued fields don't constrain.
type NodeFilter struct {
	Kinds     []model.NodeKind  // node kind is one of these
	SourceURI string            // node source_uri equals this
	Attrs     map[string]string // model.AttrString(attrs, k) == v for every pair (so "" also matches an absent attr)
}

// Match reports whether n satisfies every constraint in f.
func (f NodeFilter) Match(n model.Node) bool {
	if len(f.Kinds) > 0 && !slices.Contains(f.Kinds, n.Kind) {
		return false
	}
	if f.SourceURI != "" && n.SourceURI != f.SourceURI {
		return false
	}
	for k, v := range f.Attrs {
		if model.AttrString(n.Attrs, k) != v {
			return false
		}
	}
	return true
}

// NodesMatching returns the nodes of gs that f matches, in AllNodes order, via NodeScanner when gs has it.
func NodesMatching(ctx context.Context, gs GraphStore, f NodeFilter) ([]model.Node, error) {
	if sc, ok := gs.(NodeScanner); ok {
		return sc.NodesMatching(ctx, f)
	}
	all, err := gs.AllNodes(ctx)
	if err != nil {
		return nil, err
	}
	var out []model.Node
	for _, n := range all {
		if f.Match(n) {
			out = append(out, n)
		}
	}
	return out, nil
}

// GetNodes returns the nodes with the given IDs that exist, in ids order, reading them in one pass via
// NodeGetter when gs has it (else one GetNode per ID).
func GetNodes(ctx context.Context, gs GraphStore, ids []string) ([]model.Node, error) {
	if g, ok := gs.(NodeGetter); ok {
		return g.GetNodes(ctx, ids)
	}
	var out []model.Node
	for _, id := range ids {
		n, err := gs.GetNode(ctx, id)
		if err != nil {
			return out, err
		}
		if n != nil {
			out = append(out, *n)
		}
	}
	return out, nil
}

// EmbeddingIDs returns the node IDs es holds embeddings for, via EmbeddingIDLister when es has it.
func EmbeddingIDs(ctx context.Context, es EmbeddingStore) ([]string, error) {
	if l, ok := es.(EmbeddingIDLister); ok {
		return l.EmbeddingIDs(ctx)
	}
	embs, err := es.AllEmbeddings(ctx)
	if err != nil {
		return nil, err
	}
	ids := make([]string, len(embs))
	for i, e := range embs {
		ids[i] = e.NodeID
	}
	return ids, nil
}

// ReplaceSources makes each update its source's stored content via SourceReplacer, else by DeleteBySourceURI
// plus upserts, which also drops edges other sources attached. It returns stale-node counts per update.
func ReplaceSources(ctx context.Context, gs GraphStore, updates ...SourceUpdate) ([]int, error) {
	if r, ok := gs.(SourceReplacer); ok {
		return r.ReplaceSources(ctx, updates)
	}
	deleted := make([]int, len(updates))
	for i, u := range updates {
		if !u.Fresh {
			n, err := gs.DeleteBySourceURI(ctx, u.URI)
			if err != nil {
				return deleted, err
			}
			deleted[i] = n
		}
		if err := gs.UpsertNodes(ctx, u.Nodes); err != nil {
			return deleted, err
		}
		if err := gs.UpsertEdges(ctx, u.Edges); err != nil {
			return deleted, err
		}
	}
	return deleted, nil
}

// CountByKind tallies gs's nodes and edges by kind, via KindCounter when gs has it.
func CountByKind(ctx context.Context, gs GraphStore) (map[model.NodeKind]int, map[model.EdgeKind]int, error) {
	if c, ok := gs.(KindCounter); ok {
		return c.CountByKind(ctx)
	}
	all, err := gs.AllNodes(ctx)
	if err != nil {
		return nil, nil, err
	}
	nodes := map[model.NodeKind]int{}
	for _, n := range all {
		nodes[n.Kind]++
	}
	allEdges, err := gs.AllEdges(ctx)
	if err != nil {
		return nil, nil, err
	}
	edges := map[model.EdgeKind]int{}
	for _, e := range allEdges {
		edges[e.Kind]++
	}
	return nodes, edges, nil
}
