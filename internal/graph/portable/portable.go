// Package portable implements chiton's interchange format: the self-describing
// JSON snapshot written by GraphStore.Snapshot, and the read/merge side that
// loads one (or several) snapshots back into a store. This is the supported way
// to back up, share, migrate, or merge graphs across repositories.
//
// Cross-repo merging falls out for free: node IDs are stable hashes of
// (source_uri, label) (see model.NewNodeID), so re-importing the same graph is
// idempotent and importing two repos' graphs into one store naturally dedups
// the nodes they share (e.g. an external `fmt` package node).
package portable

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// Snapshot is the on-wire shape: every node (with its embedding inlined, when
// present) and every edge of a graph. It matches the struct GraphStore.Snapshot
// marshals, so a file written by `chiton export` round-trips through Read.
type Snapshot struct {
	Nodes []model.Node `json:"nodes"`
	Edges []model.Edge `json:"edges"`
}

// DefaultBatch is the upsert batch size used by Merge when batch <= 0.
const DefaultBatch = 1000

// Read decodes a Snapshot from r, tolerating a leading UTF-8 BOM (common on
// Windows-authored files).
func Read(r io.Reader) (*Snapshot, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("read snapshot: %w", err)
	}
	data = bytes.TrimPrefix(data, []byte{0xEF, 0xBB, 0xBF})
	var snap Snapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		return nil, fmt.Errorf("parse snapshot: %w", err)
	}
	return &snap, nil
}

// MergeStats reports what a Merge wrote.
type MergeStats struct {
	Nodes      int `json:"nodes"`
	Edges      int `json:"edges"`
	Embeddings int `json:"embeddings"`
}

// Merge upserts a snapshot's nodes and edges into gs in batches of batch
// (DefaultBatch when batch <= 0). Upserts are idempotent and keyed by the
// snapshot's stable IDs, so merging is safe to re-run and dedups shared nodes
// across graphs.
//
// Embeddings inlined on nodes are stripped from the node JSON (to keep the
// store's node records lean, matching the build/embed layout) and written to
// the dedicated embedding keyspace instead — but only when gs implements
// store.EmbeddingStore. Against a store without embedding support, vectors are
// silently dropped (the structural graph still merges correctly).
func Merge(ctx context.Context, gs store.GraphStore, snap *Snapshot, batch int) (MergeStats, error) {
	if batch <= 0 {
		batch = DefaultBatch
	}
	var stats MergeStats

	es, hasEmbed := gs.(store.EmbeddingStore)
	var embs []store.Embedding

	// Upsert nodes in batches, peeling embeddings off into the vector keyspace.
	for start := 0; start < len(snap.Nodes); start += batch {
		end := min(start+batch, len(snap.Nodes))
		chunk := make([]model.Node, 0, end-start)
		for i := start; i < end; i++ {
			n := snap.Nodes[i]
			if len(n.Embedding) > 0 {
				if hasEmbed {
					embs = append(embs, store.Embedding{NodeID: n.ID, Vec: n.Embedding})
				}
				n.Embedding = nil // keep node JSON lean; vectors live in the v: keyspace
			}
			chunk = append(chunk, n)
		}
		if err := gs.UpsertNodes(ctx, chunk); err != nil {
			return stats, fmt.Errorf("upsert nodes: %w", err)
		}
		stats.Nodes += len(chunk)
	}

	// Upsert edges in batches.
	for start := 0; start < len(snap.Edges); start += batch {
		end := min(start+batch, len(snap.Edges))
		if err := gs.UpsertEdges(ctx, snap.Edges[start:end]); err != nil {
			return stats, fmt.Errorf("upsert edges: %w", err)
		}
		stats.Edges += end - start
	}

	// Persist any embeddings we collected.
	for start := 0; start < len(embs); start += batch {
		end := min(start+batch, len(embs))
		if err := es.UpsertEmbeddings(ctx, embs[start:end]); err != nil {
			return stats, fmt.Errorf("upsert embeddings: %w", err)
		}
		stats.Embeddings += end - start
	}

	return stats, nil
}
