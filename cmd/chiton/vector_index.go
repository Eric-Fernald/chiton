package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/config"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/vector"
)

// indexReport describes an index buildIndex built, for the serve log.
type indexReport struct {
	Backend string
	Vectors int
	Skipped int
	Build   time.Duration
}

func (r indexReport) String() string {
	s := fmt.Sprintf("%d embeddings, %s index, built in %s", r.Vectors, r.Backend, r.Build.Round(time.Millisecond))
	if r.Skipped > 0 {
		s += fmt.Sprintf(", %d skipped", r.Skipped)
	}
	return s
}

// serveIndex returns the index serve answers from, and a channel closed once it is final. When cfg selects
// HNSW it returns a Flat index at once and builds HNSW in the background from reload, swapping it in when
// done: the build takes about a minute per 100k vectors, and the MCP server must not wait on it to start.
// If the background build fails, serve keeps the exact Flat index. logf reports progress.
func serveIndex(ctx context.Context, cfg config.VectorConfig, embs []store.Embedding,
	reload func(context.Context) ([]store.Embedding, error), logf func(string, ...any)) (vector.Index, <-chan struct{}, error) {
	ready := make(chan struct{})
	backend, err := vector.Resolve(vector.Options{Backend: cfg.Index, Threshold: cfg.HNSWThreshold}, len(embs))
	if err != nil {
		return nil, nil, err
	}
	idx, rep, err := buildIndex(cfg, embs, true) // Flat either way: final, or the stand-in while HNSW builds
	if err != nil {
		return nil, nil, err
	}
	if backend == "flat" {
		logf("serve: semantic_search enabled (%s)", rep)
		close(ready)
		return idx, ready, nil
	}

	sw := vector.NewSwappable(idx)
	logf("serve: semantic_search enabled (%s); building hnsw index over %d embeddings in the background", rep, rep.Vectors)
	go func() {
		defer close(ready)
		fresh, err := reload(ctx)
		if err != nil {
			logf("serve: warning: hnsw build skipped, staying on the flat index: %v", err)
			return
		}
		hcfg := cfg
		hcfg.Index = "hnsw"
		h, hrep, err := buildIndex(hcfg, fresh, false)
		if err != nil {
			logf("serve: warning: hnsw build failed, staying on the flat index: %v", err)
			return
		}
		if ctx.Err() != nil {
			return
		}
		sw.Swap(h)
		logf("serve: hnsw index ready (%s)", hrep)
	}()
	return sw, ready, nil
}

// oneShotExact reports whether a single CLI query should use exact search. Under auto it always should:
// one query never repays an HNSW build. An explicit hnsw setting is honoured.
func oneShotExact(cfg config.VectorConfig) bool {
	return !strings.EqualFold(strings.TrimSpace(cfg.Index), "hnsw")
}

// buildIndex indexes embs with the backend cfg selects, or Flat when exact is set. It clears each
// embs[i].Vec once indexed, since the index keeps its own normalized copy; that way raw and normalized
// vectors are never both fully live. Pass a clone to keep the raw vectors.
func buildIndex(cfg config.VectorConfig, embs []store.Embedding, exact bool) (vector.Index, indexReport, error) {
	opts := vector.Options{
		Backend:   cfg.Index,
		Threshold: cfg.HNSWThreshold,
		HNSW: vector.HNSWConfig{
			Dim:            majorityDim(embs),
			M:              cfg.HNSWM,
			EfSearch:       cfg.HNSWEfSearch,
			EfConstruction: cfg.HNSWEfConstruction,
			Seed:           1,
		},
	}
	if exact {
		opts.Backend = "flat"
	}
	start := time.Now()
	idx, err := vector.NewIndex(opts, len(embs))
	if err != nil {
		return nil, indexReport{}, err
	}
	item := make([]vector.Item, 1)
	for i := range embs {
		item[0] = vector.Item{ID: embs[i].NodeID, Vec: embs[i].Vec}
		idx.Add(item)
		embs[i].Vec = nil
	}
	return idx, indexReport{
		Backend: vector.BackendName(idx),
		Vectors: idx.Len(),
		Skipped: len(embs) - idx.Len(),
		Build:   time.Since(start),
	}, nil
}

// majorityDim returns the most common non-zero vector length in embs, the larger on a tie, so an HNSW
// index keeps the bulk of a store whose embedder changed and skips the stragglers. Flat ignores it.
func majorityDim(embs []store.Embedding) int {
	counts := map[int]int{}
	best := 0
	for _, e := range embs {
		d := len(e.Vec)
		if d == 0 {
			continue
		}
		counts[d]++
		if c, b := counts[d], counts[best]; c > b || (c == b && d > best) {
			best = d
		}
	}
	return best
}
