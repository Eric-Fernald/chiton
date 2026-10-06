package main

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/config"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/vector"
)

// indexReport describes an index buildIndex built or loadIndex loaded, for the serve log.
type indexReport struct {
	Backend string
	Vectors int
	Skipped int
	Build   time.Duration
	Loaded  bool // read from the saved index rather than built
}

func (r indexReport) String() string {
	verb := "built"
	if r.Loaded {
		verb = "loaded"
	}
	s := fmt.Sprintf("%d embeddings, %s index, %s in %s", r.Vectors, r.Backend, verb, r.Build.Round(time.Millisecond))
	if r.Skipped > 0 {
		s += fmt.Sprintf(", %d skipped", r.Skipped)
	}
	return s
}

// savedIndex locates the HNSW index 'chiton embed' saves beside the graph store. An empty Path means none.
type savedIndex struct {
	Path string
	Key  []byte // the store's encryption key, nil for a plaintext store
}

// savedIndexFor returns where the saved index for the store gs at graphPath lives, sealed to match gs.
func savedIndexFor(graphPath string, gs any) savedIndex {
	s := savedIndex{Path: vector.SidecarPath(graphPath)}
	if k, ok := gs.(interface{ EncryptionKey() []byte }); ok {
		s.Key = k.EncryptionKey()
	}
	return s
}

// serveIndex returns the index serve answers from, and a channel closed once it is final. When cfg selects
// HNSW it loads the index 'chiton embed' saved. Without a current one it returns a Flat index at once and
// builds HNSW in the background from reload, swapping it in when done: the MCP server must not wait on a
// build to start. If the background build fails, serve keeps the exact Flat index. logf reports progress.
func serveIndex(ctx context.Context, cfg config.VectorConfig, embs []store.Embedding,
	reload func(context.Context) ([]store.Embedding, error), saved savedIndex,
	logf func(string, ...any)) (vector.Index, <-chan struct{}, error) {
	ready := make(chan struct{})
	backend, err := vector.Resolve(vector.Options{Backend: cfg.Index, Threshold: cfg.HNSWThreshold}, len(embs))
	if err != nil {
		return nil, nil, err
	}
	var why string
	if backend == "hnsw" {
		h, rep, took, err := loadIndex(saved, cfg, embs)
		if err == nil {
			logf("serve: semantic_search enabled (%s)", rep)
			close(ready)
			return h, ready, nil
		}
		why = loadFailure(err)
		if took { // the failed load released vectors the Flat stand-in needs
			if embs, err = reload(ctx); err != nil {
				return nil, nil, fmt.Errorf("reload embeddings: %w", err)
			}
		}
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
	logf("serve: semantic_search enabled (%s); %s, so building hnsw over %d embeddings in the background "+
		"(run 'chiton embed' to save it)", rep, why, rep.Vectors)
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

// loadFailure says why serve or embed could not use the saved index.
func loadFailure(err error) string {
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return "no saved hnsw index"
	case errors.Is(err, vector.ErrStale):
		return "the saved hnsw index is out of date"
	}
	return fmt.Sprintf("the saved hnsw index is unreadable (%v)", err)
}

// saveIndex makes sure the HNSW index serve would use for es is saved, building and saving it when the
// saved one is missing or stale, and returns it. A store small enough for serve to search exactly needs
// no index, so saveIndex returns nil for it.
func saveIndex(ctx context.Context, cfg config.VectorConfig, es store.EmbeddingStore, saved savedIndex,
	logf func(string, ...any)) (vector.Index, error) {
	embs, err := es.AllEmbeddings(ctx)
	if err != nil {
		return nil, err
	}
	backend, err := vector.Resolve(vector.Options{Backend: cfg.Index, Threshold: cfg.HNSWThreshold}, len(embs))
	if err != nil || backend != "hnsw" {
		return nil, err
	}
	h, rep, took, err := loadIndex(saved, cfg, embs)
	if err == nil {
		logf("embed: saved hnsw index is current (%s)", rep)
		return h, nil
	}
	logf("embed: %s; building it", loadFailure(err))
	if took {
		if embs, err = es.AllEmbeddings(ctx); err != nil {
			return nil, err
		}
	}
	fp := fingerprint(hnswConfig(cfg, embs), embs)
	hcfg := cfg
	hcfg.Index = "hnsw"
	idx, rep, err := buildIndex(hcfg, embs, false)
	if err != nil {
		return nil, err
	}
	if err := vector.SaveHNSW(saved.Path, idx.(*vector.HNSW), fp, saved.Key); err != nil {
		return nil, err
	}
	logf("embed: saved hnsw index to %s (%s)", saved.Path, rep)
	return idx, nil
}

// loadIndex loads the saved HNSW index for embs. It clears each embs[i].Vec it takes, so raw and
// normalized vectors are never both fully live; took reports whether it cleared any, which a load that
// fails partway can leave behind.
func loadIndex(saved savedIndex, cfg config.VectorConfig, embs []store.Embedding) (
	h *vector.HNSW, rep indexReport, took bool, err error) {
	start := time.Now()
	if saved.Path == "" {
		return nil, rep, false, fs.ErrNotExist
	}
	// Fingerprinting reads every vector, so skip it when there is no file to compare against.
	if _, err := os.Stat(saved.Path); err != nil {
		return nil, rep, false, err
	}
	hcfg := hnswConfig(cfg, embs)
	next := 0
	h, err = vector.LoadHNSW(saved.Path, hcfg, fingerprint(hcfg, embs), saved.Key, func(id string) ([]float32, bool) {
		// The index lists nodes in the order they were built from embs, so one forward pass finds them all.
		for ; next < len(embs); next++ {
			if embs[next].NodeID == id {
				v := embs[next].Vec
				embs[next].Vec, took = nil, true
				next++
				return v, true
			}
		}
		return nil, false
	})
	if err != nil {
		return nil, rep, took, err
	}
	for i := range embs {
		embs[i].Vec = nil // the skipped ones too, as buildIndex does
	}
	return h, indexReport{Backend: "hnsw", Vectors: h.Len(), Skipped: len(embs) - h.Len(),
		Build: time.Since(start), Loaded: true}, took, nil
}

// oneShotExact reports whether a single CLI query should use exact search. Under auto it always should:
// one query never repays an HNSW build. An explicit hnsw setting is honoured.
func oneShotExact(cfg config.VectorConfig) bool {
	return !strings.EqualFold(strings.TrimSpace(cfg.Index), "hnsw")
}

// indexChunk is how many embeddings buildIndex hands the index at a time: enough to keep every CPU busy
// in a parallel HNSW build, few enough that the raw vectors are released as it goes.
const indexChunk = 16384

// buildIndex indexes embs with the backend cfg selects, or Flat when exact is set. It clears each
// embs[i].Vec once handed over, since the index keeps its own normalized copy; that way raw and normalized
// vectors are never both fully live. Pass a clone to keep the raw vectors.
func buildIndex(cfg config.VectorConfig, embs []store.Embedding, exact bool) (vector.Index, indexReport, error) {
	opts := vector.Options{Backend: cfg.Index, Threshold: cfg.HNSWThreshold, HNSW: hnswConfig(cfg, embs)}
	if exact {
		opts.Backend = "flat"
	}
	start := time.Now()
	idx, err := vector.NewIndex(opts, len(embs))
	if err != nil {
		return nil, indexReport{}, err
	}
	items := make([]vector.Item, 0, min(indexChunk, len(embs)))
	for lo := 0; lo < len(embs); lo += indexChunk {
		items = items[:0]
		for i := lo; i < min(lo+indexChunk, len(embs)); i++ {
			items = append(items, vector.Item{ID: embs[i].NodeID, Vec: embs[i].Vec})
			embs[i].Vec = nil
		}
		idx.Add(items)
	}
	return idx, indexReport{
		Backend: vector.BackendName(idx),
		Vectors: idx.Len(),
		Skipped: len(embs) - idx.Len(),
		Build:   time.Since(start),
	}, nil
}

// hnswConfig is the HNSW configuration cfg selects for embs.
func hnswConfig(cfg config.VectorConfig, embs []store.Embedding) vector.HNSWConfig {
	return vector.HNSWConfig{
		Dim:            majorityDim(embs),
		M:              cfg.HNSWM,
		EfSearch:       cfg.HNSWEfSearch,
		EfConstruction: cfg.HNSWEfConstruction,
		Seed:           1,
	}
}

// fingerprint is vector.Fingerprint over embs in store order.
func fingerprint(hcfg vector.HNSWConfig, embs []store.Embedding) [sha256.Size]byte {
	return vector.Fingerprint(hcfg, func(yield func(string, []float32) bool) {
		for _, e := range embs {
			if !yield(e.NodeID, e.Vec) {
				return
			}
		}
	})
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
