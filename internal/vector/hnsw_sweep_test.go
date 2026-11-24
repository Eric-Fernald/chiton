package vector

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"
)

// TestHNSWSweep prints recall@8 and latency across parameters, for choosing defaults. It is a tool, not a
// check: set CHITON_VECTOR_SWEEP=1, and optionally SWEEP_N, SWEEP_DIM, SWEEP_M, SWEEP_EFC, and
// SWEEP_DATA=iid for unstructured Gaussian vectors instead of embedding-like ones.
func TestHNSWSweep(t *testing.T) {
	if os.Getenv("CHITON_VECTOR_SWEEP") == "" {
		t.Skip("set CHITON_VECTOR_SWEEP=1 to run")
	}
	n, dim := envInt("SWEEP_N", 20000), envInt("SWEEP_DIM", 64)
	m, efc := envInt("SWEEP_M", DefaultHNSWM), envInt("SWEEP_EFC", DefaultHNSWEfConstruction)
	rng := rand.New(rand.NewSource(5))
	var items []Item
	queries := make([][]float32, 200)
	if os.Getenv("SWEEP_DATA") == "iid" {
		items = randomItems(rng, n, dim)
		for i := range queries {
			queries[i] = randomVec(rng, dim)
		}
	} else {
		gen := newEmbeddingLike(rng, dim, 200)
		items, queries = gen.items(n), gen.queries(len(queries))
	}
	f := NewFlat()
	f.Add(items)
	start := time.Now()
	for _, q := range queries {
		f.Search(q, 8)
	}
	flatMs := float64(time.Since(start).Microseconds()) / 1000 / float64(len(queries))

	start = time.Now()
	h := NewHNSW(HNSWConfig{M: m, EfConstruction: efc, Seed: 1})
	h.Add(items)
	fmt.Printf("n=%d dim=%d M=%d efC=%d data=%s build=%s flat=%.3f ms/query\n", n, dim, m, efc,
		os.Getenv("SWEEP_DATA"), time.Since(start).Round(time.Millisecond), flatMs)
	for _, ef := range []int{16, 32, 64, 128, 256} {
		h.g.EfSearch = ef
		start := time.Now()
		for _, q := range queries {
			h.Search(q, 8)
		}
		perQuery := float64(time.Since(start).Microseconds()) / 1000 / float64(len(queries))
		fmt.Printf("  ef=%-4d recall@8=%.3f  %.3f ms/query\n", ef, recallAt(f, h, queries, 8), perQuery)
	}
}

func envInt(name string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(name)); err == nil && v > 0 {
		return v
	}
	return def
}
