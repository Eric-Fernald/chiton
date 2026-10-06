package vector

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

const benchDim = 768

// benchSizes is 10k and 100k vectors, plus 1M when CHITON_VECTOR_BENCH_1M is set (about 10 GB of RAM).
func benchSizes() []int {
	sizes := []int{10_000, 100_000}
	if os.Getenv("CHITON_VECTOR_BENCH_1M") != "" {
		sizes = append(sizes, 1_000_000)
	}
	return sizes
}

func benchData(n int) ([]Item, [][]float32) {
	gen := newEmbeddingLike(rand.New(rand.NewSource(11)), benchDim, 200)
	return gen.items(n), gen.queries(500)
}

// BenchmarkSearch times one k=8 query per op for each backend and size, on 768-dimensional
// embedding-like vectors. Index builds happen outside the timed loops.
func BenchmarkSearch(b *testing.B) {
	for _, n := range benchSizes() {
		items, queries := benchData(n)
		for _, backend := range []string{"flat", "hnsw"} {
			idx, err := NewIndex(Options{Backend: backend, HNSW: HNSWConfig{Seed: 1}}, n)
			if err != nil {
				b.Fatal(err)
			}
			idx.Add(items)
			b.Run(fmt.Sprintf("%s/%d", backend, n), func(b *testing.B) {
				for i := 0; b.Loop(); i++ {
					idx.Search(queries[i%len(queries)], 8)
				}
			})
		}
	}
}

// TestVectorBenchReport prints the T5 table for docs/BENCHMARK.md: build time, heap held by the index,
// query latency percentiles, and recall@8 against Flat. Set CHITON_VECTOR_BENCH to a comma-separated
// list of sizes, e.g. 10000,100000,1000000.
func TestVectorBenchReport(t *testing.T) {
	spec := os.Getenv("CHITON_VECTOR_BENCH")
	if spec == "" {
		t.Skip("set CHITON_VECTOR_BENCH=10000,100000 to run")
	}
	fmt.Println("| Vectors | Backend | Build | Heap | Heap / raw | p50 query | p95 query | recall@8 |")
	fmt.Println("|---|---|---|---|---|---|---|---|")
	for field := range strings.SplitSeq(spec, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(field))
		if err != nil {
			t.Fatal(err)
		}
		items, queries := benchData(n)
		raw := float64(n * benchDim * 4)
		var flat Index
		for _, backend := range []string{"flat", "hnsw"} {
			heapBefore := liveHeap()
			start := time.Now()
			idx, _ := NewIndex(Options{Backend: backend, HNSW: HNSWConfig{Seed: 1}}, n)
			idx.Add(items)
			build := time.Since(start)
			heap := float64(liveHeap() - heapBefore)

			// Time each query over enough repeats to fill 5 ms, since the clock on some platforms (Windows)
			// is too coarse for a single sub-millisecond search.
			reps := 1
			for {
				s := time.Now()
				for range reps {
					idx.Search(queries[0], 8)
				}
				if time.Since(s) >= 5*time.Millisecond {
					break
				}
				reps *= 2
			}
			lat := make([]time.Duration, len(queries))
			for i, q := range queries {
				s := time.Now()
				for range reps {
					idx.Search(q, 8)
				}
				lat[i] = time.Since(s) / time.Duration(reps)
			}
			slices.Sort(lat)
			recall := 1.0
			if backend == "flat" {
				flat = idx
			} else {
				recall = recallAt(flat, idx, queries, 8)
			}
			fmt.Printf("| %d | %s | %s | %.0f MB | %.2f | %.3f ms | %.3f ms | %.3f |\n", n, backend,
				build.Round(10*time.Millisecond), heap/1e6, heap/raw, ms(lat[len(lat)/2]), ms(lat[len(lat)*95/100]), recall)
			runtime.KeepAlive(idx)
		}
	}
}

func liveHeap() uint64 {
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.HeapAlloc
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
