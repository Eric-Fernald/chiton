# chiton Benchmark — token cost per query across corpora

This is the honest, reproducible benchmark the plan calls for: token cost per
query across corpora of varying size and shape, published as a **table, not a
single hero number**. It deliberately includes the cases where chiton *loses* —
and on raw token count vs `grep+read`, that is more often than the marketing
version of this project would like.

Generated with `chiton bench` (built into the binary). It is the Phase 5
cross-corpus successor to the Phase 0 micro-study.

## What is measured

For a corpus, chiton builds an **in-memory** graph from its AST extractors (no
store, no running server), then answers a deterministic set of dependency
questions — *"what does X depend on / connect to?"* — three ways, and reports
the token cost of each:

| Strategy | Models | Cost = tokens of… |
|---|---|---|
| **naive** | a model with no tooling reading the whole scope | every file in the corpus |
| **grep+read** | the grep_search + read_file default of Copilot / Claude Code | every file that mentions the symbol |
| **chiton** | chiton's MCP `get_node_neighborhood` payload | the lean node/edge JSON returned (depth 2) |

Methodology notes (so the numbers are honest, not flattering):

- **Token estimate = characters ÷ 4**, applied identically to all three
  strategies — the same rough estimate the Phase 0 benchmark used.
- **The chiton payload is byte-faithful to the real MCP output.** The bench
  serialises the *exact* `leanNode`/`leanEdge` shape `internal/mcp/server.go`
  returns — `kind`, `label`, `source_uri`, `span`, `attrs`, `provenance`,
  `confidence`, edge `attrs`, and the neighborhood `summary` line — so the
  chiton column is not under-counted. (An earlier draft of this benchmark
  dropped `attrs`/`provenance`/`confidence` and thereby flattered chiton; that
  bias is fixed, which is why the numbers below are higher than a naive lean
  encoding would show.)
- **Query targets** are the highest-degree **corpus-internal** symbols. External
  package nodes (`fmt`, `http.Get`) are excluded: nobody asks "what does `fmt`
  depend on," and their sky-high degree would inflate every column.
- **grep+read greps the base identifier** a developer would actually type — for
  the method label `LinkService.Submit` it greps `Submit`, which (faithfully)
  matches many files. This captures grep's imprecision rather than hiding it.

### What this does NOT measure

- **Answer quality.** This counts retrieval tokens, not whether the model got
  the right answer. chiton's typed edges (`CALLS`, `IMPORTS`, …) also reduce the
  *reasoning* the model must do over raw file dumps — the real benefit, and one
  this table cannot show.
- **Prompt caching.** Naive/grep contexts are far more cacheable than chiton's
  per-query neighborhoods; with a warm cache the input-token gap narrows further
  (see PLAN.md "Query-Cost Comparison").
- **Output tokens.** At $25/M, output often dominates the bill regardless of
  retrieval strategy.

## Reproduce

```sh
chiton bench --synth 40                       # reproducible synthetic ring corpus
chiton bench ./internal/graph                 # a real medium, densely-connected package
chiton bench ./internal --queries 6           # the whole internal/ tree (large)
chiton bench <path> --json                    # machine-readable output
```

`--synth N` generates a fixed N-file Go corpus into a temp dir, so the small end
is identical on every machine.

## Results

Measured on the chiton repo at the `hnsw` branch (Go, `windows/amd64`,
2026-10-06). Corpus sizes drift as the repo grows; re-run `chiton bench` to
refresh.

### A — synthetic ring (40 files) — the case chiton loses badly

40 files · 280 nodes · 240 edges · ~2 391 tokens in scope · depth 2

| Question | naive | grep+read | chiton | chiton vs grep+read |
|---|---:|---:|---:|---|
| What does Func10 depend on / connect to? | 2 391 | 120 | 4 609 | 38× costlier |
| What does Func8 depend on / connect to? | 2 391 | 118 | 4 608 | 39× costlier |
| What does Func33 depend on / connect to? | 2 391 | 120 | 4 609 | 38× costlier |
| What does Func9 depend on / connect to? | 2 391 | 119 | 4 608 | 39× costlier |
| What does Func15 depend on / connect to? | 2 391 | 120 | 4 609 | 38× costlier |

**Median:** chiton ~4 609 tokens vs grep+read ~120 (**38× costlier**); naive ~2 391.

On a tiny corpus where each symbol is defined once and grep lands precisely on a
small file, **grep wins decisively** — and chiton's depth-2 neighborhood of a
ring node costs more than reading the whole corpus would. Don't reach for graph
memory on a hobby-sized repo.

### B — `internal/graph` (medium, densely connected)

26 files · 595 nodes · 735 edges · ~29 750 tokens in scope · depth 2

| Question | naive | grep+read | chiton | chiton vs grep+read |
|---|---:|---:|---:|---|
| What does makeNode depend on / connect to? | 29 750 | 2 770 | 1 880 | 1.5× cheaper |
| What does node depend on / connect to? | 29 750 | 26 570 | 1 403 | **19× cheaper** |
| What does TestTwoCliques depend on / connect to? | 29 750 | 1 026 | 3 027 | 3.0× costlier |
| What does sameComm depend on / connect to? | 29 750 | 1 744 | 2 013 | 1.2× costlier |
| What does makeEdge depend on / connect to? | 29 750 | 2 770 | 2 149 | 1.3× cheaper |

**Median:** chiton ~2 013 tokens vs grep+read ~2 770 (**1.4× cheaper**); naive ~29 750 (**~15× cheaper**).

For a hub symbol like `node` (referenced everywhere), grep reads nearly the whole
package (27k tokens) while chiton returns a bounded ~1.4k-token neighborhood —
**19× cheaper**. But for uniquely-named or weakly-connected symbols, grep is more
precise and chiton's depth-2 neighborhood can cost more. It nets out to
**modestly cheaper** vs grep, and a large win vs naive.

### C — `internal/` (large, 7.2k nodes)

285 files · 7 170 nodes · 7 953 edges · ~314 988 tokens in scope · depth 2

| Question | naive | grep+read | chiton | chiton vs grep+read |
|---|---:|---:|---:|---|
| What does LinkService.Submit depend on / connect to? | 314 988 | 40 340 | 34 178 | 1.2× cheaper |
| What does semanticSearchTool depend on / connect to? | 314 988 | 7 221 | 6 421 | 1.1× cheaper |
| What does Generate depend on / connect to? | 314 988 | 16 749 | 9 294 | 1.8× cheaper |
| What does TestVectorBenchReport depend on / connect to? | 314 988 | 839 | 16 947 | 20× costlier |
| What does countFindings depend on / connect to? | 314 988 | 7 861 | 30 772 | 3.9× costlier |
| What does Service.Submit depend on / connect to? | 314 988 | 40 340 | 29 546 | 1.4× cheaper |

**Median:** chiton ~23 246 tokens vs grep+read ~12 305 (**1.9× costlier**); naive ~314 988 (**~14× cheaper**).

At scale, naive context dumping is hopeless (300k+ tokens). Against grep+read
the picture has shifted since the last run: `internal/` grew from 4.6k to 7.2k
nodes, and two of the six highest-degree symbols are now test code:
`TestVectorBenchReport`, and `countFindings`, a helper defined in three test
files. grep lands on a file or three while chiton's depth-2 neighborhood of
these busy test symbols is wide. The
hub symbols still favour chiton. The reliable, large win is over **naive**
(~14–15×).

## Honest summary

- **vs naive context dumping:** chiton is reliably **~14–15× cheaper** at
  medium/large scale. This is the dependable token win.
- **vs `grep+read`:** **mixed on raw token count**. chiton is markedly cheaper
  for hub symbols (where grep drags in a whole package — up to ~19×) but
  *costlier* for uniquely-named or highly-connected symbols whose depth-2
  neighborhood is large. Medians: medium ~1.4× cheaper, large ~1.9× costlier.
  Drop to depth 1, or use `query_graph` instead of `get_node_neighborhood`,
  when token cost matters more than breadth.
- **tiny corpora:** grep wins decisively (tens of tokens vs thousands).
- **The token reduction vs grep is NOT the point** — on these corpora it barely
  exists. The point is *better* context: a typed, bounded neighborhood
  (`CALLS`/`IMPORTS`/`DECIDED_IN` edges with provenance) the model doesn't have
  to reconstruct from raw file dumps, plus the large win over naive and the
  capabilities (meeting→code links, findings) grep simply cannot provide.

These are measurements, not marketing. Re-run `chiton bench` on your own corpus.

## Vector index: flat vs HNSW

Semantic search seeds its graph walk from a nearest-neighbour index
(`internal/vector`). `flat` scans every vector; `hnsw` walks a navigable graph.
`vector.index = auto` uses `flat` below 100k vectors and `hnsw` at or above.

### Method

- 768-dimensional vectors (Ollama `nomic-embed-text` size), k = 8, 500 queries.
- Data is synthetic and "embedding-like": points from 200 clusters in a
  48-dimensional latent space, projected to 768 dimensions with noise. Real
  embeddings have this kind of low intrinsic dimension. No real 768-dimensional
  corpus was available offline when this was measured.
- recall@8 is the share of `flat`'s exact top 8 that `hnsw` also returns.
- Heap is live heap held by the index after GC; "raw" is n × 768 × 4 bytes.
- Machine: Intel i9-13900K, Windows 11, Go 1.26, one thread. Its mix of
  performance and efficiency cores makes repeated runs differ by up to 2×.

Reproduce:

```sh
CHITON_VECTOR_BENCH=10000,100000 go test ./internal/vector/ -run TestVectorBenchReport -v
CHITON_VECTOR_BENCH_1M=1 go test ./internal/vector/ -bench BenchmarkSearch -run '^$'
```

### Results (defaults: M 16, ef_search 128, ef_construction 200)

RESULTS_TABLE

### Choosing ef_search

recall@8 by `ef_search` (M 16, ef_construction 200). The query time is per
query at that setting.

| Data | ef 16 | ef 32 | ef 64 | ef 128 | ef 256 |
|---|---|---|---|---|---|
| 100k embedding-like, 768-d | 0.929 (0.12 ms) | 0.987 (0.13 ms) | 0.998 (0.21 ms) | 0.999 (0.25 ms) | 1.000 (0.36 ms) |
| 20k iid Gaussian, 64-d | 0.372 (0.01 ms) | 0.551 (0.03 ms) | 0.736 (0.05 ms) | 0.894 (0.10 ms) | 0.976 (0.16 ms) |

iid Gaussian vectors are the worst case for any approximate index: in many
dimensions every point is about equally far from every other, so there are no
neighbourhoods to navigate. Real embeddings sit between the two rows. The
default of 128 costs a fraction of a millisecond and keeps recall high even on
hard data. Raise it with `vector.hnsw_ef_search` if your corpus behaves more
like the second row.

### Takeaways

- Below 100k vectors `flat` costs at most about 30 ms a query and needs no
  build, so `auto` keeps it.
- At 100k and above, `hnsw` answers in well under a millisecond with
  near-exact recall on embedding-like data. The graph costs about a minute per
  100k vectors to build on every `serve` start; `serve` answers from `flat`
  meanwhile and switches when the build finishes. Saving the built graph to
  disk (spec R10) is the follow-up that would remove the rebuild.
- `hnsw` holds about 10% more heap than the raw vectors. `flat` holds about 2%.
