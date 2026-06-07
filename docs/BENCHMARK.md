# chiton Benchmark — token cost per query across corpora

This is the honest, reproducible benchmark the plan calls for: token cost per
query across corpora of varying size and shape, published as a **table, not a
single hero number**. It deliberately includes the cases where chiton *loses* —
and on raw token count vs `grep+read`, that is more often than the marketing
version of this project would like.

Generated with `chiton bench` (built into the binary). The Phase 0 micro-study
lives in [PHASE0_BENCHMARK.md](PHASE0_BENCHMARK.md); this is the Phase 5
cross-corpus successor.

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

Measured on the chiton repo (Go, `windows/amd64`). Corpus sizes drift as the
repo grows; re-run `chiton bench` to refresh.

### A — synthetic ring (40 files) — the case chiton loses badly

40 files · 280 nodes · 240 edges · ~2 391 tokens in scope · depth 2

| Question | naive | grep+read | chiton | chiton vs grep+read |
|---|---:|---:|---:|---|
| What does Func20 depend on / connect to? | 2 391 | 120 | 4 588 | 38× costlier |
| What does Func38 depend on / connect to? | 2 391 | 120 | 4 588 | 38× costlier |
| What does Func10 depend on / connect to? | 2 391 | 120 | 4 588 | 38× costlier |

**Median:** chiton ~4 588 tokens vs grep+read ~120 (**38× costlier**); naive ~2 391.

On a tiny corpus where each symbol is defined once and grep lands precisely on a
small file, **grep wins decisively** — and chiton's depth-2 neighborhood of a
ring node costs more than reading the whole corpus would. Don't reach for graph
memory on a hobby-sized repo.

### B — `internal/graph` (medium, densely connected)

19 files · 398 nodes · 525 edges · ~24 461 tokens in scope · depth 2

| Question | naive | grep+read | chiton | chiton vs grep+read |
|---|---:|---:|---:|---|
| What does node depend on / connect to? | 24 461 | 23 057 | 1 435 | **16× cheaper** |
| What does TestTwoCliques depend on / connect to? | 24 461 | 860 | 2 827 | 3.3× costlier |
| What does makeNode depend on / connect to? | 24 461 | 2 331 | 1 431 | 1.6× cheaper |
| What does sameComm depend on / connect to? | 24 461 | 1 471 | 1 808 | 1.2× costlier |
| What does RunIncremental depend on / connect to? | 24 461 | 3 346 | 3 380 | ~even |

**Median:** chiton ~1 808 tokens vs grep+read ~2 331 (**1.3× cheaper**); naive ~24 461 (**~14× cheaper**).

For a hub symbol like `node` (referenced everywhere), grep reads nearly the whole
package (23k tokens) while chiton returns a bounded ~1.4k-token neighborhood —
**16× cheaper**. But for uniquely-named or weakly-connected symbols, grep is more
precise and chiton's depth-2 neighborhood can cost more. It nets out to roughly
**par-to-modestly-cheaper** vs grep, and a large win vs naive.

### C — `internal/` (large, 4.6k nodes)

163 files · 4 593 nodes · 5 112 edges · ~205 511 tokens in scope · depth 2

| Question | naive | grep+read | chiton | chiton vs grep+read |
|---|---:|---:|---:|---|
| What does LinkService.Submit depend on / connect to? | 205 511 | 35 810 | 29 187 | 1.2× cheaper |
| What does semanticSearchTool depend on / connect to? | 205 511 | 7 399 | 6 547 | 1.1× cheaper |
| What does GoExtractor.ExtractFile depend on / connect to? | 205 511 | 27 035 | 35 635 | 1.3× costlier |
| What does Generate depend on / connect to? | 205 511 | 9 883 | 13 403 | 1.4× costlier |
| What does Service.Submit depend on / connect to? | 205 511 | 35 810 | 25 552 | 1.4× cheaper |
| What does JavaScriptExtractor.ExtractFile depend on / connect to? | 205 511 | 27 035 | 38 030 | 1.4× costlier |

**Median:** chiton ~27 369 tokens vs grep+read ~27 035 (**~even**); naive ~205 511 (**~7× cheaper**).

At scale, naive context dumping is hopeless (200k+ tokens). chiton vs grep+read
is **roughly a wash on raw token count** — a depth-2 neighborhood of a
heavily-connected symbol can pull in as much as, or more than, grep does. The
reliable, large win is over **naive** (~7–14×).

## Honest summary

- **vs naive context dumping:** chiton is reliably **~7–14× cheaper** at
  medium/large scale. This is the dependable token win.
- **vs `grep+read`:** roughly **par on raw token count**. chiton is markedly
  cheaper for hub symbols (where grep drags in a whole package — up to ~16×) but
  *costlier* for uniquely-named or highly-connected symbols whose depth-2
  neighborhood is large. Median is near parity (medium ~1.3× cheaper, large
  ~even). Drop to depth 1, or use `query_graph` instead of
  `get_node_neighborhood`, when token cost matters more than breadth.
- **tiny corpora:** grep wins decisively (tens of tokens vs thousands).
- **The token reduction vs grep is NOT the point** — on these corpora it barely
  exists. The point is *better* context: a typed, bounded neighborhood
  (`CALLS`/`IMPORTS`/`DECIDED_IN` edges with provenance) the model doesn't have
  to reconstruct from raw file dumps, plus the large win over naive and the
  capabilities (meeting→code links, findings) grep simply cannot provide.

These are measurements, not marketing. Re-run `chiton bench` on your own corpus.
