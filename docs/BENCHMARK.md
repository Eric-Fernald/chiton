# chiton benchmark: token cost per query across corpora

This is chiton's reproducible benchmark: token cost per query across corpora of
varying size and shape, published as a table rather than a single number, plus
the vector index and the performance work behind semantic search. It includes
the cases where chiton *loses*, and on raw token count vs `grep+read` it loses
often.

## At a glance

- **Tokens:** a depth-2 neighborhood is **~5× cheaper** than dumping a medium
  package into context and **~48× cheaper** on a large tree. Against
  `grep+read` it is roughly on par overall but mixed by corpus: about even on a
  medium package, 2.2× cheaper on a large tree, and far costlier on a tiny one.
  The depth-1 setting is 10–16× cheaper than `grep+read` on real code. See
  [Results](#results).
- **Answer quality:** on 18 structural questions about this repository, Claude
  Sonnet 5.5 scored F1 0.99 with grep alone. Adding chiton left accuracy the same
  (the model kept using grep) and raised input tokens; graph-only answers were
  never wrong but missed some items (F1 0.96). See
  [Answer quality](#answer-quality).
- **Semantic search at 1M vectors:** **0.50 ms** p50 per query with HNSW, about
  **1,200× faster** than the 600 ms exact scan, at a recall@8 of **0.984**
  (0.998 at 100k). See [Vector index](#vector-index-flat-vs-hnsw).
- **Index build and load:** 2.8 s to build at 100k vectors (down from 36.4 s),
  0.7 s to load the saved index, which is about 2% the size of the vector data.
- **Other measured improvements:** a default binary 83% smaller, live sync with
  analysis about 3× faster, and Leiden communities that no longer collapse. See
  [Other performance results](#other-performance-results).

The point of chiton is better context, not fewer tokens; [Summary](#summary)
explains why.

Generated with `chiton bench` (built into the binary).

## What is measured

For a corpus, chiton builds an **in-memory** graph from its AST extractors (no
store, no running server), then answers a deterministic set of dependency
questions of the form *"what does X depend on / connect to?"* and reports the
token cost of each retrieval strategy:

| Strategy | Models | Cost = tokens of… |
|---|---|---|
| **naive** | a model with no tooling reading the whole scope | every file in the corpus |
| **grep+read** | the grep_search + read_file default of Copilot / Claude Code | every file that mentions the symbol |
| **chiton d2** | chiton's MCP `get_node_neighborhood` at depth 2 | the lean node/edge JSON returned |
| **chiton d1** | the same tool at depth 1 | the lean node/edge JSON returned |
| **query_graph** | chiton's `query_graph` lookup for the symbol's name | up to 10 matching nodes, no edges |

Depth 1 returns only the symbol's direct neighbors, and `query_graph` only
finds the symbol, so they answer less than depth 2 does. They show what the
cheaper settings cost when a question doesn't need the wider neighborhood.

Methodology notes:

- **Token estimate = characters ÷ 4**, applied identically to every strategy.
- **The chiton payload is byte-faithful to the real MCP output.** The bench
  serialises the *exact* `leanNode`/`leanEdge` shape `internal/mcp/server.go`
  returns, so the chiton columns are not under-counted. That shape has `kind`,
  `label`, `source_uri`, `span`, `attrs`, `provenance`, `confidence`, edge
  `attrs`, and the neighborhood `summary` line. (An earlier draft of this benchmark
  dropped `attrs`/`provenance`/`confidence` and thereby flattered chiton; that
  bias is fixed, which is why the numbers below are higher than a naive lean
  encoding would show.)
- **Query targets** are the highest-degree **corpus-internal** symbols outside
  test files. External package nodes (`fmt`, `http.Get`) are excluded: nobody
  asks "what does `fmt` depend on," and their high degree would inflate every
  column. Test symbols are excluded too: a busy test helper is rarely what a
  developer asks about, and its wide neighborhood dominated the old sample.
  `--include-tests` restores them, and the results below show both. Test files
  stay in the corpus either way, so naive and grep+read are unaffected.
- **grep+read greps the base identifier** a developer would type. For the
  method label `LinkService.Submit` it greps `Submit`. The match is a substring
  match, so a very common name like `New` also matches `NewEngine`, `NewStore`
  and so on, and grep+read reads every one of those files. A developer would
  usually search more precisely, so treat grep+read's cost on such names as an
  upper bound.

### What this does not measure

- **Answer quality.** This counts retrieval tokens, not whether the model got
  the right answer. [Answer quality](#answer-quality) measures that separately.
- **Prompt caching.** Naive/grep contexts are far more cacheable than chiton's
  per-query neighborhoods; with a warm cache the input-token gap narrows further.
- **Output tokens.** At $25/M, output often dominates the bill regardless of
  retrieval strategy.

## Reproduce

```sh
chiton bench --synth 40                       # reproducible synthetic ring corpus
chiton bench ./internal/graph                 # a real medium, densely-connected package
chiton bench ./internal --queries 6           # the whole internal/ tree (large)
chiton bench ./internal --include-tests       # let test symbols be query targets
chiton bench <path> --json                    # machine-readable output
```

`--synth N` generates a fixed N-file Go corpus into a temp dir, so the small end
is identical on every machine.

## Results

Measured on the chiton repo at commit `b535d99` (Go, `windows/amd64`,
2026-10-07). Corpus sizes drift as the repo grows; re-run `chiton bench` to
refresh.

### A. Synthetic ring (40 files), where chiton loses

40 files · 280 nodes · 240 edges · ~2 391 tokens in scope

| Question | naive | grep+read | chiton d2 | chiton d1 | query_graph | d2 vs grep+read | d1 vs grep+read |
|---|---:|---:|---:|---:|---:|---|---|
| What does Func14 depend on / connect to? | 2 391 | 120 | 4 609 | 265 | 80 | 38× costlier | 2.2× costlier |
| What does Func39 depend on / connect to? | 2 391 | 120 | 4 609 | 265 | 80 | 38× costlier | 2.2× costlier |
| What does Func25 depend on / connect to? | 2 391 | 120 | 4 609 | 265 | 80 | 38× costlier | 2.2× costlier |
| What does Func19 depend on / connect to? | 2 391 | 120 | 4 609 | 265 | 80 | 38× costlier | 2.2× costlier |
| What does Func16 depend on / connect to? | 2 391 | 120 | 4 609 | 265 | 80 | 38× costlier | 2.2× costlier |

**Median:** grep+read ~120 tokens; chiton d2 ~4 609 (**38× costlier**), d1 ~265
(2.2× costlier), query_graph ~80; naive ~2 391.

On a tiny corpus where each symbol is defined once and grep lands precisely on a
small file, **grep wins decisively**. Even depth 1 costs twice as much, and the
depth-2 neighborhood of a ring node costs more than reading the whole corpus.
Don't reach for graph memory on a hobby-sized repo.

### B. `internal/graph` (medium, densely connected)

47 files · 930 nodes · 1 256 edges · ~53 930 tokens in scope

| Question | naive | grep+read | chiton d2 | chiton d1 | query_graph | d2 vs grep+read | d1 vs grep+read |
|---|---:|---:|---:|---:|---:|---|---|
| What does scan depend on / connect to? | 53 930 | 15 691 | 10 905 | 1 266 | 515 | 1.4× cheaper | 12× cheaper |
| What does replaceSource depend on / connect to? | 53 930 | 9 750 | 10 266 | 1 044 | 904 | 1.1× costlier | 9.3× cheaper |
| What does Engine.Query depend on / connect to? | 53 930 | 3 434 | 5 813 | 749 | 437 | 1.7× costlier | 4.6× cheaper |
| What does deleteNode depend on / connect to? | 53 930 | 9 750 | 9 963 | 982 | 343 | 1.0× costlier | 9.9× cheaper |
| What does RunIncremental depend on / connect to? | 53 930 | 3 693 | 4 933 | 870 | 141 | 1.3× costlier | 4.2× cheaper |

**Median:** grep+read ~9 750 tokens; chiton d2 ~9 963 (**about even**), d1 ~982
(**9.9× cheaper**), query_graph ~437; naive ~53 930 (d2 **~5× cheaper**).

At depth 2 chiton costs about what grep+read does on a medium package: cheaper
for a hub like `scan`, a little costlier for the rest. Depth 1 is roughly 10×
cheaper than grep+read when the question only needs direct callers and callees.

### C. `internal/` (large)

332 files · 6 863 nodes · 9 003 edges · ~401 955 tokens in scope

| Question | naive | grep+read | chiton d2 | chiton d1 | query_graph | d2 vs grep+read | d1 vs grep+read |
|---|---:|---:|---:|---:|---:|---|---|
| What does New depend on / connect to? | 401 955 | 238 102 | 7 979 | 1 319 | 723 | 30× cheaper | 181× cheaper |
| What does Analyze depend on / connect to? | 401 955 | 34 891 | 47 741 | 1 240 | 574 | 1.4× costlier | 28× cheaper |
| What does neighborhoodTool depend on / connect to? | 401 955 | 6 485 | 8 509 | 981 | 63 | 1.3× costlier | 6.6× cheaper |
| What does semanticSearchTool depend on / connect to? | 401 955 | 8 385 | 5 356 | 926 | 63 | 1.6× cheaper | 9.1× cheaper |
| What does LocalSource.Messages depend on / connect to? | 401 955 | 19 138 | 24 960 | 1 120 | 507 | 1.3× costlier | 17× cheaper |
| What does jsonResult depend on / connect to? | 401 955 | 17 455 | 8 378 | 1 126 | 183 | 2.1× cheaper | 16× cheaper |

**Median:** grep+read ~18 296 tokens; chiton d2 ~8 443 (**2.2× cheaper**), d1
~1 123 (**16× cheaper**), query_graph ~345; naive ~401 955 (d2 **~48× cheaper**).

At scale, naive context dumping is hopeless (400k tokens). Against grep+read,
depth 2 wins three of the six questions and loses the other three by at most
1.4×. The `New` row overstates grep's cost (see the substring note above), but
leaving it out barely moves the medians: depth 2 is still about 2× cheaper, and
depth 1 still wins every question.

### Test symbols as targets: the previous method

Before this change, the benchmark let test symbols be query targets. Run on the
same commit with `--include-tests`:

| Corpus | grep+read | chiton d2 | chiton d1 | query_graph |
|---|---:|---:|---:|---:|
| `internal/graph` | 3 434 | 9 150 (2.7× costlier) | 932 (3.7× cheaper) | 437 |
| `internal/` | 5 823 | 5 802 (about even) | 1 701 (3.4× cheaper) | 176 |

Medians. The synthetic corpus has no tests and doesn't change. Test helpers such
as `newHarness` and `TestVectorBenchReport` are among the most connected
symbols, and grep lands on the one or two files that use them, which is why
they pulled the old depth-2 medians toward grep. The last version of this page,
measured on an older commit with test targets, reported depth-2 medians of
1.4× cheaper on `internal/graph` and 1.9× costlier on `internal/`.

## Summary

- **vs naive context dumping:** chiton's depth-2 neighborhood is **~5× cheaper**
  on a medium package and **~48× cheaper** on a large tree.
- **vs `grep+read`:** **mixed at depth 2.** About even on a medium package and
  2.2× cheaper on a large tree, winning big for hub symbols (where grep drags in
  much of a package) and losing narrowly for symbols with a wide neighborhood.
- **Depth 1 is the cheap setting:** ~10× cheaper than grep+read on the medium
  package and ~16× cheaper on the large tree, when direct callers and callees are
  enough. `query_graph` costs a few hundred tokens to find a symbol.
- **Tiny corpora:** grep wins decisively (tens of tokens vs hundreds or thousands).
- **Token cost isn't the main argument.** The point is *better* context: a
  typed, bounded neighborhood (`CALLS`/`IMPORTS`/`DECIDED_IN` edges with
  provenance) the model doesn't have to reconstruct from raw file dumps, plus
  capabilities grep cannot provide (meeting→code links, findings).
  [Answer quality](#answer-quality) tests whether that context leads to better
  answers.

Re-run `chiton bench` on your own corpus.

## Answer quality

The token tables measure what each strategy costs, not whether the model ends up
with the right answer. This benchmark asks Claude Code structural questions about
a real repository and grades the answers.

### Method

- **Questions.** 18 questions about chiton's own Go code at commit `b535d99`:
  6 *callers* ("which functions call `internal/config:StateDir`?"), 4 *callees*,
  4 *imports* ("which repository packages does `internal/analyze` import?") and
  4 *importers*. `qualbench gen` picks them deterministically: every expected
  answer has 3 to 12 items, and caller and callee answers span at least two
  packages, so no single file holds the answer.
- **Ground truth** comes from the Go source with `go/ast`, independently of
  chiton's extractor. It covers every non-test `.go` file regardless of build
  tags. Spot checks against `git grep` matched.
- **Model.** Claude Code 2.1.289 in headless mode (`claude -p`) with Claude
  Sonnet 5.5 (`--model sonnet`). Each question was asked twice under each
  condition: 108 runs. `--tools` limits the built-in tools, `--setting-sources
  project` keeps user plugins and hooks out, and `--strict-mcp-config` loads only
  the MCP servers a condition names.
- **Conditions.**
  - **grep:** the built-in Read, Grep and Glob tools only.
  - **chiton:** the same tools plus chiton's MCP server, in a checkout where
    `chiton install` wrote its rules block and skills. The model chooses which
    tools to use, as it would in practice.
  - **chiton-only:** chiton's MCP server plus Read, with Grep and Glob removed,
    so the answer has to come from the graph.
- **Graph.** `chiton build` and `chiton cluster` over the checkout (3 996 nodes,
  10 548 edges). No embeddings, so `semantic_search` was unavailable.
- **Scoring.** The model answers with a JSON list. Each answer gets precision,
  recall and F1 against the ground truth; *exact* means the sets match.

### Results

| Condition | Mean F1 | Precision | Recall | Exact answers | Median input tokens | Median turns | Median time | Total cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| grep | 0.99 | 0.99 | 0.99 | 34 / 36 | 22 537 | 4 | 6 s | $0.99 |
| chiton | 0.99 | 0.99 | 0.99 | 35 / 36 | 42 386 | 4 | 6 s | $1.21 |
| chiton-only | 0.96 | 1.00 | 0.95 | 32 / 36 | 61 022 | 5 | 8 s | $2.94 |

Input tokens count cached and uncached input together. Cost is the API price
Claude Code reports for the runs.

| Question kind | grep | chiton | chiton-only |
|---|---:|---:|---:|
| callers | 0.96 | 0.97 | 0.94 |
| callees | 1.00 | 1.00 | 1.00 |
| imports | 1.00 | 1.00 | 0.91 |
| importers | 1.00 | 1.00 | 1.00 |

Mean F1. Per-question scores, every answer and every tool call are in
[`benchmark-data/answer-quality-results.json`](benchmark-data/answer-quality-results.json);
the questions and their ground truth are in
[`benchmark-data/answer-quality-questions.json`](benchmark-data/answer-quality-questions.json).

### What the results show

- **These questions don't separate the conditions on accuracy.** With grep
  alone, Sonnet 5.5 answered 34 of 36 exactly on a 380-file repository. Its two
  misses credited a call to the wrong enclosing function: the call sits on the
  first line of `runDiff`, and the model attributed it to `analyzeCmd` above.
  Showing a quality difference needs harder questions: transitive callers,
  larger repositories, or questions about decisions that only meeting links
  can answer.
- **Given chiton alongside grep, the model didn't use it.** The chiton runs
  made about 2.9 Grep calls each and no chiton calls, even with chiton's rules
  block telling the assistant to prefer its tools. The rules, skills and tool
  definitions still entered the context, so median input rose from about 22.5k
  to 42.4k tokens and cost rose 22%, for the same answers.
- **The graph alone was never wrong, but missed items.** Every item the
  chiton-only runs returned was correct (precision 1.00), but they missed some
  (recall 0.95), and they used about 2.7× the input tokens of grep. The misses
  had two causes:
  - *Imports are recorded per file.* For `internal/ingest/browser`, both runs
    missed the two packages that only `watch.go` imports, although the graph
    holds those edges. Answering a package question means checking every file's
    imports; a package-level view would answer it in one call.
  - *Common names are harder.* Asked for the callers of `internal/analyze:New`,
    a name many packages use, one run found only 1 of the 5 callers.

These are the next things to fix: tool descriptions and rules that lead the
model to chiton when it helps, a package-level imports view, and better
disambiguation for common names. Then re-run with harder questions.

### Reproduce

The harness is `internal/bench/qualbench`. It needs the `claude` CLI on `PATH`,
signed in. Runs count against your Claude plan.

```sh
# two clean checkouts of the commit under test
git clone <repo> grep-checkout
git clone <repo> chiton-checkout
cd chiton-checkout && chiton build . && chiton cluster && chiton install --assistant claude-code && cd ..

go run ./internal/bench/qualbench gen -repo grep-checkout -out questions.json
go run ./internal/bench/qualbench run -questions questions.json \
  -grep-dir grep-checkout -chiton-dir chiton-checkout -chiton-bin "$(which chiton)" \
  -repeat 2 -out results.json
go run ./internal/bench/qualbench report -questions questions.json -results results.json
```

With `-parallel N`, give each worker its own copy of the graph through
`-graph-dirs` (directories that each hold a copy of `chiton-out/`).

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
- Saved index is the size of `hnsw.idx`, which holds links but no vectors.
  Load is the time to rebuild the index from it as `serve` does: fingerprint
  every vector, read the file, and normalize each vector into the graph. It
  excludes reading the vectors from the store.
- Machine: Intel i9-13900K (32 threads), Windows 11, Go 1.26. Queries run on one
  thread; the HNSW build uses all of them. The mix of performance and
  efficiency cores makes repeated runs differ by up to 2×.

Reproduce:

```sh
CHITON_VECTOR_BENCH=10000,100000,1000000 go test ./internal/vector/ -run TestVectorBenchReport -v -timeout 60m
CHITON_VECTOR_BENCH_1M=1 go test ./internal/vector/ -bench BenchmarkSearch -run '^$'
```

The 1M row needs about 13 GB of RAM: the report holds the source vectors, both
indexes, and a loaded copy at once.

### Results (defaults: M 16, ef_search 128, ef_construction 100)

| Vectors | Backend | Build | Heap | Heap / raw | p50 query | p95 query | recall@8 | Saved index | Load |
|---|---|---|---|---|---|---|---|---|---|
| 10k | flat | 0.01 s | 32 MB | 1.03 | 2.77 ms | 5.77 ms | 1.000 | — | — |
| 10k | hnsw | 0.20 s | 34 MB | 1.10 | 0.20 ms | 0.27 ms | 0.999 | 0.5 MB | 0.04 s |
| 100k | flat | 0.27 s | 315 MB | 1.02 | 58.9 ms | 64.5 ms | 1.000 | — | — |
| 100k | hnsw | 2.8 s | 336 MB | 1.09 | 0.10 ms | 0.14 ms | 0.998 | 5.8 MB | 0.70 s |
| 1M | flat | 2.4 s | 3168 MB | 1.03 | 600 ms | 634 ms | 1.000 | — | — |
| 1M | hnsw | 78 s | 3373 MB | 1.10 | 0.50 ms | 0.64 ms | 0.984 | 68.5 MB | 7.7 s |

Before the build went parallel and `ef_construction` dropped from 200 to 100,
the same machine built 10k in 5.3 s and 100k in 36.4 s, with recall@8 of 1.000
at both sizes. The 100k flat query time doubled between runs (29 ms before) with
no change to `flat`, which is the core-mix noise noted above.

### Choosing ef_search

recall@8 by `ef_search` (M 16, ef_construction 100, 200 queries). The query
time is the mean per query at that setting; at these sub-millisecond times the
core-mix noise can reorder neighbouring columns.

| Data | ef 16 | ef 32 | ef 64 | ef 128 | ef 256 |
|---|---|---|---|---|---|
| 100k embedding-like, 768-d | 0.931 (0.14 ms) | 0.983 (0.26 ms) | 0.998 (0.34 ms) | 0.999 (0.41 ms) | 0.999 (0.31 ms) |
| 20k iid Gaussian, 64-d | 0.358 (0.01 ms) | 0.543 (0.03 ms) | 0.720 (0.05 ms) | 0.869 (0.08 ms) | 0.965 (0.17 ms) |

With ef_construction 200 the same rows read 0.999 and 0.894 at ef 128: halving
it cost nothing on embedding-like data and 2.5 points on iid data.

iid Gaussian vectors are the worst case for any approximate index: in many
dimensions every point is about equally far from every other, so there are no
neighbourhoods to navigate. Real embeddings sit between the two rows. The
default of 128 costs a fraction of a millisecond and keeps recall high even on
hard data. Raise it with `vector.hnsw_ef_search` if your corpus behaves more
like the second row.

### Takeaways

- Below 100k vectors `flat` costs at most a few tens of milliseconds a query
  and needs no build, so `auto` keeps it.
- At 100k and above, `hnsw` answers in well under a millisecond with
  near-exact recall on embedding-like data.
- `chiton embed` builds the graph on every core, about 3 s at 100k vectors and
  under a minute and a half at 1M, and saves it. `serve` loads the saved index
  in under a second at 100k and about 8 s at 1M, then answers from HNSW at once.
  Only when the saved index is missing or out of date does `serve` answer from
  `flat` while it builds HNSW in the background.
- `hnsw` holds about 10% more heap than the raw vectors. `flat` holds about 3%.
  The saved index adds about 2% of the vector data on disk.

### Recall in context

recall@8 asks whether the right material reaches the answer. Without an index,
a model has to find that material inside its own context window. Anthropic's
Claude Opus 4.6 launch (February 2026) reported these scores on MRCR v2 with 8
needles at 1M tokens, a test of picking the right item out of eight look-alikes
in a million tokens of raw text:

| Retrieval | Score |
|---|---|
| chiton HNSW, recall@8 at 1M vectors | 0.984 |
| Claude Opus 4.6 alone, MRCR v2 8-needle at 1M tokens | 0.76 |
| Claude Sonnet 4.5 alone, same test | 0.185 |

These are different tests. chiton's number is index recall against an exact
scan on synthetic vectors, and it covers only the seed step; the graph walk then
adds the connected code. MRCR scores a model reading raw text. Both measure
whether the right material reaches the answer, but they are not a head-to-head
result. A direct comparison would run one question set on a real repository
with and without chiton and grade the final answers.

## Other performance results

| Change | Before | After | Notes |
|---|---|---|---|
| Default binary size | 142 MB | 24.6 MB (83% smaller) | The Microsoft Graph SDK linked about 117 MB for two API calls; it now builds only with the opt-in `msgraph` tag. |
| Live sync with analysis | 10.1 s | 3.4 s (about 3× faster) | `sync once` on Go's `src/net` (390 files). Findings are pruned through their `AFFECTS` edges instead of a scan of the whole store. |
| HNSW build, 100k vectors | 36.4 s | 2.8 s | The build runs on every core, and `ef_construction` dropped from 200 to 100 with no recall loss on embedding-like data. |
| Largest Leiden community, chiton's own graph | 87.5% of nodes | 15% | A correctness fix: aggregation dropped each community's internal weight. Zachary's karate club now scores a modularity of about 0.42, up from 0. Re-run `chiton cluster` after upgrading. |