# chiton benchmark: token cost per query across corpora

This is chiton's reproducible benchmark: token cost per query across corpora of
varying size and shape, published as a table rather than a single number, plus
the vector index and the performance work behind semantic search. It includes
the cases where chiton *loses*, and on raw token count vs `grep+read` it loses
often.

## At a glance

- **Answer quality:** on 20 multi-hop questions about this repository (callers
  within 2-3 calls, transitive dependencies), Claude Sonnet 5.5 with chiton
  answered **40 of 40 exactly**, against 35 of 40 with grep alone, in **3 turns
  instead of 8**, **5 seconds instead of 16**, and with 39% fewer input tokens.
  On direct one-hop questions every setup was exact, and chiton's instructions
  and tool definitions made it 15% costlier. See
  [Answer quality](#answer-quality).
- **Tokens:** a depth-2 neighborhood is **~4× cheaper** than dumping a medium
  package into context and **~23× cheaper** on a large tree. Against
  `grep+read`, for the most connected symbols, it is about even on a medium
  package and 3.6× cheaper on a large tree, and far costlier on a tiny one. The
  depth-1 setting is 9–14× cheaper than `grep+read` on real code. See
  [Results](#results).
- **Semantic search at 1M vectors:** **0.50 ms** p50 per query with HNSW, about
  **1,200× faster** than the 600 ms exact scan, at a recall@8 of **0.984**
  (0.998 at 100k). See [Vector index](#vector-index-flat-vs-hnsw).
- **Index build and load:** 2.8 s to build at 100k vectors (down from 36.4 s),
  0.7 s to load the saved index, which is about 2% the size of the vector data.
- **Other measured improvements:** a default binary 83% smaller, live sync with
  analysis about 3× faster, and Leiden communities that no longer collapse. See
  [Other performance results](#other-performance-results).

The point of chiton is better context, not fewer tokens; [Summary](#summary)
and [Answer quality](#answer-quality) show what that buys.

Generated with `chiton bench` (built into the binary).

## What is measured

For a corpus, chiton builds an **in-memory** graph from its AST extractors (no
store, no running server), then answers a deterministic set of dependency
questions of the form *"what does X depend on / connect to?"* and reports the
token cost of each retrieval strategy:

| Strategy | Models | Cost = tokens of… |
|---|---|---|
| **naive** | a model with no tooling reading the whole scope | every file in the corpus |
| **grep+read** | the grep_search + read_file default of Copilot / Claude Code | every file that mentions the symbol as a whole word |
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
  `attrs`, and the neighborhood `summary` line. Like the server, the bench joins
  each function to the references its callers in other files and packages point
  at, so a neighborhood holds every caller the server would return.
- **Query targets** are the highest-degree **corpus-internal** symbols outside
  test files. External package nodes (`fmt`, `http.Get`) are excluded: nobody
  asks "what does `fmt` depend on," and their high degree would inflate every
  column. Test symbols are excluded too (`--include-tests` restores them). Test
  files stay in the corpus either way, so naive and grep+read are unaffected.
  Highest-degree symbols are hubs, which is where grep reads the most files, so
  this sample favours chiton against grep+read; a randomly chosen symbol would
  usually be cheaper to grep.
- **grep+read greps the base identifier** a developer would type, as a whole
  word (`grep -w`). For the method label `LinkService.Submit` it greps `Submit`.
  A very common name like `New` still matches every `New(` call across the tree,
  so treat grep+read's cost on such names as an upper bound.

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

Measured on the chiton repo at commit `ddfab54` (Go, `windows/amd64`,
2026-10-07). Corpus sizes drift as the repo grows; re-run `chiton bench` to
refresh.

### A. Synthetic ring (40 files), where chiton loses

40 files · 280 nodes · 240 edges · ~2 391 tokens in scope

| Question | naive | grep+read | chiton d2 | chiton d1 | query_graph | d2 vs grep+read | d1 vs grep+read |
|---|---:|---:|---:|---:|---:|---|---|
| What does Func30 depend on / connect to? | 2 391 | 120 | 4 609 | 265 | 80 | 38× costlier | 2.2× costlier |
| What does Func23 depend on / connect to? | 2 391 | 120 | 4 609 | 265 | 80 | 38× costlier | 2.2× costlier |
| What does Func11 depend on / connect to? | 2 391 | 120 | 4 609 | 265 | 80 | 38× costlier | 2.2× costlier |
| What does Func18 depend on / connect to? | 2 391 | 120 | 4 609 | 265 | 80 | 38× costlier | 2.2× costlier |
| What does Func28 depend on / connect to? | 2 391 | 120 | 4 609 | 265 | 80 | 38× costlier | 2.2× costlier |

**Median:** grep+read ~120 tokens; chiton d2 ~4 609 (**38× costlier**), d1 ~265
(2.2× costlier), query_graph ~80; naive ~2 391.

On a tiny corpus where each symbol is defined once and grep lands precisely on a
small file, **grep wins decisively**. Even depth 1 costs twice as much, and the
depth-2 neighborhood of a ring node costs more than reading the whole corpus.
Don't reach for graph memory on a hobby-sized repo.

### B. `internal/graph` (medium, densely connected)

47 files · 917 nodes · 1 444 edges · ~54 910 tokens in scope

| Question | naive | grep+read | chiton d2 | chiton d1 | query_graph | d2 vs grep+read | d1 vs grep+read |
|---|---:|---:|---:|---:|---:|---|---|
| What does Open depend on / connect to? | 54 910 | 27 197 | 22 220 | 3 885 | 708 | 1.2× cheaper | 7.0× cheaper |
| What does Run depend on / connect to? | 54 910 | 13 645 | 7 616 | 1 844 | 678 | 1.8× cheaper | 7.4× cheaper |
| What does scan depend on / connect to? | 54 910 | 12 685 | 12 495 | 1 533 | 544 | about even | 8.3× cheaper |
| What does OpenReadOnly depend on / connect to? | 54 910 | 18 907 | 13 083 | 1 417 | 323 | 1.4× cheaper | 13× cheaper |
| What does RunIncremental depend on / connect to? | 54 910 | 3 693 | 6 212 | 1 364 | 141 | 1.7× costlier | 2.7× cheaper |

**Median:** grep+read ~13 645 tokens; chiton d2 ~12 495 (**about even**), d1
~1 533 (**8.9× cheaper**), query_graph ~544; naive ~54 910 (d2 **~4.4× cheaper**).

At depth 2 chiton costs about what grep+read does on a medium package: a little
cheaper for the hubs, costlier for `RunIncremental`, which few files name. Depth
1 is roughly 9× cheaper when the question only needs direct callers and callees.

### C. `internal/` (large)

335 files · 6 560 nodes · 10 289 edges · ~414 736 tokens in scope

| Question | naive | grep+read | chiton d2 | chiton d1 | query_graph | d2 vs grep+read | d1 vs grep+read |
|---|---:|---:|---:|---:|---:|---|---|
| What does Register depend on / connect to? | 414 736 | 62 275 | 15 172 | 6 125 | 481 | 4.1× cheaper | 10× cheaper |
| What does Open depend on / connect to? | 414 736 | 89 131 | 40 160 | 5 828 | 735 | 2.2× cheaper | 15× cheaper |
| What does FileLines depend on / connect to? | 414 736 | 24 548 | 15 880 | 4 316 | 92 | 1.5× cheaper | 5.7× cheaper |
| What does New depend on / connect to? | 414 736 | 151 322 | 19 819 | 3 686 | 782 | 7.6× cheaper | 41× cheaper |
| What does NewNodeID depend on / connect to? | 414 736 | 66 789 | 28 512 | 4 772 | 84 | 2.3× cheaper | 14× cheaper |
| What does FileURI depend on / connect to? | 414 736 | 31 475 | 16 414 | 4 576 | 752 | 1.9× cheaper | 6.9× cheaper |

**Median:** grep+read ~64 532 tokens; chiton d2 ~18 116 (**3.6× cheaper**), d1
~4 674 (**14× cheaper**), query_graph ~608; naive ~414 736 (d2 **~23× cheaper**).

At scale, naive context dumping is hopeless (400k tokens), and the hubs grep has
to chase through dozens of files favour the graph: depth 2 wins all six
questions. Remember the sampling note above: these are the most connected
symbols in the tree, and `New` and `Open` are names every package uses.

### How these numbers changed

Two changes since the previous version of this page moved the results, and both
are in the bench code:

- **Calls now resolve across files and packages.** Before, the extractor linked
  a call only when caller and callee shared a file, so a function's neighborhood
  missed its callers elsewhere. Neighborhoods now include them, which makes them
  larger, and the most connected symbols are now real cross-package hubs rather
  than test helpers.
- **grep+read matches whole words.** It used to match substrings, so `Open` also
  read every file mentioning `OpenReadOnly`. That made grep look costlier than a
  developer's search would be.

On the previous code (commit `b535d99`), the medians were: `internal/graph`
grep+read ~9 750 vs chiton d2 ~9 963, and `internal/` grep+read ~18 296 vs
chiton d2 ~8 443. With test symbols allowed as targets (`--include-tests`), the
current medians don't change: one `internal/graph` target becomes a test, and
none in `internal/` do.

## Summary

- **vs naive context dumping:** chiton's depth-2 neighborhood is **~4× cheaper**
  on a medium package and **~23× cheaper** on a large tree.
- **vs `grep+read`:** about even on a medium package and **3.6× cheaper** on a
  large tree, for the most connected symbols. Hubs favour chiton; a symbol few
  files name favours grep.
- **Depth 1 is the cheap setting:** ~9× cheaper than grep+read on the medium
  package and ~14× cheaper on the large tree, when direct callers and callees are
  enough. `query_graph` costs a few hundred tokens to find a symbol.
- **Tiny corpora:** grep wins decisively (tens of tokens vs hundreds or thousands).
- **Tokens are only part of it.** [Answer quality](#answer-quality) measures
  what the model does with the context: on multi-hop questions, chiton's answers
  were exact every time and took a third of the time.

Re-run `chiton bench` on your own corpus.

## Answer quality

The token tables measure what each strategy costs, not whether the model ends up
with the right answer. This benchmark asks Claude Code structural questions about
a real repository and grades the answers.

### Method

- **Questions** about chiton's own Go code, picked deterministically by
  `qualbench gen` so that no single file holds the answer:
  - **Multi-hop set** (20 questions, the main result): callers within 2 calls
    (6) and within 3 calls (4), callees within 2 calls (4), every package a
    package depends on through its imports (3), and every package that depends
    on a package (3). Answers have 7 to 22 items across at least three packages.
    Caller chains only pass through package-level functions, since a method's
    callers can't be resolved from syntax alone.
  - **Direct set** (18 questions): direct callers (6), callees (4), imports (4)
    and importers (4), with 3 to 12 items each.
- **Ground truth** comes from the Go source with `go/ast`, independently of
  chiton's extractor. It covers every non-test `.go` file regardless of build
  tags. Spot checks against `git grep` matched.
- **Model.** Claude Code 2.1.289 in headless mode (`claude -p`) with Claude
  Sonnet 5.5 (`--model sonnet`). Each question was asked twice under each
  condition. `--tools` limits the built-in tools, `--setting-sources project`
  keeps user plugins and hooks out, and `--strict-mcp-config` loads only the MCP
  servers a condition names.
- **Conditions.**
  - **grep:** the built-in Read, Grep and Glob tools only.
  - **chiton:** the same tools plus chiton's MCP server, in a checkout where
    `chiton install` wrote its rules block and skills. The model chooses which
    tools to use, as it would in practice.
  - **chiton-only:** chiton's MCP server plus Read, with Grep and Glob removed,
    so the answer has to come from the graph.
- **Graph.** `chiton build` and `chiton cluster` over the checkout. No
  embeddings, so `semantic_search` was unavailable.
- **Scoring.** The model answers with a JSON list; a reply that corrects itself
  is graded on its last list. Each answer gets precision, recall and F1 against
  the ground truth; *exact* means the sets match.

### Results: multi-hop questions

Commit `e3b40c0`, 120 runs.

| Condition | Mean F1 | Exact answers | Median input tokens | Median output tokens | Median turns | Median time | Total cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| grep | 0.99 | 35 / 40 | 52 666 | 1 629 | 8 | 16 s | $2.66 |
| chiton | **1.00** | **40 / 40** | **32 069** | **493** | **3** | **5 s** | **$1.62** |
| chiton-only | **1.00** | **40 / 40** | 28 640 | 345 | 2 | 5 s | $1.34 |

| Question kind | grep | chiton | chiton-only |
|---|---:|---:|---:|
| callers within 2 calls | 1.00 | 1.00 | 1.00 |
| callers within 3 calls | 1.00 | 1.00 | 1.00 |
| callees within 2 calls | 0.95 | 1.00 | 1.00 |
| all packages depended on | 0.98 | 1.00 | 1.00 |
| all dependent packages | 1.00 | 1.00 | 1.00 |

Mean F1. Input tokens count cached and uncached input together; cost is the API
price Claude Code reports for the runs.

- **With chiton available, the model answered every question exactly** and
  needed far less work: 3 turns instead of 8, 5 seconds instead of 16, 39% fewer
  input tokens, 70% fewer output tokens, and 39% lower cost.
- **It used chiton when chiton was the better tool.** In the chiton condition
  it used a chiton tool in 39 of 40 runs, making about 1 chiton call and 0.9
  grep calls per question, against 6.2 grep and 1.5 read calls without chiton.
- **grep's misses were chain errors.** Its five inexact answers counted callees
  one step too deep (one run listed 15 extra functions) or missed two packages
  at the end of an import chain. Those are the mistakes a multi-step manual
  search makes; a graph walk with an explicit depth doesn't.
- **Sonnet 5.5 is good at this with grep alone.** F1 0.99 means grep's answers
  were nearly right; chiton's gain is mostly speed, cost and reliability rather
  than rescuing wrong answers.

### Results: direct questions

The same kinds of direct question as the first run (callers, callees, imports,
importers), regenerated at commit `e3b40c0`, 108 runs.

| Condition | Mean F1 | Exact answers | Median input tokens | Median output tokens | Median turns | Median time | Total cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| grep | 1.00 | 36 / 36 | 22 609 | 573 | 4 | 7 s | $0.96 |
| chiton | 1.00 | 36 / 36 | 32 061 | 354 | 3 | 5 s | $1.10 |
| chiton-only | 1.00 | 36 / 36 | 41 203 | 400 | 3 | 6 s | $1.07 |

Every condition answered every direct question exactly. Here chiton's cost is
visible: its rules block, skills and tool definitions add about 9 500 input
tokens to every question, so the chiton condition cost 15% more for an answer a
couple of seconds sooner. One grep call answers a direct question, which leaves
the graph little to save. The model still used a chiton tool in 25 of the 36
runs, against none before the fixes.

### What changed between runs

The first run of this benchmark, on the direct questions at commit `b535d99`,
found that chiton added nothing: the model made about 2.9 grep calls per
question and **no** chiton calls, even with chiton's rules block telling it to
prefer them, while chiton's rules, skills and tool definitions doubled the input
tokens. Graph-only answers were never wrong but missed items (F1 0.96).
Investigating those results found real gaps in chiton:

- **Calls between files were missing.** The Go extractor only linked a call when
  caller and callee shared a file, so the graph couldn't answer "who calls X"
  from another package or a sibling file.
- **Imports were per file only.** A package-level question meant checking every
  file's imports.
- **Common names resolved to the wrong symbol.** A lookup for `New` returned the
  first label containing "New".

The fixes, all in this release: calls now resolve across files and packages;
`get_node_neighborhood` takes a `direction` (callers with `"in"`, callees with
`"out"`) and `exclude_tests`; a new `package_deps` tool lists a package's direct
or transitive imports and importers; lookups rank exact names first, accept
`"<dir>:<Name>"`, and say when a name is shared; and the tool descriptions, the
`chiton install` rules block and the graph skill now give a recipe for each kind
of question.

| Run | Condition | Mean F1 | Exact | Median input tokens | chiton calls per question |
|---|---|---:|---:|---:|---:|
| Direct set, before the fixes (`b535d99`) | grep | 0.99 | 34 / 36 | 22 537 | – |
| | chiton | 0.99 | 35 / 36 | 42 386 | 0 |
| | chiton-only | 0.96 | 32 / 36 | 61 022 | 4.9 |
| Direct set, after the fixes (`e3b40c0`) | grep | 1.00 | 36 / 36 | 22 609 | – |
| | chiton | 1.00 | 36 / 36 | 32 061 | 0.7 |
| | chiton-only | 1.00 | 36 / 36 | 41 203 | 1.3 |

Every run, with its question set and ground truth, is in
[`benchmark-data/`](benchmark-data/): `answer-quality-direct-before-*`,
`answer-quality-direct-after-*` and `answer-quality-hard-*`.

### Reproduce

The harness is `internal/bench/qualbench`. It needs the `claude` CLI on `PATH`,
signed in. Runs count against your Claude plan.

```sh
# two clean checkouts of the commit under test
git clone <repo> grep-checkout
git clone <repo> chiton-checkout
cd chiton-checkout && chiton build . && chiton cluster && chiton install --assistant claude-code && cd ..

go run ./internal/bench/qualbench gen -set hard -repo grep-checkout -out questions.json   # or -set basic
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