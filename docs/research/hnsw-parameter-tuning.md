# HNSW parameter tuning for chiton

## Question

What should chiton's HNSW parameters (M, efConstruction, efSearch), distance handling, neighbour selection, delete/update policy and sweep method be, at 1e3 to 1e6 vectors of 768-d (Ollama `nomic-embed-text`) or 256-d (hash) embeddings?

An answer names defaults with a primary-source reason for each, says whether `internal/vector/hnswlib` follows the paper and hnswlib, and says what the current sweep does and does not establish.

Repo state checked: branch `hnsw`, commits dd63596 (backend) and 620b883 (config wiring). Date 2026-10-06.

## Short answer

- Keep M=16, efConstruction=200, efSearch=128 for 768-d text embeddings up to ~1M vectors. They match hnswlib's own defaults (M=16, efC=200) and the paper's 5-48 range for M. Raise M to 32 only if a real-embedding sweep misses recall@8 >= 0.95; efSearch is the cheap knob.
- The implementation follows the paper on mL = 1/ln(M) and Mmax0 = 2M. It uses Algorithm 4 without extendCandidates or keepPrunedConnections, like hnswlib, with one real difference: hnswlib skips pruning when it has fewer than M candidates and this code always prunes (see 4).
- Deletes are tombstones with no compaction or counter. Replace-on-save leaks the old vector and degrades the graph. Since `serve` builds once and never mutates, this only matters if incremental updates are added; then rebuild when dead/total exceeds roughly 20-30% (a judgement, not a sourced figure).
- The current sweep varies only efSearch, uses one seed, 200 queries, mean latency, and synthetic data that is easy (recall 1.0 at 768-d). It cannot choose M or efConstruction. See 6.
- Decisions for you are listed under "Open questions".

## Findings

### 1. What the parameters do

- **M**: links per node. Paper: "A reasonable range of M is from 5 to 48"; smaller M is better for lower recall or lower-dimensional data, bigger for high recall or high-dimensional data; memory is proportional to M. (Paper section 4.1, arXiv 1603.09320.) hnswlib ALGO_PARAMS: range 2-100, "M=12-48 is ok for the most of the use cases", memory about M * 8-10 bytes per element. [hnswlib ALGO_PARAMS](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) (fetched through a summarizer; wording is paraphrase-level, medium confidence).
- **efConstruction**: candidate list size at insert. Paper: large enough that construction-time K-ANNS recall is "close to unity", "0.95 is enough for most use-cases"; beyond about 100 on 10M SIFT there is "little extra performance" for much longer build. ALGO_PARAMS: bigger means longer build and better quality, with diminishing returns; if recall at ef=efC is below 0.9 there is room to improve.
- **efSearch**: candidate list size per query, must be >= k. Higher is slower and more accurate (ALGO_PARAMS). Query-time only, so it is the knob to tune last and to expose in config.
- **Defaults elsewhere**: hnswlib Python M=16, efC=200 ([README](https://github.com/nmslib/hnswlib)); its C++ constructor sets `ef_ = 10` and `ef_construction_ = max(efC, M)` (hnswalg.h lines 120-121, master). Faiss memory per vector for IndexHNSW is `4*d + x*M*2*4` bytes ([Faiss wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)); the Faiss wiki gives no default numbers in what I could retrieve. Our `New` uses M16/efS64/efC128; `vector.HNSW` overrides to 16/128/200.
- **Memory here**: per node at 768-d, vector 3072 B plus layer-0 links 2*16*4 = 128 B plus about 1/16 of a node's upper-layer links. Links are ~4-5% of vector bytes, so M barely moves heap at 768-d; at 256-d it is ~12%. The spec's "under 15% overhead" target is reachable at either dimension (inferred from `graph.go` layout, `links [][][]int32`, slice headers add more; measure it).
- **Ranges by size (inferred, not sourced)**: no primary source gives size-specific defaults. The paper's complexity result is logarithmic in n, so the same M, efC work across 1e3-1e6; what changes is how much recall you can see on a given dataset. Below ~1e4 vectors exact `Flat` costs about 1 ms or less per query (measured below: 0.75 ms at 20k, 128-d), so the 100k auto threshold in `index.go` is sensible.

### 2. mL and M0

- Paper: "a simple choice for the optimal mL is 1/ln(M)"; Mmax0 = 2M autoselected, with M0 = M noted as clearly worse (Fig. 6; text partly truncated in my extraction). hnswlib: `mult_ = 1 / log(1.0 * M_)`, `maxM0_ = M_ * 2` (hnswalg.h lines 119, 152).
- Ours: `drawLevel` uses `-ln(1-U) * 1/ln(max(M,2))` floored (`graph.go:70`), `maxLinks(0)` returns 2M (`graph.go:63`). Same as hnswlib. `LevelMult` is exposed but unused by `vector.HNSW`. High confidence (read the code).
- New nodes pick M links at every layer including 0, and layer-0 lists are only shrunk to 2M on overflow (`graph.go:134-146,159`). hnswlib does the same (`getNeighborsByHeuristic2(top_candidates, M_)`, hnswalg.h line 545).

### 3. Distance metric

- Cosine on unit vectors equals dot product up to `1 - x`; L2 on unit vectors is `2 - 2*dot`, a monotonic transform, so all three rank identically on normalized input and the same M/ef values apply. This is arithmetic, not a citation.
- hnswlib's cosine space "currently returns normalized vectors" and its IP distance is `1 - dot` ([README](https://github.com/nmslib/hnswlib)), the same trick `UnitCosine` uses (`hnswlib/distance.go`). Normalizing on insert and query (as `vector.HNSW` does) is required for `UnitCosine` to be a true cosine.
- Where tuning does differ: un-normalized inner product is not a metric and graph search can fail on it; avoid. Nomic models are trained with cosine-style contrastive loss (inferred; the model card I fetched does not state it) and Matryoshka truncation (64-768) exists on v1.5 ([model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)). Truncating to e.g. 256-d would cut memory and distance cost 3x but must be re-normalized; chiton does not do it and it is outside this question.
- Search cost is dominated by distance evaluations. coder/hnsw's README reports ~70% of query time in the distance function at 1536-d. Our `Search` evaluates about efSearch * (links per visited node) distances; at ef=128, M0=32 that is on the order of thousands of 768-d dot products per query (inferred), consistent with the ~0.6 ms measured at 20k.

### 4. Neighbour selection

- Paper Algorithm 4 (heuristic) has `extendCandidates` (default false; "useful only for extremely clustered data") and `keepPrunedConnections` (refills to a fixed link count with discarded candidates). Heuristic versus simple selection matters most at high recall and on clustered data; on uniform very high-dimensional data there is little difference (paper section 4.1, Fig. 7).
- hnswlib implements neither flag. Its heuristic returns the candidate list unchanged when it has fewer than M entries (`if (top_candidates.size() < M) return;`, hnswalg.h lines 475-479) and otherwise keeps a candidate only if no kept link is closer to it than the base node is.
- Ours (`pickLinks`, `graph.go:176`) implements the same keep rule but has no "fewer than M, keep all" early return. When the candidate pool is small, such as early in the build, at upper layers with few nodes, or when tombstones thin `searchLayer` results, diversity pruning can leave a node with only one or two links. Paper's `keepPrunedConnections` or hnswlib's early return both address this. This is a plausible recall cost on clustered data (chiton's data: symbols, doc sections, near-duplicate code) but I did not measure it. Suggested experiment: add the early return, rerun the sweep.
- `extendCandidates`: skip. Neither hnswlib nor Faiss exposes it as a default and the paper limits it to extreme clustering.

### 5. Deletes, updates, incremental inserts

- Paper lists "element updates and removal" as future work and gives no algorithm (section 5). Faiss `IndexHNSW` does not support removal ("would destroy the graph structure", [Faiss wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)). hnswlib offers `mark_deleted` (excluded from results, still traversed), `replace_deleted` (new inserts reuse deleted slots) and in-place updates via `add_items` with an existing label, "slower than insertion" ([README](https://github.com/nmslib/hnswlib)); hnswlib's `updatePoint` repairs neighbour lists. coder/hnsw's README states delete cost O(M^2 log n) but says little else.
- Ours: `Remove` tombstones (`dead[id]=true`), keeps the node routing (`graph.go:286`), re-elects the entry point by an O(n) scan only when the entry is removed (`graph.go:303`). `Insert` on an existing key calls `Remove` then appends a new node (`graph.go:116`). Consequences, read from the code:
  - Nothing frees the old vector or link slices, so heap grows with every re-embed; there is no dead count or `Compact` to decide on.
  - Live nodes' link lists keep pointing at tombstones. Layer-0 slots wasted on dead nodes reduce live fan-out; `searchLayer` still expands tombstones, so latency rises and recall can fall as the dead fraction grows.
  - If all nearby nodes are dead, `searchLayer` can return nothing and `Insert` skips linking at that level (`graph.go:~143`), producing weakly connected new nodes.
- Context: `serve` builds once and never mutates (spec "Current state"); live sync (`chiton sync watch`) re-embeds changed files on save, but the index is only rebuilt at `serve` start. So today deletes and updates do not reach the index except through full rebuild.
- Policy (judgement, no primary source gives a threshold): treat the index as a cache. Rebuild from the store at `serve` start and after `chiton embed`. If incremental updates are added later, expose `Dead()` and rebuild when dead/total passes ~20-30%, or when replaced keys since the last rebuild exceed ~10% of live; also add a recall check on a held-out query set after rebuild.

### 6. Sweep method versus `hnsw_sweep_test.go`

Standard method (ann-benchmarks, [README](https://github.com/erikbern/ann-benchmarks)): fixed train/test split with brute-force ground truth for the test queries, recall at fixed k, single-threaded single queries, many parameter settings per library, report the recall-versus-throughput frontier. The summary I fetched says build time and index size are not part of the plotted output; I add them.

What `TestHNSWSweep` does: one index per run at env-chosen M/efC, sweeps ef in {16..256}, 200 queries from the same generator, recall@8 versus `Flat` (exact), mean ms/query, one seed. Gaps:

1. M and efC are not swept in one run (needs shell loops over env vars, as I did below). No recall-vs-efC study.
2. Data is the synthetic `embeddingLike` (48-d latent, projected). At 768-d it gives recall 1.0 from ef=32, so it cannot discriminate settings. The iid mode is the opposite extreme. Neither is real text embeddings. Real nomic vectors of this repo's own symbols/docs are the dataset that matters; `chiton embed` output in a Badger store is available.
3. Queries are held out, which is right for `chiton search`/retrieve, but `linkSimilarNodes` queries with the stored vectors themselves (`semantic.go:156`, `Search(e.Vec, 6)`); in-index queries are a different workload and should be a second query set.
4. Reports mean only; use p50/p95/p99. `TestVectorBenchReport` in the untracked `bench_test.go` already adds p50/p95, heap and build time. Good.
5. Single seed, no variance. Rebuild with 3 seeds (the `Seed` field exists) and report spread; 200 queries at k=8 gives about +/-0.01 on recall near 0.95 (inferred).
6. Ties and duplicates: near-duplicate vectors make "true top-8" ambiguous; compare by distance threshold (count approximate result with score >= exact 8th score minus 1e-6) as ann-benchmarks' epsilon-recall does, or dedupe.
7. Not run at 100k-1M; recall depends on n, so the 20k acceptance test (T2) says little about 1M.
8. Report: n, dim, dataset, M, efC, efS, k, recall@k mean and min/seed, p50/p95/p99 latency, build time and build-throughput, heap/raw ratio, Go version, CPU (AVX2 or not, since `vek32` falls back). Hold n, dim fixed per table.

### Measurements run for this note

`go test ./internal/vector -run TestHNSWSweep`, N=20000, one seed, this machine (Windows, `vek32`), unmodified code:

| Data | Dim | M | efC | ef=16 | 32 | 64 | 128 | 256 | build |
|---|---|---|---|---|---|---|---|---|---|
| embedding-like | 768 | 16 | 200 | 0.984 | 1.000 | 1.000 | 1.000 | 1.000 | 10.2 s |
| iid Gaussian | 128 | 8 | 100 | 0.095 | 0.156 | 0.245 | 0.405 | 0.608 | 1.3 s |
| iid Gaussian | 128 | 16 | 100 | 0.209 | 0.330 | 0.500 | 0.703 | 0.868 | 2.0 s |
| iid Gaussian | 128 | 16 | 200 | 0.217 | 0.334 | 0.504 | 0.708 | 0.887 | 3.3 s |
| iid Gaussian | 128 | 32 | 200 | 0.360 | 0.530 | 0.730 | 0.901 | 0.978 | 4.1 s |

Latency at ef=128: 0.57 ms (768-d embedding-like), 0.13-0.22 ms (iid 128-d); `Flat` 5.97 ms and 0.72-0.75 ms. Reading: on structured data every setting is fine and recall is saturated, so defaults there are chosen by cost; on hard (unstructured) data M dominates, efC from 100 to 200 gives ~0.5-2 points, and ef=128 at M=16 is below 0.95. Real text embeddings sit between; only a sweep on them settles M=16 versus 32.

### 7. Starting defaults for chiton

| Scale | M | efConstruction | efSearch (k=8) | Backend |
|---|---|---|---|---|
| < 1e5 | n/a | n/a | n/a | `Flat` (auto threshold 100k; exact, no build) |
| 1e5 - 1e6, 768-d | 16 | 200 | 128 | HNSW |
| same, if sweep on real data gives recall@8 < 0.95 | 32 | 200 | 128-256 | HNSW |
| `--link-similar` bulk | 16 | 200 | 64 | HNSW; edges are INFERRED, misses are acceptable per spec R9 |

These were the constants in `internal/vector/hnsw.go` when this note was written. `ef_construction` has since been lowered to 100 to halve build time; a rerun of the sweep in docs/BENCHMARK.md measured no recall change on embedding-like data and 2.5 points (0.894 to 0.869 at ef 128) on iid data. The build is also parallel now. Basis: M=16/efC=200 are hnswlib's defaults and inside the paper's range; efSearch=128 is a measured choice here (recall 1.0 on structured data, sub-millisecond), not a sourced one. The comment in `hnsw.go` says the defaults came from "the recall sweep in docs/BENCHMARK.md", but BENCHMARK.md contained no HNSW section when I checked (grep for hnsw/recall found nothing); update one or the other.

## Conflicts and caveats

- Spec says use `github.com/coder/hnsw`; the repo instead has its own `internal/vector/hnswlib` (from the paper) and the only new dependency is `github.com/viterin/vek v0.4.2` (`go.mod`). The task brief described these files as uncommitted; they are committed in dd63596 (git status at the time of writing shows only `bench_test.go` untracked). I trusted the files.
- Spec R3 says a faster custom unit-vector distance should match stock cosine within 1e-5; `UnitCosine` implements that.
- Source versus docs: hnswlib's README/ALGO_PARAMS say nothing about the "fewer than M candidates" shortcut; the source (hnswalg.h at master) has it. I trust the source. I did not pin a hnswlib release; line numbers are for master as fetched today and may drift.
- ANN web summaries came through a summarizing fetch tool; paper quotes were checked against text extracted from the arXiv PDF. The paper's Mmax0 sentence was cut by extraction; the "2M autoselected" part is confirmed from the Fig. 6 caption.
- The FAISS wiki pages did not state M/efSearch defaults in what I retrieved; I did not read FAISS source. (FAISS `IndexHNSW` defaults I recall are efConstruction=40, efSearch=16, but that is memory, unchecked.)
- All measurements are from one machine, one seed, synthetic data.

## Open questions

1. Should the sweep run on real `nomic-embed-text` vectors from this repo (chiton's own graph), with M in {16, 32} and 3 seeds, before the M default is final?
2. Add hnswlib's "fewer than M candidates, keep all" shortcut to `pickLinks`? It changes graph structure; verify recall before and after.
3. Is incremental update in `serve` planned? If not, add a note that tombstone growth is irrelevant; if so, add `Dead()`, compaction and a rebuild trigger.
4. Fix or rewrite the `hnsw.go` comment citing docs/BENCHMARK.md once real results exist.

## Sources

- Malkov and Yashunin, "Efficient and robust approximate nearest neighbor search using HNSW graphs": https://arxiv.org/abs/1603.09320 (PDF read locally)
- hnswlib ALGO_PARAMS: https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
- hnswlib README: https://github.com/nmslib/hnswlib
- hnswlib source: https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h (master, lines cited as fetched)
- Faiss index table: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes ; Indexing 1M vectors: https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors
- ann-benchmarks: https://github.com/erikbern/ann-benchmarks
- coder/hnsw README: https://github.com/coder/hnsw
- nomic-embed-text-v1.5 model card: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- Repo: `docs/specs/hnsw-index.md`, `internal/vector/hnsw.go`, `internal/vector/hnswlib/graph.go`, `internal/vector/hnswlib/distance.go`, `internal/vector/hnsw_sweep_test.go`, `internal/vector/datagen_test.go`, `internal/vector/index.go`, `internal/config/config.go`, `cmd/chiton/semantic.go:156`, `go.mod`
