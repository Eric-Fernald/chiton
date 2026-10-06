# Spec: HNSW vector index

Status: PR 1 (R1 to R9) implemented on the `hnsw` branch; R10 not started. Audience: an implementing engineer or model with no prior context. Facts marked "verify" came from web summaries or memory and must be checked before relying on them.

## Implementation notes

Where the build departs from this spec, and why:

- **No coder/hnsw.** It does not compile on Windows: `SavedGraph` imports `google/renameio`, whose files are all `!windows`, which breaks this machine and the `windows-amd64` release. A vendored copy measured slower than `Flat` at 5k vectors (map adjacency, a sort and allocation per hop) with recall@8 of 0.81 at ef 64, and its delete left dangling links. `internal/vector/hnswlib` is a new implementation written from the paper. It shares no code with coder/hnsw.
- **Distance.** R3's custom distance is the built-in `hnswlib.UnitCosine` metric, so there is no function registry.
- **Extra key.** `vector.hnsw_ef_construction` sets the build-time candidate list, which this spec did not list.
- **Defaults.** `M` 16, `ef_search` 128, `ef_construction` 200, threshold 100k, from the sweep in `docs/BENCHMARK.md`.
- **Recall test data.** T2's random-vector test uses clustered "embedding-like" vectors. iid Gaussian vectors in 64 or more dimensions have almost no neighbour structure, so they are kept as a separate worst-case test (recall@8 at least 0.85).
- **Determinism.** The entry point is the live node with the highest level and lowest id, so identical queries return identical results, not only reproducible builds.
- **Race test.** T3's `-race` run needs cgo. The development machine has no C toolchain, so only CI runs it.
- **No blocking build.** R7 put the selected backend at all three call sites. Building HNSW takes about a minute per 100k vectors, and that would delay `serve`'s MCP handshake past assistant startup timeouts. So `serve` starts on Flat and swaps HNSW in from a background build (`vector.Swappable`), and a one-shot `chiton search` stays exact unless `vector.index` is `hnsw`.
- **Hardened decoding.** `ReadFrom` already validates every bound and cross-reference R10 will need, and has a fuzz test.
- **Mixed dimensions.** `buildIndex` sizes HNSW to the store's most common vector length rather than the embedder's `Dim()`, which reports the configured default until Ollama's first call.

## Goal

Add an HNSW approximate nearest-neighbor backend behind the existing `vector.Index` interface so semantic search stays fast past ~1M vectors. Keep the exact `Flat` index for small graphs, tests, and as a fallback. The graph traversal, reranking, and clustering code do not change.

## Current state

Checked against this repo at commit 3e0680d.

- `internal/vector/vector.go` defines `Index` with `Add`, `Search`, `Delete`, `Len`. `internal/vector/flat.go` is the only implementation. It is an exact cosine scan, single-threaded, O(n·d) per query. The package comment says it suits "a few hundred thousand vectors".
- `cmd/chiton/semantic.go` builds the index in `consumeIntoFlatIndex`. Three callers use it:
  - `cmd/chiton/main.go:968`, `buildSemanticDeps`, used by `chiton serve`.
  - `cmd/chiton/semantic.go:145`, `linkSimilarNodes`, used by `chiton embed --link-similar`.
  - `cmd/chiton/semantic.go:231`, `chiton search`.
- `serve` builds the index once at startup from `store.AllEmbeddings` and never mutates it afterward. Picking up new embeddings requires `chiton embed` and a restart.
- `linkSimilarNodes` calls `Search(vec, 6)` once per embedding and keeps pairs with cosine >= 0.85 (default). With `Flat` that is O(n²·d), which is the main reason it does not scale.
- `internal/graph/retrieve/retrieve.go` calls `index.Search(query, SeedK)` (default 8) and uses `Result.Score` as cosine similarity in [-1, 1] inside its rerank weights.
- Embeddings live in BadgerDB's `v:` keyspace as raw little-endian float32 (`internal/graph/store/badger/store.go:674`). Ollama `nomic-embed-text` produces 768 dimensions. The hash embedder produces 256.

`Flat` behavior the new backend must match:

- `Add` replaces a vector with the same ID and skips zero-length vectors.
- Vectors are L2-normalized on insert.
- `Search` returns nil when `k <= 0`, the index is empty, or the query is zero, empty, or has the wrong dimension.
- `Search` skips stored vectors whose dimension differs from the query (mixed embedders).
- Results sort by descending score, ties by ascending ID.
- `Delete` ignores unknown IDs.

## Library choice

Use `github.com/coder/hnsw` (CC0, pure Go). Verify each claim below against the repo before starting.

- It has a generic `Graph[K cmp.Ordered]` with `Add`, `Delete`, `Search`, `Lookup`, `Export`, `Import`, and a `SavedGraph` file wrapper. String node IDs satisfy `cmp.Ordered`.
- Its cosine and euclidean distances call `vek32` from `github.com/viterin/vek`. Verify whether `vek32` uses assembly SIMD and whether it falls back to pure Go on every release target.
- Defaults are `M=16`, `Ml=0.25`, `EfSearch=20`.
- `Add` panics on a dimension mismatch.
- It documents no thread-safety guarantees and has no mutex.
- The latest tag, v0.6.1, dates from June 2024. Commits on `main` from June 22, 2026 fix search termination (recall) and a hardcoded `CosineDistance` in `replenish()`. Do not use `@latest`. Pin a `main` commit at or after those fixes, confirm from `git log` that both are included, and record the SHA and the reason in the PR description.

Rejected alternatives:

- `hypermodeAI/hnsw`: a fork archived in October 2024.
- `TFMV/hnsw`: a fork whose last commit is March 2025, before the recall fix. It uses the same `vek32` distance functions, so its core speed should match.
- CGo libraries (hnswlib bindings, usearch): the default build is pure Go, with CGo only behind the `whisper_cgo` and `msgraph` tags.
- Server-based vector stores: chiton runs on-device with no server.

## Requirements

### R1. HNSW type

Add `vector.HNSW` in `internal/vector/hnsw.go`, implementing `Index`. It matches every `Flat` behavior listed above except that results are approximate.

### R2. Input guards

- Fix the expected dimension from the constructor argument, or from the first valid vector if none is given.
- Skip vectors with a different dimension, zero length, all-zero components, or NaN or Inf values. Never let them reach the library, since a mismatch panics.
- Count skipped vectors, expose the count, and log it once at load.
- `Search` returns nil for a wrong-dimension query.

### R3. Scores and distance

- Normalize on insert, as `Flat` does.
- Return `Score = 1 - Distance` so scores match `Flat`'s cosine range, because the rerank weights depend on it.
- Try a custom distance for unit vectors (`1 - vek32.Dot(a, b)`), registered with `RegisterDistanceFunc`. Benchmark it against the stock `CosineDistance` and keep the faster one if scores agree within 1e-5. The name must be registered before any `Import`.

### R4. Result order

Sort the k results by descending score, then ascending ID, to match `Flat`.

### R5. Determinism

Seed `Graph.Rng` with a fixed value so builds are reproducible. `AllEmbeddings` returns keys in Badger key order, so insertion order is already stable.

### R6. Concurrency

`serve` may run concurrent `Search` calls against a graph that no longer changes.

- Read `Search` in the chosen commit and list any shared mutable state.
- Add a `go test -race` test with many goroutines searching one frozen index.
- If either check finds shared state, guard `Search` with a `sync.RWMutex` read lock. Document that `Add` and `Delete` are not safe to run concurrently with `Search`, as the `Index` interface already states.

### R7. Selection and configuration

- Replace `consumeIntoFlatIndex` with a function that picks the backend, and use it at all three call sites.
- Add config keys, following the viper and env conventions in `internal/config/config.go`:
  - `vector.index`: `auto`, `flat`, or `hnsw`. Default `auto`.
  - `vector.hnsw_threshold`: `auto` uses HNSW at or above this vector count. Default 100000.
  - `vector.hnsw_m`: default 16.
  - `vector.hnsw_ef_search`: choose the default from the recall benchmark. The library's 20 is likely too low.
  - Env vars follow the existing pattern, for example `CHITON_VECTOR_INDEX`.
- Keep the existing memory behavior: the current function clears each raw `embs[i].Vec` after indexing so raw and normalized copies are never both live. The library stores its own copy of each vector. Measure peak heap during load and confirm it does not approach twice the vector data.

### R8. Logging

At `serve` startup, log the backend, vector count, skipped count, and build time.

### R9. `linkSimilarNodes`

- Use the same backend selection. The `Flat` version is O(n²·d) and unusable at scale.
- `SIMILAR_TO` edges are already `INFERRED`, so missing a few neighbors is acceptable. Note that in the command help.
- Add an `--exact` flag on `chiton embed` that forces `Flat`.

### R10. Persistence (separate second PR)

Rebuilding HNSW on every `serve` start takes minutes at millions of vectors, so save the graph.

- Write a sidecar file next to the graph store, following `cluster.SidecarPath` and `cluster.LoadSidecar`.
- Use `Export`/`Import` or `SavedGraph`. Write to a temporary file and rename, so a crash never leaves a partial file.
- Store a validity key with the graph: vector count, dimension, embedder model, and a hash of the sorted ID list. Load only when it matches the store.
- A missing, stale, or unreadable sidecar falls back to an in-memory rebuild with a warning, never an error.
- Have `chiton embed` write the sidecar, since it knows when embeddings change. `serve` only reads it.

## Testing and acceptance

- **T1. Conformance suite.** Run one set of behavior tests against both backends, covering replace-on-same-ID, delete, empty index, zero vectors, wrong-dimension query, `k <= 0`. Exact tie-order cases apply to `Flat`, and to HNSW only when the index is small enough that search is exhaustive.
- **T2. Recall.** Seeded test: at least 20k random unit vectors, plus a text set through the hash embedder. Recall@8 against `Flat` must be at least 0.95 at the default parameters.
- **T3. Race.** The concurrent-search test from R6, run under `go test -race`. CI has a separate race job that needs cgo; see `.github/workflows/ci.yml`.
- **T4. Guards.** Mixed dimensions, zero vectors, NaN, replace, delete then search.
- **T5. Benchmarks.** `go test -bench`, not run in CI. Report build time, query latency, and heap for 10k, 100k, and 1M vectors at 768 dimensions, HNSW against `Flat`. Targets to measure, not guarantees: p50 query under 10 ms at 1M vectors, recall@8 at least 0.95, heap overhead under 15% above the raw vector data. Report any target that is missed.
- **T6. Build matrix.** `CGO_ENABLED=0 go build ./...` for every target in `.github/workflows/release.yml`. That confirms the `vek32` dependency builds without CGo everywhere.
- **T7. End to end.** On this repo, run `chiton embed --provider hash`, then `chiton search` with `vector.index=flat` and with `hnsw`. Compare the top results.
- CI must stay green: `go vet ./...` and `go test ./... -count=1`.

## Docs to update

- The package comment in `internal/vector/vector.go` (the "few hundred thousand vectors" claim).
- `CLAUDE.md`, the semantic layer and config sections.
- Wherever config keys are documented (`docs/INSTALL.md`).
- `docs/BENCHMARK.md`, with the T5 results.

## Constraints

- The default build stays pure Go, with no CGo.
- Do not change `Flat`'s behavior.
- Match the surrounding code style. Comments are short and explain why, not what. Go doc comments start with the identifier name.
- Development is on Windows. Use PowerShell for commands. See `CONTRIBUTING.md` and the `Makefile` for build and test targets.

## Out of scope

- Quantization (int8, product quantization).
- Disk-backed or memory-mapped vectors.
- Loading vectors by community shard.
- Incremental index updates inside `serve`.
- Any change to Leiden clustering, the embedder, or the stored embedding format.

## Open questions for the implementer

1. Which `coder/hnsw` commit to pin, and confirmation that it includes both June 2026 fixes.
2. Whether `Search` is safe for concurrent readers, from the code and the race test.
3. Whether `vek32` builds and runs correctly on every release target.
4. Final defaults for `hnsw_ef_search` and `hnsw_threshold`, from the benchmark.
5. Whether `SavedGraph` or `Export`/`Import` fits the sidecar better.

## Delivery

- PR 1: R1 to R9, tests T1 to T7, docs.
- PR 2: R10, after PR 1 lands and the benchmark shows build time is a real problem.
