# Phase 0 Benchmark — chiton MCP vs grep+read

Corpus: **Go_Projects** (10 files, ~5 800 chars, ~1 445 tokens total)
Date: 2026-05-09
Methodology: token count ≈ response chars ÷ 4 (standard rough estimate).
"Grep+read" = what Claude Code does naturally: grep for the symbol, then read the relevant file.
"Naive" = read every file in the corpus up front.

---

## Results

| # | Question | Naive | Grep+read | MCP | MCP correct? | Ratio (MCP/grep) |
|---|---|---|---|---|---|---|
| 1 | What does fetchall depend on? | ~1 445 | ~300 | ~875 | Yes | **2.9× worse** |
| 2 | Which file defines lissajous? | ~1 445 | ~25 | ~150 | Yes | **6× worse** |
| 3 | Show all functions that call `http.Get` | ~1 445 | ~50 | ~5 (false neg) | **No** | N/A |
| 4 | Call graph from `main` in fetchall | ~1 445 | ~300 | ~200 (wrong file) | **No** | N/A |
| 5 | What packages does dup3 import? | ~1 445 | ~120 | ~300 | Yes | **2.5× worse** |

**Exit criterion from PLAN.md:** MCP must be ≥ 3× cheaper than grep+read.
**Result: criterion NOT met.** Per the plan, stop and re-evaluate before building Phase 1.

---

## What went wrong

### 1. JSON metadata dominates small responses

Every MCP response carries UUIDs, full source URIs, ISO timestamps, and content SHA256 hashes for every node and edge. This bookkeeping is useful for the store internally but is noise to the LLM. On a small corpus, the metadata can exceed the actual informational content.

Example — Q5 ("dup3 imports"):
- Useful information: `fmt`, `os`, `strings` — about 20 chars.
- MCP response: ~1 200 chars, of which ~1 180 are metadata.

**Fix:** strip `id`, `content_sha256`, `created_at`, `updated_at` from MCP tool results. Keep `label`, `kind`, `source_uri`, `span`, `attrs`. Estimated savings: 60–70% of response size.

### 2. No call-site indexing (Q3 false negative)

The extractor tracks package-level `IMPORTS` edges (`fetchall.go → net/http`) but not individual call expressions (`http.Get` at a specific line). So `query_graph("http.Get")` returns nothing even though two files call it.

The plan noted this ("CALLS edges between functions within the same file only — cross-file CALLS resolution requires go/types; skipped here") but the gap is sharper than expected: even intra-package external calls (stdlib) aren't tracked.

**Fix (short-term):** document the limitation clearly; grep is the right tool for this class of query.
**Fix (long-term):** add a call-site pass using `go/types` for resolved calls, or emit `CALLS` edges to external package nodes (e.g. `fetchall.main → net/http`) which is coarser but at least answers "what does X call into".

### 3. Label ambiguity (Q4 wrong file)

`get_node_neighborhood("main")` matched `lissajous.go`'s `main` function (the first inserted) instead of `fetchall.go`'s. Every `main` package defines a `main` symbol; the graph has ten of them.

**Fix:** when the query is scoped to a file context (e.g. the user said "in fetchall"), the tool should accept a `file` hint and filter candidates to that source URI. Alternatively, `query_graph` should return all matches ranked by relevance so the caller can pick.

### 4. Corpus is too small to show graph advantage

At 10 files and ~1 445 total tokens, even the naive approach (read everything) is cheap. The graph's advantage only appears when:
- The corpus is large enough that reading all files would blow context (Corpus B: ~250k LOC, ~60 000+ tokens naive).
- The question requires multi-hop traversal that grep cannot express (e.g. "what transitively depends on X?").

On a 10-file corpus, grep is fast and precise. This benchmark measures the floor, not the ceiling.

---

## Recommended fixes before Phase 1

In priority order:

1. **Slim MCP output.** Strip internal metadata fields from tool responses. Target: responses should contain only what the LLM needs to answer the question. Expected improvement: 60–70% token reduction, which would flip Q1 and Q5 to MCP-wins.

2. **Add file-scoped disambiguation.** Accept an optional `file` parameter on `get_node` and `get_node_neighborhood` to resolve ambiguous labels like `main`.

3. **Re-run on a larger corpus.** After fix 1, re-run these 5 questions on a 50–100 file Go corpus to validate the token advantage at realistic scale. The current corpus is too small to be representative.

4. **Document the call-site gap.** Add a note to `CLAUDE.md` that `query_graph("http.Get")` won't work; use grep for call-site queries until cross-file CALLS resolution is implemented.

---

## What worked

- Q1 correctness: the neighborhood accurately identified all imports (`net/http`, `fmt`, `os`, `time`, `io`) and all symbols (`main`, `fetch`) from a single tool call with no file reads.
- Q2 correctness: `query_graph("lissajous")` returned the file name and exact line range (25–60) with zero file I/O.
- Q5 correctness: `get_node_neighborhood("dup3.go", depth=1, edge_kinds=[IMPORTS])` returned exactly `fmt`, `os`, `strings` — no irrelevant content.
- Build performance: 10 files indexed in 78 ms, incremental cache working correctly.

The graph structure is sound. The retrieval interface needs output trimming and disambiguation before it beats grep.
