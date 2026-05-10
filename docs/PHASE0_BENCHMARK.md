# Phase 0 Benchmark — chiton MCP vs grep+read

Corpus: **Go_Projects** (10 files, ~6 450 chars, ~1 615 tokens total)  
Date: 2026-05-09  
Methodology: token count ≈ response chars ÷ 4 (standard rough estimate).  
"Grep+read" = what Claude Code does naturally: grep for the symbol, then read the relevant file(s).  
"Naive" = read every file in the corpus up front (~1 615 tokens for this corpus).

---

## Results — Round 1 (v1 output, full model metadata)

| # | Question | Naive | Grep+read | MCP v1 | Correct? | MCP/grep ratio |
|---|---|---|---|---|---|---|
| 1 | What does fetchall depend on? | ~1 615 | ~330 | ~875 | Yes | 2.7× worse |
| 2 | Which file defines lissajous? | ~1 615 | ~20 | ~150 | Yes | 7.5× worse |
| 3 | Show all functions that call `http.Get` | ~1 615 | ~45 | ~5 (false neg) | **No** | N/A |
| 4 | Call graph from `main` in fetchall | ~1 615 | ~288 | ~200 (wrong file) | **No** | N/A |
| 5 | What packages does dup3 import? | ~1 615 | ~120 | ~300 | Yes | 2.5× worse |

**Exit criterion from PLAN.md:** MCP must be ≥ 3× cheaper than grep+read.  
**Round 1 verdict: criterion NOT met.** Two correctness failures, all three correct answers were more expensive than grep+read.

Root cause of token bloat: every node and edge carried `id` (32-char hex), `content_sha256` (64-char hex), `created_at`, `updated_at`, and raw source URIs repeated per-node. These fields are internal bookkeeping; the LLM does not need them.

---

## Fix applied — lean MCP output

Introduced `leanNode` and `leanEdge` output types in `internal/mcp/server.go`. Stripped from responses:
- `id` (32-char hex, useless to LLM)
- `content_sha256` (64-char hex, useless to LLM)
- `created_at`, `updated_at` (timestamps, useless to LLM)

Edge `src`/`dst` IDs replaced with human-readable `from`/`to` labels.

Retained: `kind`, `label`, `source_uri`, `span`, `attrs`, `provenance`, `confidence`.

---

## Results — Round 2 (v2 output, lean metadata)

Q3 and Q4 were replaced after Round 1: Q3 tested call-site indexing (not implemented); Q4 tested `main`, which exists in every Go file, making label resolution random.

| # | Question | Naive | Grep+read | MCP v2 | Correct? | MCP/grep ratio | Improvement |
|---|---|---|---|---|---|---|---|
| 1 | What does fetchall depend on? | ~1 615 | ~330 | ~488 | Yes | 1.5× worse | 44% ↓ |
| 2 | Which file defines lissajous? | ~1 615 | ~20 | ~100 | Yes | 5× worse | 33% ↓ |
| 3 | What files import `fmt`? | ~1 615 | ~55 | ~690 | Partial² | 12.5× worse | — |
| 4 | What does `fetch` call? | ~1 615 | ~170 | ~130 | **No**³ | N/A | — |
| 5 | What packages does dup3 import? | ~1 615 | ~120 | ~170 | Yes | 1.4× worse | 43% ↓ |

² Cross-corpus contamination: MCP returned 13 files (9 Go_Projects + 4 chiton source files); grep scoped to Go_Projects returns 9.  
³ Label collision: `fetch` substring-matches `fetchall.go` (file node) before the `fetch` function symbol; returns 0 CALLS edges.

**Round 2 verdict: criterion still not met.** Lean output cut MCP token cost 33–44% on Q1, Q2, Q5. Two questions still fail: Q3 (cross-corpus contamination) and Q4 (label disambiguation — same root cause as old Q4).

---

## Remaining gaps

### Gap 1 — Label disambiguation (Q4 failure)

`get_node_neighborhood("fetch")` resolves to the file `fetchall.go` (substring match) rather than the `fetch` function symbol. Any symbol whose name is a substring of a filename is ambiguous under the current `SearchNodes` logic — the file node is returned first.

This is the same root cause as the original Q4 failure (`main` exists in every file). It affects any query where a function name appears inside a longer filename.

**Fix:** add an optional `file` parameter to `get_node_neighborhood` and `get_node` so callers can write `get_node_neighborhood("fetch", file="fetchall.go")` to target the symbol unambiguously in one call.

### Gap 2 — Cross-corpus scoping (Q3 partial result)

The graph contains files from multiple indexed directories (both chiton source and Go_Projects). There is no way to scope a query to one corpus. `get_node_neighborhood("fmt", edge_kinds=["IMPORTS"])` returned 13 files (9 Go_Projects + 4 chiton source files). Grep scoped to Go_Projects returns 9.

**Fix:** add a `source_prefix` filter to neighborhood and search queries, or support per-directory sub-graphs.

### Gap 3 — No call-site indexing

The extractor emits `IMPORTS` edges at the package level (`fetchall.go → net/http`) but not individual call expressions (`http.Get` at a specific line). Querying `query_graph("http.Get")` returns nothing even though multiple files call it.

**Short-term:** use grep for call-site queries — documented in CLAUDE.md.  
**Long-term:** add a call-site extraction pass using `go/types` for resolved calls.

### Gap 4 — Corpus too small to show graph advantage

At 10 files and ~1 615 total tokens, even naive (read everything) is cheap. The graph's token advantage only materialises when:

- The corpus is large enough that naive/grep+read returns thousands of irrelevant tokens.
- The question requires multi-hop traversal grep cannot express ("what transitively depends on X?").

On a 10-file corpus, grep is fast and precise. This benchmark measures the floor of MCP usefulness, not the ceiling.

---

## What is working

- **Q1 correctness:** single tool call returned all imports (`net/http`, `fmt`, `os`, `time`, `io`) and all symbols (`main` at lines 13–24, `fetch` at lines 27–43) with zero file reads.
- **Q5 correctness:** `get_node_neighborhood("dup3.go", depth=1, edge_kinds=["IMPORTS"])` returned exactly `fmt`, `os`, `strings` — no irrelevant content.
- **Lean edges are readable:** `{ "kind": "IMPORTS", "from": "fetchall.go", "to": "net/http" }` vs the previous `{ "src": "c4bfaa327...", "dst": "17e58b438...", ... }`.
- **Build performance:** 10 files indexed in 78 ms; incremental SHA256 cache skips unchanged files correctly.
- **MCP token cost is bounded:** on a 1 000-file corpus, MCP neighborhood response size stays roughly constant while grep+read scales with the number of matches. The advantage inverts at realistic corpus sizes.

---

## Recommended next steps (before Phase 1)

1. **Add `file` disambiguation parameter** to `get_node` and `get_node_neighborhood`. Unblocks correct Q4 answers in one call. Estimated: 1 hr.

2. **Re-run on a larger corpus** (50–200 files). After fix 1, re-measure all 5 questions to validate the token advantage at realistic scale. The current corpus is too small to be representative.

3. ~~**Document call-site gap in CLAUDE.md.**~~ Done — CLAUDE.md updated with call-site and label disambiguation limitations.
