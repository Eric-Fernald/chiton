# Chiton

Like the mollusk it's named after, chiton is many independent plates flexing as
one organism. AST extraction, semantic embeddings, Leiden clustering, a
deterministic + LLM rule engine, and an MCP server, fused into a single Go
binary that gives AI assistants a graph-grounded memory of your codebase.

chiton is an **on-device knowledge-graph engine**. It walks through your code, 
docs, and images, extracts a typed graph (symbols, files, calls, imports, doc 
sections, concepts, findings…), stores it locally, clusters it, and serves precise,
relationship-aware neighborhoods to assistants over the Model Context Protocol.
The point isn't cheaper context (although that's still a benefit with ~10% token savings),
it's *better* context: the model gets exactly what's relevant instead of a haystack of 
files to grep through.

> **Status:** actively developed, single-user and on-device first. Phases 0–3.5
> (core graph, rule engine, install layer, semantic layer, LLM rules + diff
> analysis, live sync, and broad multi-assistant installers) are implemented.
> Meeting/transcript ingestion and the HTML dashboard are on the roadmap (see
> [docs/PLAN.md](docs/PLAN.md)).

---

## Why a graph instead of grep?

Chiton answers structural and navigational questions with typed edges
(`CALLS`, `IMPORTS`, `CONTAINS`, `DESCRIBES`, `VIOLATES`, …) rather than raw file
dumps. Ask "what does `X` depend on?" or "how are `A` and `B` connected?" and the
assistant gets a small, structured subgraph — far fewer tokens, far less
attention dilution, and no manual cross-referencing.

Every node and edge carries **provenance** and **confidence**:

- `EXTRACTED` (confidence 1.0) — structural facts from the parser. Ground truth.
- `INFERRED` — LLM-extracted concepts and findings. Useful hints.
- `AMBIGUOUS` — below the confidence threshold; hidden by default.

When sources conflict, prefer `EXTRACTED` facts over `INFERRED` ones.

## How it works

```
  ingest            extract              store + index            serve
  ------            -------              -------------            -----
  file walk    →    Go AST (go/ast)  →   BadgerDB graph      →   MCP (stdio/HTTP)
  fsnotify          line scanners        (encrypted at rest)      14 graph + semantic
  git hooks         Markdown structure   pure-Go vector index     + analysis tools
                    image nodes          Leiden communities       CLI (thin wrapper)
                    rule engine          betweenness / god nodes
```

- **Single binary, zero runtime services.** Pure-Go throughout — BadgerDB for the
  graph store (native AES-256-GCM at rest), a pure-Go flat-cosine vector index,
  an in-process job queue, and a from-scratch Leiden implementation. No Redis, no
  CGo, no external database.
- **chiton never calls an LLM itself.** Semantic extraction and LLM rules run
  through your *host assistant's* subagents: chiton emits schema-pinned task
  specs, the assistant runs them, and the structured results are written back.
  No API key, no extra spend over your existing subscription. (A direct
  daemon-mode LLM path is planned but not yet shipped.)
- **Incremental by default.** A SHA-256 content cache skips unchanged files;
  live sync re-indexes only what changed and re-clusters just the touched
  neighborhood.

## Requirements

- **Go 1.25+** to build from source.
- **Optional: [Ollama](https://ollama.com)** for local embeddings (default model
  `nomic-embed-text`, served at `http://localhost:11434`). Set
  `CHITON_EMBED_PROVIDER=hash` for a deterministic offline/CI embedder that needs
  no model server.
- **Optional: an MCP-capable assistant** (Claude Code, GitHub Copilot in VS
  Code) to consume the graph and run extraction subagents.

## Install

Build from source:

```sh
go build -o chiton ./cmd/chiton
# or install onto your PATH:
go install github.com/Eric-Fernald/chiton/cmd/chiton@latest
```

`chiton --help` lists every subcommand.

## Quickstart

```sh
chiton init                 # scaffold .chiton/ (policy files + encryption keyfile)
chiton build .              # walk the tree, extract the graph → chiton-out/
chiton cluster              # Leiden communities + betweenness → cluster.json
chiton report               # write GRAPH_REPORT.md (god nodes + community summaries)
chiton query Server         # look up a node and its neighborhood from the CLI
chiton install              # register chiton's MCP server + rules with detected assistants
```

Then point your assistant at the MCP server (a `.mcp.json` is written by
`chiton install`) and ask architecture questions — it will call chiton's tools
instead of grepping. To enable natural-language search:

```sh
chiton embed .              # embed symbols/docs/concepts (Ollama by default)
chiton search "where do we handle auth?"
```

## CLI reference

| Command | Purpose |
|---|---|
| `chiton init` | Scaffold `.chiton/` (policy files, `.gitignore`, encryption keyfile). `--encrypt=false` for plaintext. |
| `chiton build <path>` | Walk a tree, extract code/docs/images into the graph (SHA-256 cache skips unchanged files). |
| `chiton query <label>` | Look up a node by label substring and print its neighborhood (`--depth`, `--edge-kinds`, `--json`). |
| `chiton cluster` | Run Leiden community detection + betweenness; writes `cluster.json` sidecar. |
| `chiton report` | Generate `GRAPH_REPORT.md` (god nodes, community summaries). Needs a prior `cluster`. |
| `chiton embed <path>` | Embed nodes for semantic search (`--provider`, `--all`, `--link-similar`). |
| `chiton search <query>` | Hybrid semantic + graph search from the CLI. |
| `chiton extract pending\|submit` | Emit semantic-extraction task specs / ingest the structured results. |
| `chiton analyze <path>` | Run deterministic rules; persist findings as graph nodes. `--diff` / `--since <ref>`, `--no-llm`. |
| `chiton analyze pending\|submit` | Emit LLM-rule task specs / ingest LLM findings. |
| `chiton findings list\|explain` | Inspect persisted findings (filter by severity, category, scope). |
| `chiton sync watch\|git\|once` | Live sync: fsnotify watcher, git-driven re-index, or one-shot reconcile. |
| `chiton hooks install\|uninstall\|status\|run` | Manage git hooks (gating pre-commit/pre-push; syncing post-commit/post-checkout). |
| `chiton install\|uninstall\|doctor` | Register/remove chiton with AI assistants; report install state. `--assistant <name>` (claude-code, copilot-vscode, cursor, codex, gemini-cli, opencode, aider, antigravity, copilot-cli), `all`, or `auto` (default: detected). |
| `chiton serve --stdio` | Start the MCP server over stdio (use `--http :7474` for debugging, `--writable` for submit tools). |

## MCP tools

The server (`chiton serve --stdio`) exposes these tools. The CLI is a thin local
wrapper over the same logic.

| Tool | Use when |
|---|---|
| `get_node` | Fetch a single node by label (exact or substring). |
| `get_node_neighborhood` | "What does X depend on / call / import?" — node + N-hop neighbors. |
| `query_graph` | Find nodes whose label matches a known substring. |
| `shortest_path` | "How are A and B connected?" |
| `god_nodes` | Highest-betweenness nodes — the architectural hubs (needs `cluster`). |
| `graph_summary` | Token-budgeted overview: counts by kind, communities, god nodes. |
| `semantic_search` | Natural-language questions, ranked by meaning then graph-expanded (needs `embed`). |
| `list_findings` / `explain_finding` | Rule-engine findings as graph nodes. AMBIGUOUS LLM findings hidden unless `include_ambiguous`. |
| `diff_findings` | `{new, resolved, changed}` versus a saved baseline. |
| `pending_extractions` / `submit_extraction` | Subagent semantic extraction: task specs in, concepts + edges out. |
| `pending_analyses` / `submit_analysis` | Subagent LLM-rule analysis: task specs in, findings out. |

> **Note:** `query_graph` matches node *labels*, not call sites — chiton indexes
> package-level `IMPORTS` but not individual call expressions, so use grep for
> "which lines call X?". `CALLS` edges connect symbol nodes, not file nodes.

Write tools (`submit_extraction`, `submit_analysis`) need a writable store: run
`chiton serve --writable`, or use the `chiton extract submit` / `chiton analyze
submit` CLI. The default read-only server lets `build`, `embed`, and `cluster`
run concurrently.

## The rule engine

Deterministic rules run free and inline via `chiton analyze` across eight
categories — security, bugs, quality, architecture, performance, reliability,
database, and style. Findings become `finding` nodes (`EXTRACTED`,
`rule_provenance=DETERMINISTIC`) with `VIOLATES → rule` and `AFFECTS → code`
edges; re-analyzing a file prunes findings that no longer fire, so the store
reflects live truth.

LLM rules (`llm/…`) cover the semantic cases AST rules can't reach
(abstraction leaks, contract-vs-comment drift, missing authz checks, N+1
queries, race/resource-leak risks, comment rot, …). They dispatch through the
same subagent path as semantic extraction, land as `INFERRED`/`AMBIGUOUS`
findings, and never block the pre-commit gate.

Wire findings into git with `chiton hooks install`; the per-repo
`.chiton/hooks.yaml` sets the block-on severity threshold per stage.

## Live sync

Instead of a full rebuild, keep the graph current incrementally:

```sh
chiton sync watch .          # fsnotify watcher: re-index files on save (Ctrl-C to stop cleanly)
chiton sync once .           # one-shot reconcile: re-index changed, prune removed
chiton sync git --since <ref>  # re-index files changed since a git ref
chiton hooks install --stage post-commit,post-checkout   # auto-sync on git ops
```

Each changed file is re-extracted (stale nodes/edges replaced) and — when enabled
(`--embed`, `--analyze`) — its embeddings and deterministic findings refreshed.
After a batch settles, the touched 2-hop neighborhood is incrementally
re-clustered. Betweenness is a global metric carried forward between full
`chiton cluster` passes.

> Stop the watcher with Ctrl-C so BadgerDB closes cleanly; a hard kill can leave
> the write-ahead log needing recovery on the next open.

## Configuration

chiton reads `.chitonrc` (TOML) from the working directory or `$HOME`, with
`CHITON_*` environment overrides (e.g. `CHITON_EMBED_PROVIDER=hash`).

| Key | Default | Notes |
|---|---|---|
| `graph_path` | `./chiton-out/graph…` | Where the BadgerDB store lives. |
| `log_level` | `info` | |
| `embed.provider` | `ollama` | `ollama` or `hash` (offline/CI). |
| `embed.model` | `nomic-embed-text` | |
| `embed.endpoint` | `http://localhost:11434` | Ollama endpoint. |
| `embed.dim` | `768` | Embedding dimension. |
| `semantic.confidence_threshold` | `0.5` | Splits `INFERRED` from `AMBIGUOUS`. |

The committed `.chiton/` files (`hooks.yaml`, `sources.yaml`) are the only shared
state — the graph store, cache, vector index, and keyfile are git-ignored.

## Encryption at rest

The graph is encrypted by default (AES-256-GCM via BadgerDB; key derived from a
passphrase with argon2id, salt stored in a sidecar keyfile next to the store).
The passphrase is read from `CHITON_PASSPHRASE` or prompted interactively. Opt
out for CI or synthetic data with `chiton init --encrypt=false`.

## Supported sources

- **Go** — full AST extraction via the standard-library parser (`go/ast`):
  files, symbols, `CONTAINS`, `IMPORTS`, and intra-corpus `CALLS` edges.
- **Python, JavaScript/TypeScript, Rust, Java** — lightweight line-scanning
  extractors for symbols and imports.
- **Markdown** — `doc_section` nodes per heading with `CONTAINS` nesting.
- **Images** — image nodes at build time, described via vision subagents.

## Architecture

```
chiton/
├── cmd/chiton/            CLI entrypoint (cobra)
├── internal/
│   ├── extract/ast/       per-language extractors + Markdown + images
│   ├── extract/subagent/  schema-pinned task specs for host-assistant extraction
│   ├── graph/store/       GraphStore interface + BadgerDB impl (Kuzu stub)
│   ├── graph/cluster/     Leiden community detection (full + incremental)
│   ├── graph/metrics/     betweenness, degree, god-node surfacing
│   ├── graph/retrieve/    hybrid vector + graph retrieval policy
│   ├── vector/            pure-Go flat-cosine ANN index
│   ├── embed/             embedding provider interface (Ollama + hash)
│   ├── analyze/           rule engine, findings, diff, LLM-rule dispatch
│   ├── ingest/            incremental updater + fsnotify + git front-ends
│   ├── crypto/            argon2id KDF + keyfile for encryption at rest
│   ├── install/           per-assistant installers (Claude Code, Copilot, Cursor, Codex, Gemini CLI, OpenCode, Aider, Antigravity, Copilot CLI)
│   ├── hooks/             git hook install + gating runner
│   ├── mcp/               MCP server and tool handlers
│   └── report/            GRAPH_REPORT.md generator
└── docs/                  PLAN.md (full roadmap), PHASE0_BENCHMARK.md
```

The `GraphStore` interface is the project's spine — every package depends on it,
not on a concrete store. BadgerDB ships in v1; Kuzu lives as a stub
(`internal/graph/store/kuzu/`) so it can slot in behind the same interface once
its Go bindings stabilize on Windows. The same boundary-first approach applies to
the vector index (flat today, HNSW/sqlite-vec deferred) and the job queue
(in-process today, distributed later).

## Roadmap

Shipped through **Phase 3.5** (live sync + broad multi-assistant installers:
Cursor, Codex, Gemini CLI, OpenCode, Aider, Antigravity, Copilot CLI). Next up:
meeting/transcript ingestion with decision-to-code linkage, URL/paper ingest, an
HTML graph export, and an honest cross-corpus benchmark suite. See
[docs/PLAN.md](docs/PLAN.md) for the full plan and design rationale.

## License

Proprietary — Copyright © 2026 Eric Fernald. All rights reserved. See
[LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Third-party dependencies remain
under their own licenses.
