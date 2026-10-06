# Chiton

Like the mollusk it's named after, chiton is many independent plates flexing as
one organism. AST extraction, semantic embeddings, Leiden clustering, a
deterministic + LLM rule engine, and an MCP server, fused into a single Go
binary that gives AI assistants a graph-grounded memory of your codebase.

chiton is an **on-device knowledge-graph engine**. It walks through your code, 
docs, and images, extracts a typed graph (symbols, files, calls, imports, doc 
sections, concepts, findings…), stores it locally, clusters it, and serves precise,
relationship-aware neighborhoods to assistants over the Model Context Protocol.
The point isn't cheaper context — measured against `grep`+`read` it lands roughly
at par, and it is only dramatically cheaper than dumping whole files (see
[docs/BENCHMARK.md](docs/BENCHMARK.md) for the numbers). The point is *better*
context: the model gets exactly what's relevant instead of a haystack of files
to grep through.

---

## Why a graph instead of grep?

Chiton answers structural and navigational questions with typed edges
(`CALLS`, `IMPORTS`, `CONTAINS`, `DESCRIBES`, `VIOLATES`, …) rather than raw file
dumps. Ask "what does `X` depend on?" or "how are `A` and `B` connected?" and the
assistant gets a small, structured subgraph — precise, with little attention
dilution and no manual cross-referencing. On raw token count it is dramatically
cheaper than dumping whole files and roughly on par with `grep`+`read` (see
[docs/BENCHMARK.md](docs/BENCHMARK.md) for honest, measured numbers); the real
gain is *better* context, not fewer tokens.

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
  fsnotify          line scanners        (encrypted at rest)      19 graph + semantic
  git hooks         Markdown structure   pure-Go vector index     + analysis tools
                    image nodes          Leiden communities       CLI (thin wrapper)
                    rule engine          betweenness / god nodes
```

- **Single binary, zero runtime services.** Pure-Go throughout — BadgerDB for the
  graph store (native AES-256 encryption at rest), a pure-Go vector index (exact
  flat scan, or an HNSW graph for large stores, built on every core and saved to
  disk),
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

- **Go 1.25.5 or newer** to build from source (`go.mod` pins `toolchain go1.26.4`, which
  any recent Go downloads automatically).
- **Optional: [Ollama](https://ollama.com)** for local embeddings (default model
  `nomic-embed-text`, served at `http://localhost:11434`). Set
  `CHITON_EMBED_PROVIDER=hash` for a deterministic offline/CI embedder that needs
  no model server.
- **Optional: an MCP-capable assistant** (Claude Code, GitHub Copilot in VS
  Code) to consume the graph and run extraction subagents.

## Install

chiton is proprietary and the repository is private, so installing means
building from source with repository access:

```sh
git clone https://github.com/Eric-Fernald/chiton.git
cd chiton
make build              # or: go build ./cmd/chiton
```

`chiton --help` lists every subcommand.

Microsoft Graph support — `chiton teams fetch`, `chiton teams serve-webhook`,
and `chiton email outlook` — sits behind the `msgraph` build tag and is **not**
in the default build or in any release artifact. The Graph SDK's generated
request builders cost roughly 120 MB for the two API calls chiton makes through
them (a stripped default `windows/amd64` binary is 24.0 MB; with the tag it is
144.0 MB), so they are opt-in:

```sh
make build-msgraph      # or: go build -tags msgraph ./cmd/chiton
```

Without the tag those three commands still appear in `--help` and fail with a
message telling you to rebuild. Everything else is unaffected, including
`chiton email ingest` and `chiton email gmail`.

Other useful targets: `make check` (gofmt + vet + test), `make check-msgraph`
(keeps the tag-gated path compiling and passing), and `make release`, which
cross-compiles the release binaries into `dist/` and writes
`dist/checksums.txt` — the install scripts refuse a binary that file omits.

## Use it as a Claude Code plugin

The repository doubles as a Claude Code plugin and as a single-plugin
marketplace, so Claude Code can install it in two commands:

```sh
claude plugin marketplace add Eric-Fernald/chiton
claude plugin install chiton@chiton-marketplace
```

The repository is private, so both commands go through your existing git
credential setup. `owner/repo` shorthand clones over SSH by default; set
`CLAUDE_CODE_PLUGIN_PREFER_HTTPS=1` to use HTTPS instead. For background
auto-updates against a private repo, use SSH with a key loaded in `ssh-agent`,
or run `gh auth setup-git` and set
`CLAUDE_CODE_PLUGIN_KEEP_MARKETPLACE_ON_FAILURE=1` — a bare `GITHUB_TOKEN` in
the environment is not enough on its own.

**The plugin does not ship the binary.** It registers the MCP server, and that
server is launched by bare name (`chiton serve --stdio`), so `chiton` has to be
resolvable on `PATH` before any graph tool answers. Run `/chiton:setup`, which
walks through locating or installing it, or install it directly with
`scripts/install-chiton.sh` (`scripts/install-chiton.ps1` on Windows): each
detects OS/arch, downloads the matching release asset, verifies its SHA-256
against the release's `checksums.txt`, refuses to install a binary that checksum
file does not list, and prints the line to add to your shell profile when its
install directory is not already on `PATH`.

What the plugin provides:

- **Three skills** — `/chiton:graph` (when to reach for the graph instead of
  grep), `/chiton:subagent-tasks` (semantic extraction, LLM rules, meeting→code
  linkage), and `/chiton:setup` (first-run install, build, verify).
- **The MCP server**, read-only over stdio by default. The four write tools
  (`submit_extraction`, `submit_analysis`, `submit_meeting_links`, `add_url`)
  report themselves unavailable until you run `chiton serve --writable` or use
  the CLI write path.
- **`bin/` shims** that put `chiton` on the Bash tool's `PATH`, so the assistant
  can run CLI commands even when the binary lives somewhere your own `PATH`
  does not cover.

The plugin's MCP entry sets `CHITON_PROJECT_DIR` to the Claude Code project
directory. chiton anchors `.chitonrc` lookup and a relative `graph_path` to that
directory instead of to the server process's working directory — which is what
keeps an assistant-launched server reading the same store `chiton build` wrote.

### Plugin or `chiton install`?

| | plugin | `chiton install` |
|---|---|---|
| Assistants | Claude Code only | nine, including Claude Code |
| Setup | two commands, versioned through the marketplace | run per repo from a working binary |
| Writes into your repo | nothing | `.mcp.json` (or the assistant's equivalent), a rules block, and — for Claude Code — skills |
| Skills | `/chiton:graph`, `/chiton:subagent-tasks`, `/chiton:setup` | Claude Code only: `chiton` and `chiton-subagent-tasks` |

Both paths read the same skill markdown, embedded from the repository's
`skills/` directory, so their content cannot drift apart. They are not mutually
exclusive, but running both in one repo gives Claude Code two copies of the same
guidance.

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
chiton embed                # embed symbols/docs/concepts (Ollama by default)
chiton search "where do we handle auth?"
```

## CLI reference

| Command | Purpose |
|---|---|
| `chiton init` | Scaffold `.chiton/` (policy files, `.gitignore`, encryption keyfile). `--encrypt=false` for plaintext. |
| `chiton build <path>` | Walk a tree, extract code/docs/images into the graph (SHA-256 cache skips unchanged files). Skips VCS, `node_modules`, `vendor`, virtualenv, `target`/`dist` and chiton output directories, plus anything in a repo-root `.chitonignore`. Removes symbols and files deleted since the last build (`--prune`, default true) and extracts in parallel. |
| `chiton query <label>` | Look up a node by label substring and print its neighborhood (`--depth`, `--edge-kinds`, `--json`). |
| `chiton cluster` | Run Leiden community detection + betweenness; writes `cluster.json` sidecar. |
| `chiton report` | Generate `GRAPH_REPORT.md` (god nodes, community summaries). Needs a prior `cluster`. |
| `chiton visualize` | Export an interactive HTML graph (`--format html`) or a Mermaid diagram (`--format mermaid`). `--scope`, `--max-nodes`. |
| `chiton dashboard build` | Generate a read-only static dashboard (interactive graph + findings explorer) into a directory (`--out`, default `chiton-dashboard/`). `--scope`, `--max-nodes`, `--title`, `--include-ambiguous`. Not a server — open `index.html` directly. |
| `chiton export` | Write the whole graph to a portable JSON snapshot (`--out`, default stdout). |
| `chiton merge <snap>…` | Merge exported snapshot(s) into the current graph (cross-repo merging; idempotent, dedups shared nodes). Alias: `import`. |
| `chiton bench [path]` | Benchmark token cost (chiton vs naive vs grep+read); `--synth N` for a synthetic corpus, `--json`. |
| `chiton version` | Print version + build metadata. |
| `chiton embed` | Embed nodes for semantic search (`--provider`, `--all`, `--link-similar`, `--exact`). Takes no path — it always covers the whole store. For stores of 100k+ vectors it also builds and saves the HNSW index (`hnsw.idx`) that `serve` loads, even when nothing new needed embedding. |
| `chiton search <query>` | Hybrid semantic + graph search from the CLI. |
| `chiton extract pending\|submit` | Emit semantic-extraction task specs / ingest the structured results. |
| `chiton analyze <path>` | Run deterministic rules; persist findings as graph nodes. `--diff` / `--since <ref>`, `--no-llm`. |
| `chiton analyze pending\|submit` | Emit LLM-rule task specs / ingest LLM findings. |
| `chiton findings list\|explain` | Inspect persisted findings (filter by severity, category, scope). |
| `chiton sync watch\|git\|once` | Live sync: fsnotify watcher, git-driven re-index, or one-shot reconcile. |
| `chiton hooks install\|uninstall\|status\|run` | Manage git hooks (gating pre-commit/pre-push; syncing post-commit/post-checkout). |
| `chiton install\|uninstall\|doctor` | Register/remove chiton with AI assistants; report install state. `--assistant <name>` (claude-code, copilot-vscode, cursor, codex, gemini-cli, opencode, aider, antigravity, copilot-cli), `all`, or `auto` (default: detected). |
| `chiton meeting ingest\|transcribe\|pending\|submit` | Ingest meeting transcripts (`.vtt`/`.srt`/`.txt`), whisper-transcribe audio first, or emit/ingest meeting→code linkage task specs. |
| `chiton add-url <url-or-path>` | Ingest a web page, PDF, or YouTube transcript as a `url` node with `doc_section` children (`--kind`, `--author`, `--contributor`). |
| `chiton teams fetch\|serve-webhook` | Fetch + ingest all transcripts for one Teams meeting (`--meeting <id>`), or run the Graph change-notification webhook receiver. Needs `CHITON_TEAMS_{TENANT_ID,CLIENT_ID,CLIENT_SECRET,USER_ID}` and a `-tags msgraph` build. The receiver refuses to start without `CHITON_TEAMS_CLIENT_STATE`; `--addr` defaults to `127.0.0.1:8444`. |
| `chiton slack fetch --channel <id>` | Fetch a Slack channel — or one thread with `--thread <ts>` — and ingest it as a meeting subgraph (`CHITON_SLACK_TOKEN`). |
| `chiton slack serve-events` | Run the Slack Events API receiver: a new message re-ingests its thread (or channel). Refuses to start without `CHITON_SLACK_SIGNING_SECRET`; `--addr` defaults to `127.0.0.1:8445`. |
| `chiton browser ingest\|watch` | Ingest local browser history (Chrome/Edge/Brave) as `url` nodes; `watch` re-ingests on change. `--db <History>`, `--browser`, `--since`, `--min-visits`, `--keep-query`. |
| `chiton email ingest\|gmail\|outlook` | Ingest email: local `.eml`/`.mbox` files, the Gmail API (`CHITON_GMAIL_TOKEN`, `--query`), or Microsoft Graph (`CHITON_OUTLOOK_*`, `--max`; needs a `-tags msgraph` build). |
| `chiton serve` | Start the MCP server: stdio by default (`--stdio` still accepted), `--http 127.0.0.1:7474` (SSE / shared daemon; needs an explicit address), or `--unix <path>` (socket). `--writable` for submit tools. `--include-pii` returns people's names and addresses from the graph tools, which redact them by default. HTTP requires the bearer token from `<graph dir>/mcp-token`. |

> **Note:** the two event receivers — `teams serve-webhook` and `slack
> serve-events` — bind loopback by default and print a warning when `--addr`
> points at a wildcard address, since both providers need a public HTTPS URL and
> the intended shape is a tunnel or reverse proxy in front of localhost. Each
> also fails fast without its shared secret rather than starting a receiver that
> cannot authenticate a caller.

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
| `search_meetings` | Find ingested meetings by content, with their decisions and linked code symbols. Participant names (PII) hidden unless `include_pii`. |
| `decisions_about` | Decisions made about a topic or code symbol — rationale, the meeting they were decided in, and the code they govern. |
| `pending_meeting_links` / `submit_meeting_links` | Subagent meeting→code linkage: transcript + candidate-symbol task specs in, decisions + edges out. |
| `add_url` | Fetch a URL (article, paper PDF, or YouTube transcript) and ingest it as a `url` node with `doc_section` children. |

> **Note:** `query_graph` matches node *labels*, not call sites — chiton indexes
> package-level `IMPORTS` but not individual call expressions, so use grep for
> "which lines call X?". `CALLS` edges connect symbol nodes, not file nodes.

Write tools (`submit_extraction`, `submit_analysis`, `submit_meeting_links`,
`add_url`) need a writable store: run `chiton serve --writable`, or use the CLI
write path (`chiton extract submit`, `chiton analyze submit`, `chiton meeting
submit`, `chiton add-url`). The default read-only server shares the store with
other read-only commands (`query`, `cluster`, `search`). On Linux/macOS, writers
such as `build` and `embed` can also run beside it (restart the server to see
their changes). On Windows they fail fast with `in use by another chiton
process` until the server exits, so submit through `chiton serve --writable`
there.

The server speaks the same tool surface over three transports: stdio
(the default; assistant-launched), `--http 127.0.0.1:7474` (SSE; a shared daemon
across sessions or for debugging, started only when `--http <addr>` is given), and `--unix <path>` (a Unix-domain socket for
Linux/macOS power users). Full reference: [docs/MCP.md](docs/MCP.md).

## Assistant support

`chiton install` registers the MCP server and an always-on rules block. Run it
with no flags to auto-detect assistants in the workspace, `--assistant all` for
every one, `--assistant <name>` for a specific one, or `--user-scope` for
user-global config. `chiton doctor` reports the state; `chiton uninstall`
reverses it.

Skills are Claude-Code-specific: only that installer writes them, and it writes
two — `.claude/skills/chiton/` (graph queries) and
`.claude/skills/chiton-subagent-tasks/` (extraction, LLM rules, meeting→code
linkage). The other eight assistants get the MCP registration and the rules
block, which is the surface they support.

| Assistant (`--assistant`) | MCP config | Rules file | MCP? |
|---|---|---|---|
| `claude-code` | `.mcp.json` | `CLAUDE.md` + `.claude/skills/{chiton,chiton-subagent-tasks}/` | yes |
| `copilot-vscode` | `.vscode/mcp.json` | `.github/copilot-instructions.md` | yes |
| `cursor` | `.cursor/mcp.json` | `.cursor/rules/chiton.mdc` | yes |
| `codex` | `~/.codex/config.toml` | `AGENTS.md` | yes |
| `gemini-cli` | `.gemini/settings.json` | `GEMINI.md` | yes |
| `opencode` | `opencode.json` (`mcp`) | `AGENTS.md` | yes |
| `antigravity` | `~/.gemini/config/mcp_config.json` | `AGENTS.md` | yes |
| `copilot-cli` | `~/.copilot/mcp-config.json` | `AGENTS.md` | yes |
| `aider` | `.aider.conf.yml` (`read:`) | `CONVENTIONS.md` | no¹ |

¹ Aider has no native MCP yet, so chiton wires its rules/CLI fallback only. Full
per-platform paths, caveats, and troubleshooting: [docs/INSTALL.md](docs/INSTALL.md).
The slash-command reference is in [docs/SKILL.md](docs/SKILL.md).

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
`.chiton/hooks.yaml` sets the block-on severity threshold per stage. The
pre-commit gate checks the staged content of the files in the commit (file
rules only), pre-push scans the whole tree, and both run in memory. A locked or
passphrase-protected store, a git error, or a run past `timeout-seconds` prints
a warning instead of blocking, and a bad policy file falls back to the defaults.

## Live sync

Instead of a full rebuild, keep the graph current incrementally:

```sh
chiton sync watch .          # fsnotify watcher: re-index files on save (Ctrl-C to stop cleanly)
chiton sync once .           # one-shot reconcile: re-index changed, prune removed
chiton sync git --since <ref>  # re-index files changed since a git ref
chiton hooks install --stage post-commit,post-checkout   # auto-sync on git ops
```

Each changed file is re-extracted (stale nodes/edges replaced; edges meetings
and concepts attached to surviving symbols are kept), its deterministic
findings refreshed (`--analyze`, on by default), and — with `--embed` — its
embeddings refreshed. Re-embedding makes the saved HNSW index stale; run
`chiton embed` afterwards to save a current one.
After a batch settles, the touched 2-hop neighborhood is incrementally
re-clustered. Betweenness is a global metric carried forward between full
`chiton cluster` passes.
If the OS drops file events (a bulk checkout can overflow the Windows buffer),
`sync watch` rescans the tree automatically.

### Ignoring paths

`sync`, `sync watch` and `bench` skip these directory names at any depth:
`.git`, `.hg`, `.svn`, `.chiton`, `chiton-out`, `node_modules`, `vendor`,
`.venv`, `venv`, `__pycache__`, `target`, `dist`. Symlinks are never followed.

A repo-root `.chitonignore` adds more, one entry per line:

- `name` or a glob such as `*.pb.go` matches at any depth; `#` starts a comment.
- A trailing `/` matches directories only (`build/`).
- A leading or inner `/` anchors the pattern to the repo root (`/gen/`, `docs/api`).
- `!name` re-includes a built-in directory (`!vendor`).
- `**` is not supported.

`sync watch` reads the file at start, so restart it after an edit.

> Stop the watcher with Ctrl-C so BadgerDB closes cleanly; a hard kill can leave
> the write-ahead log needing recovery on the next open.

## Visualizing & sharing the graph

```sh
chiton visualize --format html --out graph.html   # interactive, pan/zoom/search (vis-network)
chiton visualize --format mermaid --out graph.mmd  # Mermaid flowchart for Markdown docs
chiton dashboard build                             # static dashboard: graph + findings explorer → chiton-dashboard/
chiton export --out graph.json                     # portable JSON snapshot of the whole graph
chiton merge other-repo.json                       # merge another repo's snapshot into this graph
```

The HTML export is a self-contained page: nodes coloured by community, sized by
degree, with a live search box. Mermaid output drops straight into a Markdown
file a renderer (GitHub, Obsidian, mkdocs) draws inline; both honour `--scope`
and `--max-nodes`. `dashboard build` adds a read-only **findings explorer** next
to that graph — filter findings by severity, category, community, file, and
provenance, and click any finding to focus its node in the graph — written as
static `index.html` you open directly (no server). `export`/`merge` are the
portable interchange path — node IDs are stable content hashes, so merging
several repos' snapshots into one store dedups shared nodes and builds a
**cross-repo** graph. The honest token-cost benchmark lives in
[docs/BENCHMARK.md](docs/BENCHMARK.md) (`chiton bench`).

## Configuration

chiton reads `.chitonrc` (TOML) from the project directory or `$HOME`, with
`CHITON_*` environment overrides (e.g. `CHITON_EMBED_PROVIDER=hash`). The
project directory is `$CHITON_PROJECT_DIR` when that is set — the Claude Code
plugin sets it — and the process working directory otherwise. When it is set, a
relative `graph_path` resolves against it too, so an assistant-launched server
cannot drift onto a different store than `chiton build` wrote. The `.chiton/` state directory (keyfile lookup, `analyze --diff` baselines, LLM-rule state) lives under it too. A `.chitonrc` that exists but fails to parse is an
error that names the file.

| Key | Default | Notes |
|---|---|---|
| `graph_path` | `./chiton-out/graph.kuzu` | Where the BadgerDB store lives (the name is historical). |
| `log_level` | `info` | |
| `embed.provider` | `ollama` | `ollama` or `hash` (offline/CI). |
| `embed.model` | `nomic-embed-text` | |
| `embed.endpoint` | `http://localhost:11434` | Ollama endpoint. A non-loopback value in the project's `.chitonrc` is refused; set `CHITON_EMBED_ENDPOINT` (or `$HOME/.chitonrc`, read only when the project has none) to use a remote server. |
| `embed.dim` | *(unset)* | Embedding dimension. Deliberately has no default: left at 0, each provider applies its native size (Ollama 768, hash 256) instead of the Ollama size being forced onto the hash provider. Set it only to override. |
| `semantic.confidence_threshold` | `0.5` | Splits `INFERRED` from `AMBIGUOUS`. |
| `vector.index` | `auto` | Index behind semantic search: `flat` (exact scan), `hnsw` (approximate graph), or `auto`, which uses `hnsw` at or above `vector.hnsw_threshold` vectors. `chiton embed` builds the `hnsw` graph and saves it as `hnsw.idx` beside the store, and `serve` loads it at startup. If it is missing or out of date, `serve` starts on `flat` and builds `hnsw` in the background, then switches over; rerun `chiton embed` to save it again. A one-shot `chiton search` stays exact unless this is set to `hnsw`. |
| `vector.hnsw_threshold` | `100000` | Vector count at which `auto` switches to `hnsw`. 0 or less means the default. |
| `vector.hnsw_m` | `16` | Maximum links per HNSW node (twice that on the bottom layer). Higher raises recall, memory, and build time. |
| `vector.hnsw_ef_search` | `128` | HNSW candidate list per query. Higher raises recall and query time. |
| `vector.hnsw_ef_construction` | `100` | HNSW candidate list while building. Higher raises graph quality and build time. |

The committed `.chiton/` files (`hooks.yaml`, `sources.yaml`) are the only shared
state — `chiton init` writes a `.chiton/.gitignore` that keeps everything else
under `.chiton/` (including the encryption keyfile) out of git. The graph store,
cache, cluster sidecar, and saved HNSW index (`hnsw.idx`) live under
`chiton-out/`, which is **not** auto-ignored — add `chiton-out/` to your repo's
`.gitignore`.

## Encryption at rest

The graph is encrypted by default with BadgerDB's native AES-256 encryption
(CTR mode), under a key derived from a passphrase with argon2id. The salt and KDF
parameters live in `.chiton/keyfile.json`; the passphrase and key are never
written to disk. The passphrase is read from `CHITON_PASSPHRASE` or prompted
interactively. An assistant-launched `chiton serve --stdio` cannot prompt, so export `CHITON_PASSPHRASE` in the environment that starts the assistant. The saved HNSW index is sealed too (AES-256-GCM, under a key
derived from the store's); the other files in `chiton-out/` and `.chiton/` stay
plaintext. Opt out for CI or synthetic data with `chiton init --encrypt=false`.

## Supported sources

- **Go** — full AST extraction via the standard-library parser (`go/ast`):
  files, symbols, `CONTAINS`, `IMPORTS`, and intra-corpus `CALLS` edges.
- **Python, JavaScript/TypeScript, Rust, Java** — lightweight line-scanning
  extractors for symbols and imports.
- **Markdown** — `doc_section` nodes per heading with `CONTAINS` nesting.
- **Images** — image nodes at build time, described via vision subagents.
- **Meetings** — Teams/Zoom/whisper transcripts (WebVTT/SRT) become `meeting`,
  `meeting_turn`, and `person` nodes; subagent linkage ties the decisions made
  in them back to the code symbols they reference (`chiton meeting`).
- **Slack** — channels and threads via `chiton slack fetch`. Each conversation
  becomes the same `meeting` + `meeting_turn` + `person` subgraph a transcript
  does, keyed on a stable `slack://` URI so re-ingest replaces rather than
  duplicates — which means `search_meetings` and `decisions_about` serve Slack
  discussions unchanged.
- **URLs** — articles, PDFs, and YouTube transcripts via `chiton add-url`.
- **Email** — local `.eml`/`.mbox`, Gmail, or Outlook (`chiton email`). Messages
  become `email` nodes with `doc_section` body children — so semantic extraction
  picks up their concepts for free — plus PII-tagged `person` participants that
  share the meeting subsystem's `person://` scheme, keyed on the address.
- **Browser history** — Chrome/Edge/Brave (`chiton browser`). Only the local
  History database's page *metadata* is read (title, visit count, last visit),
  as `url` nodes; chiton never fetches the pages, so this does no network I/O.
  URLs are stored without userinfo, query or fragment unless you pass `--keep-query`.

Encryption at rest, secrets, and meeting PII are covered in
[docs/SECURITY.md](docs/SECURITY.md).

## Architecture

```
chiton/
├── cmd/chiton/            CLI entrypoint (cobra)
├── internal/
│   ├── extract/ast/       per-language extractors + Markdown + images
│   ├── extract/meeting/   transcript parsing, meeting subgraphs, code linkage
│   ├── extract/subagent/  schema-pinned task specs for host-assistant extraction
│   ├── graph/model/       node/edge kinds, provenance, confidence
│   ├── graph/store/       GraphStore interface + BadgerDB impl (Kuzu stub)
│   ├── graph/cluster/     Leiden community detection (full + incremental)
│   ├── graph/metrics/     betweenness, degree, god-node surfacing
│   ├── graph/retrieve/    hybrid vector + graph retrieval policy
│   ├── graph/portable/    JSON snapshot read/merge (export, cross-repo merging)
│   ├── vector/            nearest-neighbour index: flat scan + HNSW (hnswlib/)
│   ├── embed/             embedding provider interface (Ollama + hash)
│   ├── analyze/           rule engine, findings, diff, LLM-rule dispatch
│   ├── ingest/            incremental updater + source front-ends:
│   │                        fswatch/ git/ audio/ url/ browser/ email/ slack/
│   │                        teams/  (teams + email's Outlook source: -tags msgraph)
│   │                        plus shared chunk/ (doc splitting) and bearer/ (HTTP auth)
│   ├── cache/             content-SHA-256 cache that skips unchanged files
│   ├── config/            .chitonrc + CHITON_* env, CHITON_PROJECT_DIR anchoring
│   ├── jobs/              in-process debouncing/coalescing job queue
│   ├── crypto/            argon2id KDF + keyfile for encryption at rest, file sealing
│   ├── fsutil/            atomic file writes
│   ├── install/           per-assistant installers (Claude Code, Copilot, Cursor, Codex, Gemini CLI, OpenCode, Aider, Antigravity, Copilot CLI)
│   ├── hooks/             git hook install + gating runner
│   ├── mcp/               MCP server and tool handlers (stdio / SSE / unix)
│   ├── report/            GRAPH_REPORT.md + interactive HTML + Mermaid + dashboard
│   └── bench/             token-cost benchmark harness + synthetic corpus
├── skills/                skill markdown, embedded (plugin + `chiton install` share it)
├── .claude-plugin/        plugin.json + marketplace.json
├── bin/                   chiton shims that resolve the binary for the plugin
├── scripts/               install-chiton.sh / .ps1 (release download + checksum)
└── docs/                  PLAN.md, INSTALL.md, MCP.md, SKILL.md, SECURITY.md, BENCHMARK.md
```

The `GraphStore` interface is the project's spine — every package depends on it,
not on a concrete store. BadgerDB ships in v1; Kuzu lives as a stub
(`internal/graph/store/kuzu/`) so it can slot in behind the same interface once
its Go bindings stabilize on Windows. The same boundary-first approach applies to
the vector index (flat and HNSW behind one `Index` interface) and the job queue
(in-process today, distributed later).

## Roadmap

Shipped through **Phase 6** — core graph, rule engine, install layer, semantic
layer, LLM rules + diff analysis, live sync, broad multi-assistant installers,
meeting/transcript ingestion with decision-to-code linkage, Slack conversation
ingestion (Phase 4.5), the polish layer (HTML + Mermaid export, cross-repo
merging, Unix-socket transport, cross-corpus benchmark), the static
findings/graph dashboard (Phase 5.5), and browser-history + email ingestion
(Phase 6). Remaining: a first release, a first-party VS Code extension, and an
optional daemon-mode LLM path. See
[docs/PLAN.md](docs/PLAN.md) for the full plan and design rationale.

## License

Proprietary — Copyright © 2026 Eric Fernald. All rights reserved. See
[LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Third-party dependencies remain
under their own licenses.
