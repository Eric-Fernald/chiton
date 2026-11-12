<p align="center">
  <img src="assets/logo.svg" alt="chiton logo: eight overlapping diamond rings shading from blue to pink" width="480">
</p>

# Chiton

Like the mollusk it's named after, chiton is many independent plates flexing as
one organism. AST extraction, semantic embeddings, Leiden clustering, a
deterministic + LLM rule engine, and an MCP server, built into a single Go
binary that gives AI assistants a graph-grounded memory of your codebase.

chiton is an **on-device knowledge-graph engine**. It walks through your code,
docs, and images, extracts a typed graph (symbols, files, calls, imports, doc
sections, concepts, findings…), stores it locally, clusters it, and serves
relationship-aware neighborhoods to assistants over the Model Context Protocol.

---

## Why a graph instead of grep?

Chiton answers structural and navigational questions with typed edges
(`CALLS`, `IMPORTS`, `CONTAINS`, `DESCRIBES`, `VIOLATES`, …) rather than raw file
dumps. Ask "what does `X` depend on?" or "how are `A` and `B` connected?" and the
assistant gets a small, structured subgraph instead of whole files to
cross-reference. On raw token count it is much cheaper than dumping whole files
and roughly on par with `grep`+`read` (see [docs/BENCHMARK.md](docs/BENCHMARK.md)
for measured numbers). The gain is *better* context, not fewer tokens.

Every node and edge carries **provenance** and **confidence**:

- `EXTRACTED` (confidence 1.0) marks structural facts from the parser. These are
  ground truth.
- `INFERRED` marks LLM-extracted concepts and findings. Treat them as hints.
- `AMBIGUOUS` marks results below the confidence threshold, hidden by default.

When sources conflict, prefer `EXTRACTED` facts over `INFERRED` ones.

## How it works

```
  ingest            extract              store + index            serve
  ------            -------              -------------            -----
  file walk    →    Go AST (go/ast)  →   BadgerDB graph      →   MCP (stdio/HTTP)
  fsnotify          line scanners        (encrypted at rest)      22 graph + semantic
  git hooks         Markdown structure   pure-Go vector index     + analysis tools
                    image nodes          Leiden communities       CLI (thin wrapper)
                    rule engine          betweenness / god nodes
```

- **Single binary, no runtime services.** The graph store (BadgerDB, AES-256
  encrypted at rest), the vector index, the job queue, and Leiden clustering all
  run in-process. There is no Redis, CGo, or external database to set up.
- **chiton never calls an LLM itself.** Semantic extraction and LLM rules run
  through your *host assistant's* subagents: chiton emits task specs, the
  assistant runs them, and chiton writes the structured results back. It needs
  no API key and costs nothing beyond your existing subscription.
- **Incremental by default.** A SHA-256 content cache skips unchanged files;
  live sync re-indexes only what changed and re-clusters just the touched
  neighborhood.

## Install

### As a Claude Code plugin

```sh
claude plugin marketplace add Eric-Fernald/chiton
claude plugin install chiton@chiton-marketplace
```

Then run `/chiton:setup` in a project to build its graph.

The plugin ships without the chiton binary. The first time its MCP server
starts, it downloads the binary for your OS and architecture (about 25 MB) from
the matching GitHub release, checks it against SHA-256 checksums that come with
the plugin, and caches it in `~/.chiton/versions/<version>/`. Later starts use
the cache. On Linux and macOS this needs `curl` or `wget` plus `sha256sum` or
`shasum`; Windows uses the built-in PowerShell. If that first start times out
while the download runs, reconnect with `/mcp` once it finishes.

The plugin provides:

- **Three skills:** `/chiton:graph` (when to reach for the graph instead of
  grep), `/chiton:subagent-tasks` (semantic extraction, LLM rules, meeting→code
  linkage), and `/chiton:setup` (first-run install, build, verify).
- **The MCP server**, read-only by default. The five write tools
  (`submit_extraction`, `submit_analysis`, `submit_meeting_links`,
  `submit_verification`, `add_url`) report themselves unavailable until you run `chiton serve --writable` or use
  the CLI write path.
- **A `chiton` command** on the assistant's shell `PATH`, so it can run CLI
  commands even when the binary is not on yours.

The plugin works in Claude Code only, not in claude.ai or Cowork, because chiton
needs a local filesystem.

### Standalone binary

Prebuilt binaries cover Linux and macOS (amd64, arm64) and Windows (amd64;
Windows on Arm runs it under emulation). Download one from the
[releases page](https://github.com/Eric-Fernald/chiton/releases), or let the
install script pick the right one and verify its checksum:

```sh
curl -fsSL https://raw.githubusercontent.com/Eric-Fernald/chiton/main/scripts/install-chiton.sh | sh
```

```powershell
irm https://raw.githubusercontent.com/Eric-Fernald/chiton/main/scripts/install-chiton.ps1 -OutFile install-chiton.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File install-chiton.ps1
```

Both install into `~/.chiton/bin` and print the line to add to your `PATH`.
With Go 1.25.5 or newer you can also build it yourself:

```sh
go install github.com/Eric-Fernald/chiton/cmd/chiton@latest
```

`chiton --help` lists every subcommand.

### Other assistants

`chiton install` registers chiton with nine assistants, Claude Code included.
See [Assistant support](#assistant-support).

| | plugin | `chiton install` |
|---|---|---|
| Assistants | Claude Code only | nine, including Claude Code |
| Setup | two commands, updated through the marketplace | run per repo from an installed binary |
| Writes into your repo | nothing | `.mcp.json` (or the assistant's equivalent), a rules block, and, for Claude Code, skills |
| Skills | `/chiton:graph`, `/chiton:subagent-tasks`, `/chiton:setup` | Claude Code only: `chiton` and `chiton-subagent-tasks` |

You can use both in one repo, but Claude Code then gets two copies of the same
guidance.

### Requirements

- **Git 2.30 or newer** for `sync git`, the git hooks, and `analyze --diff` /
  `diff_findings`.
- **Optional: [Ollama](https://ollama.com)** for local embeddings (default model
  `nomic-embed-text` at `http://localhost:11434`). Set
  `CHITON_EMBED_PROVIDER=hash` for an offline embedder that needs no model
  server.
- **Optional: an MCP-capable assistant** to query the graph and run extraction
  subagents.

### Microsoft Teams and Outlook

`chiton teams fetch`, `chiton teams serve-webhook`, and `chiton email outlook`
use Microsoft Graph, whose SDK adds about 117 MB to the binary, so release
binaries leave it out. Those commands print a message asking you to rebuild. To
use them, build from a clone with the `msgraph` tag:

```sh
git clone https://github.com/Eric-Fernald/chiton.git
cd chiton
make build-msgraph      # or: go build -tags msgraph ./cmd/chiton
```

`chiton email ingest` and `chiton email gmail` work in every build.

## Quickstart

```sh
chiton init                 # scaffold .chiton/ (policy files + encryption keyfile)
chiton build .              # walk the tree, extract the graph → chiton-out/
chiton cluster              # Leiden communities + betweenness → cluster.json
chiton report               # write GRAPH_REPORT.md (god nodes + community summaries)
chiton query Server         # look up a node and its neighborhood from the CLI
chiton install              # register chiton's MCP server + rules with detected assistants
```

Then ask your assistant architecture questions. It will call chiton's tools
instead of grepping. To enable natural-language search:

```sh
chiton embed                # embed symbols/docs/concepts (Ollama by default)
chiton search "where do we handle auth?"
```

`chiton init` encrypts the graph with a passphrase. An assistant-launched server
cannot prompt for it, so export `CHITON_PASSPHRASE` in the environment that
starts your assistant (see [Encryption at rest](#encryption-at-rest)).

## CLI reference

| Command | Purpose |
|---|---|
| `chiton init` | Scaffold `.chiton/` (policy files, `.gitignore`, encryption keyfile). `--encrypt=false` for plaintext. |
| `chiton build <path>` | Walk a tree and extract code, docs and images into the graph. Unchanged files are skipped, and symbols and files deleted since the last build are removed with their findings (`--prune`, default true). See [Ignoring paths](#ignoring-paths). |
| `chiton query <label>` | Look up a node by label substring and print its neighborhood (`--depth`, `--edge-kinds`, `--json`). |
| `chiton cluster` | Run Leiden community detection + betweenness; writes the `cluster.json` sidecar. |
| `chiton report` | Generate `GRAPH_REPORT.md` (god nodes, community summaries). Needs a prior `cluster`. |
| `chiton visualize` | Export an interactive HTML graph (`--format html`) or a Mermaid diagram (`--format mermaid`). `--scope`, `--max-nodes`. |
| `chiton dashboard build` | Generate a static dashboard (interactive graph + findings explorer) into a directory (`--out`, default `chiton-dashboard/`). `--scope`, `--max-nodes`, `--title`, `--include-ambiguous`. Open its `index.html` directly; it is not a server. |
| `chiton export` | Write the whole graph to a portable JSON snapshot (`--out`, default stdout). |
| `chiton merge <snap>…` | Merge exported snapshot(s) into the current graph, deduplicating shared nodes. Alias: `import`. |
| `chiton bench [path]` | Benchmark token cost (chiton vs naive vs grep+read); `--synth N` for a synthetic corpus, `--json`. |
| `chiton version` | Print version + build metadata. |
| `chiton embed` | Embed nodes for semantic search (`--provider`, `--all`, `--link-similar`, `--exact`). Covers the whole store, re-embedding only nodes whose text or model changed. |
| `chiton search <query>` | Hybrid semantic + graph search from the CLI. |
| `chiton extract pending\|submit` | Emit semantic-extraction task specs / ingest the structured results. |
| `chiton verify sweep` | Demote concepts no source backs any more to `AMBIGUOUS`, and re-link decisions whose governed symbol moved files. A name several symbols share is listed, not guessed. Deterministic, no LLM. |
| `chiton verify pending\|submit` | Emit tasks that re-check inferred concepts and decisions against their current sources / apply the verdicts. Only claims whose evidence changed are emitted. |
| `chiton analyze <path>` | Run deterministic rules; persist findings as graph nodes. `--diff` / `--since <ref>`, `--no-llm`. |
| `chiton analyze pending\|submit` | Emit LLM-rule task specs / ingest LLM findings. |
| `chiton findings list\|explain` | Inspect persisted findings (filter by severity, category, scope). |
| `chiton sync watch\|git\|once` | Live sync: fsnotify watcher, git-driven re-index, or one-shot reconcile. |
| `chiton hooks install\|uninstall\|status\|run` | Manage git hooks (gating pre-commit/pre-push; syncing post-commit/post-checkout). |
| `chiton install\|uninstall\|doctor` | Register/remove chiton with AI assistants; report install state. `--assistant <name>` (claude-code, copilot-vscode, cursor, codex, gemini-cli, opencode, aider, antigravity, copilot-cli), `all`, or `auto` (default: detected). |
| `chiton meeting ingest\|transcribe\|pending\|submit` | Ingest meeting transcripts (`.vtt`/`.srt`/`.txt`), whisper-transcribe audio first, or emit/ingest meeting→code linkage task specs. |
| `chiton add-url <url-or-path>` | Ingest a web page, PDF, or YouTube transcript as a `url` node with `doc_section` children (`--kind`, `--author`, `--contributor`). |
| `chiton teams fetch\|serve-webhook` | Fetch + ingest all transcripts for one Teams meeting (`--meeting <id>`), or run the Graph change-notification receiver. Needs `CHITON_TEAMS_{TENANT_ID,CLIENT_ID,CLIENT_SECRET,USER_ID}` and an `msgraph` build. The receiver refuses to start without `CHITON_TEAMS_CLIENT_STATE`; `--addr` defaults to `127.0.0.1:8444`. |
| `chiton slack fetch --channel <id>` | Fetch a Slack channel, or one thread with `--thread <ts>`, and ingest it as a meeting subgraph (`CHITON_SLACK_TOKEN`). |
| `chiton slack serve-events` | Run the Slack Events API receiver: a new message re-ingests its thread (or channel). Refuses to start without `CHITON_SLACK_SIGNING_SECRET`; `--addr` defaults to `127.0.0.1:8445`. |
| `chiton browser ingest\|watch` | Ingest local browser history (Chrome/Edge/Brave) as `url` nodes; `watch` re-ingests on change. `--db <History>`, `--browser`, `--since`, `--min-visits`, `--keep-query`. |
| `chiton email ingest\|gmail\|outlook` | Ingest email: local `.eml`/`.mbox` files, the Gmail API (`CHITON_GMAIL_TOKEN`, `--query`), or Microsoft Graph (`CHITON_OUTLOOK_*`, `--max`; needs an `msgraph` build). |
| `chiton serve` | Start the MCP server: stdio by default, `--http 127.0.0.1:7474` (SSE; a shared daemon), or `--unix <path>` (socket). `--writable` for submit tools. `--include-pii` returns people's names and addresses from the graph tools, which redact them by default. HTTP requires the bearer token from `<graph dir>/mcp-token`. |

> **Note:** the two event receivers, `teams serve-webhook` and `slack
> serve-events`, bind loopback by default and print a warning when `--addr`
> points at a wildcard address. Both providers need a public HTTPS URL, so put a
> tunnel or reverse proxy in front of localhost. Each also refuses to start
> without its shared secret.

## MCP tools

The server (`chiton serve`) exposes these tools. The CLI is a thin local
wrapper over the same logic.

| Tool | Use when |
|---|---|
| `get_node` | Fetch a single node by label (exact names rank first; `"<dir>:<Name>"` picks one package). |
| `get_node_neighborhood` | "What does X depend on / call / import?" Returns the node and its N-hop neighbors. `direction: "in"` with `edge_kinds: ["CALLS"]` lists callers, `"out"` lists callees; calls resolve across files and packages. |
| `package_deps` | "What does package P import, or what imports it?" Direct or `transitive`, joined over every file in the Go package. |
| `query_graph` | Find nodes whose label matches a known substring. |
| `shortest_path` | "How are A and B connected?" |
| `god_nodes` | Highest-betweenness nodes, which are the architectural hubs (needs `cluster`). |
| `graph_summary` | Token-budgeted overview: counts by kind, communities, god nodes. |
| `semantic_search` | Natural-language questions, ranked by meaning then graph-expanded (needs `embed`). |
| `list_findings` / `explain_finding` | Rule-engine findings as graph nodes. AMBIGUOUS LLM findings hidden unless `include_ambiguous`. |
| `diff_findings` | `{new, resolved, changed}` versus a saved baseline. |
| `pending_extractions` / `submit_extraction` | Subagent semantic extraction: task specs in, concepts + edges out. |
| `pending_analyses` / `submit_analysis` | Subagent LLM-rule analysis: task specs in, findings out. |
| `search_meetings` | Find ingested meetings by content, with their decisions and linked code symbols. Participant names (PII) hidden unless `include_pii`. |
| `decisions_about` | Decisions made about a topic or code symbol, with their rationale, the meeting they were decided in, and the code they govern. |
| `pending_meeting_links` / `submit_meeting_links` | Subagent meeting→code linkage: transcript + candidate-symbol task specs in, decisions + edges out. |
| `pending_verifications` / `submit_verification` | Subagent re-check of inferred claims: a concept against the text it came from, a decision against the code it governs. Verdicts must quote the evidence. Drift shows up in `decisions_about`. |
| `add_url` | Fetch a URL (article, paper PDF, or YouTube transcript) and ingest it as a `url` node with `doc_section` children. |

> **Note:** `query_graph` matches node *labels*, not call sites. A `CALLS` edge
> says that function A calls function B, not on which line, so use grep for
> "which lines call X?". chiton tracks calls to package-level functions but not
> method calls on values (`x.Method()`). `CALLS` edges connect symbol nodes, not
> file nodes.

Write tools (`submit_extraction`, `submit_analysis`, `submit_meeting_links`,
`submit_verification`, `add_url`) need a writable store: run `chiton serve
--writable`, or use the CLI write path (`chiton extract submit`, `chiton analyze
submit`, `chiton meeting submit`, `chiton verify submit`, `chiton add-url`). On Windows, writers such as `build` and `embed`
fail with `in use by another chiton process` while a server is running, so stop
it first or submit through `chiton serve --writable`.

The full reference is in [docs/MCP.md](docs/MCP.md).

## Assistant support

`chiton install` registers the MCP server and an always-on rules block. Run it
with no flags to auto-detect assistants in the workspace, `--assistant all` for
every one, `--assistant <name>` for a specific one, or `--user-scope` for
user-global config. `chiton doctor` reports the state; `chiton uninstall`
reverses it.

Only the Claude Code installer writes skills, two of them:
`.claude/skills/chiton/` (graph queries) and
`.claude/skills/chiton-subagent-tasks/` (extraction, LLM rules, meeting→code
linkage). The other eight assistants get the MCP registration and the rules
block.

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
categories: security, bugs, quality, architecture, performance, reliability,
database, and style. Findings become `finding` nodes (`EXTRACTED`,
`rule_provenance=DETERMINISTIC`) with `VIOLATES → rule` and `AFFECTS → code`
edges; re-analyzing a file prunes findings that no longer fire, so the store
reflects the current code.

LLM rules (`llm/…`) cover the semantic cases AST rules can't reach
(abstraction leaks, contract-vs-comment drift, missing authz checks, N+1
queries, race/resource-leak risks, comment rot, …). They dispatch through the
same subagent path as semantic extraction, land as `INFERRED`/`AMBIGUOUS`
findings, and never block the pre-commit gate.

Wire findings into git with `chiton hooks install`; the per-repo
`.chiton/hooks.yaml` sets the block-on severity threshold per stage. The
pre-commit gate checks the staged content of the files in the commit (file
rules only), and pre-push scans the whole tree. A locked or
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

For each changed file, sync re-extracts it, refreshes its deterministic findings
(`--analyze`, on by default), and, with `--embed`, its embeddings. It keeps the
edges that meetings and concepts attached to symbols that still exist, then
re-clusters the neighborhood of the changed files. Run `chiton cluster` now and
then for a full pass, and `chiton embed` after syncing with `--embed` to save a
current search index.

Stop the watcher with Ctrl-C so the store closes cleanly; a hard kill can leave
it needing recovery on the next open.

### Ignoring paths

`build`, `sync`, `sync watch`, `bench`, `analyze`, `analyze pending` and the
git hooks skip these directory names at any depth:
`.git`, `.hg`, `.svn`, `.chiton`, `chiton-out`, `node_modules`, `vendor`,
`.venv`, `venv`, `__pycache__`, `target`, `dist`. Symlinks are never followed.

A repo-root `.chitonignore` adds more, one entry per line:

- `name` or a glob such as `*.pb.go` matches at any depth; `#` starts a comment.
- A trailing `/` matches directories only (`build/`).
- A leading or inner `/` anchors the pattern to the repo root (`/gen/`, `docs/api`).
- `!name` re-includes a built-in directory (`!vendor`).
- `**` is not supported.

`sync watch` reads the file at start, so restart it after an edit.

## Visualizing & sharing the graph

```sh
chiton visualize --format html --out graph.html   # interactive, pan/zoom/search (vis-network)
chiton visualize --format mermaid --out graph.mmd  # Mermaid flowchart for Markdown docs
chiton dashboard build                             # static dashboard: graph + findings explorer → chiton-dashboard/
chiton export --out graph.json                     # portable JSON snapshot of the whole graph
chiton merge other-repo.json                       # merge another repo's snapshot into this graph
```

The HTML export is a self-contained page with nodes coloured by community, sized
by degree, and a search box. Mermaid output renders inline on GitHub, Obsidian,
or mkdocs; both honour `--scope` and `--max-nodes`. `dashboard build` adds a
**findings explorer** that filters findings by severity, category, community,
file, and provenance; clicking a finding focuses its node in the graph. Node IDs
are stable content hashes, so merging several repos' snapshots into one store
deduplicates shared nodes and builds a **cross-repo** graph.

## Configuration

chiton reads `.chitonrc` (TOML) from the project directory or `$HOME`, with
`CHITON_*` environment overrides (e.g. `CHITON_EMBED_PROVIDER=hash`). The
project directory is `$CHITON_PROJECT_DIR` when set (the Claude Code plugin sets
it), and the working directory otherwise. A relative `graph_path` and the
`.chiton/` state directory resolve against it, so an assistant-launched server
reads the same store `chiton build` wrote. A `.chitonrc` that fails to parse is
an error that names the file.

| Key | Default | Notes |
|---|---|---|
| `graph_path` | `./chiton-out/graph.kuzu` | Where the graph store lives (the name is historical). |
| `log_level` | `info` | |
| `embed.provider` | `ollama` | `ollama` or `hash` (offline/CI). |
| `embed.model` | `nomic-embed-text` | |
| `embed.endpoint` | `http://localhost:11434` | Ollama endpoint. A non-loopback value in the project's `.chitonrc` is refused; set `CHITON_EMBED_ENDPOINT` (or `$HOME/.chitonrc`, read only when the project has none) to use a remote server. |
| `embed.dim` | *(unset)* | Embedding dimension. Unset, each provider uses its native size (Ollama 768, hash 256). Set it only to override. |
| `semantic.confidence_threshold` | `0.5` | Splits `INFERRED` from `AMBIGUOUS`. |
| `vector.index` | `auto` | Index behind semantic search: `flat` (exact scan), `hnsw` (approximate graph), or `auto`, which uses `hnsw` at or above `vector.hnsw_threshold` vectors. `chiton embed` saves the `hnsw` index beside the store; if it is missing or stale, `serve` answers exactly and builds it in the background. |
| `vector.hnsw_threshold` | `100000` | Vector count at which `auto` switches to `hnsw`. 0 or less means the default. |
| `vector.hnsw_m` | `16` | Maximum links per HNSW node (twice that on the bottom layer). Higher raises recall, memory, and build time. |
| `vector.hnsw_ef_search` | `128` | HNSW candidate list per query. Higher raises recall and query time. |
| `vector.hnsw_ef_construction` | `100` | HNSW candidate list while building. Higher raises graph quality and build time. |

Commit `.chiton/hooks.yaml` and `.chiton/sources.yaml` to share hook policy with
your team. `chiton init` writes a `.chiton/.gitignore` that keeps everything
else under `.chiton/`, including the encryption keyfile, out of git, and chiton
writes `chiton-out/.gitignore` so the graph store is never committed.

## Encryption at rest

The graph is encrypted by default with AES-256 under a key derived from a
passphrase with argon2id. `.chiton/keyfile.json` holds the salt and KDF
parameters; the passphrase and key are never written to disk. chiton reads the
passphrase from `CHITON_PASSPHRASE` or prompts for it. An assistant-launched
`chiton serve` cannot prompt, so export `CHITON_PASSPHRASE` in the environment
that starts the assistant. The saved search index and cluster sidecar are
encrypted too; the other files in `chiton-out/` and `.chiton/` are plaintext.
Opt out for CI or synthetic data with `chiton init --encrypt=false`.

## Supported sources

- **Go.** Full AST extraction via the standard-library parser (`go/ast`):
  files, symbols, `CONTAINS`, `IMPORTS`, and `CALLS` edges for calls to
  package-level functions, across files and packages. File nodes carry the
  package's `import_path` and `module`.
- **Python, JavaScript/TypeScript, Rust, Java.** Lightweight line-scanning
  extractors for symbols and imports.
- **Markdown.** `doc_section` nodes per heading with `CONTAINS` nesting.
- **Images.** Image nodes at build time, described via vision subagents.
- **Meetings.** Teams/Zoom/whisper transcripts (WebVTT/SRT) become `meeting`,
  `meeting_turn`, and `person` nodes; subagent linkage ties the decisions made
  in them back to the code symbols they reference (`chiton meeting`).
- **Slack.** Channels and threads via `chiton slack fetch`. Each conversation
  becomes the same meeting subgraph a transcript does, so `search_meetings` and
  `decisions_about` cover Slack discussions too. Re-ingesting replaces rather
  than duplicates.
- **URLs.** Articles, PDFs, and YouTube transcripts via `chiton add-url`.
- **Email.** Local `.eml`/`.mbox`, Gmail, or Outlook (`chiton email`). Messages
  become `email` nodes with body sections and `person` participants, so
  semantic extraction picks up their concepts too.
- **Browser history.** Chrome/Edge/Brave (`chiton browser`). chiton reads only
  page *metadata* from the local History database (title, visit count, last
  visit) and never fetches the pages. URLs are stored without userinfo, query
  or fragment unless you pass `--keep-query`.

Everything beyond your code is opt-in: nothing is read until you run the command
for it. Names of people (speakers, participants, authors) are tagged as personal
data and hidden from the MCP tools by default. See
[docs/PRIVACY.md](docs/PRIVACY.md) and [docs/SECURITY.md](docs/SECURITY.md).

## Roadmap

Planned: a first-party VS Code extension and an optional daemon-mode LLM path
for running extraction without a host assistant.

## Contributing and support

Report bugs and request features in
[GitHub issues](https://github.com/Eric-Fernald/chiton/issues). Report security
problems privately as described in [SECURITY.md](SECURITY.md). Pull requests are
not accepted; see [CONTRIBUTING.md](CONTRIBUTING.md), which also covers building
from source, the repository layout, and adding rules.

## License

Copyright © 2026 Eric Fernald. Licensed under the
[PolyForm Strict License 1.0.0](https://polyformproject.org/licenses/strict/1.0.0)
with additional permissions: you may use chiton for any purpose, including
commercial work, and install or build it by any route this documentation
describes, but may not distribute it or make changes or new works based on it.
Plugin marketplaces and mirrors may host unmodified copies. See
[LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
Third-party dependencies remain under their own licenses.
