# Changelog

All notable changes to chiton are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**There are no released versions yet.** No git tag has been cut, so there is no
`1.0.0` and no release history — everything below sits under `Unreleased`.
Until a tag exists, `make` derives the version from
`git describe --tags --always --dirty` and `chiton version` falls back to Go
module and VCS metadata. The first tagged release will move these entries into a
dated section.

## [Unreleased]

### Added

- **Graph store layout version.** The Badger store now records its key layout under `m:schema`. Opening a store written by a newer chiton fails with an "upgrade chiton" error instead of misreading it, and an older layout is migrated on the first open (a read-only open does it through a one-time read-write pass). `.chiton/schema_version` is unchanged and still describes only the `.chiton/` scaffold.
- **`.chitonignore`.** A repo-root ignore file (names and globs, trailing `/` for directories, a leading or inner `/` anchors to the root, `!name` re-includes a built-in directory) now applies to `sync`, `sync watch`, and `bench`.

- **HNSW index for semantic search.** Large stores no longer scan every vector
  per query. `vector.index = auto` (the default) keeps the exact flat scan below
  100k embeddings and switches to an HNSW graph at or above it; `flat` and
  `hnsw` force one. At 100k 768-dimensional vectors a query drops from about
  30 ms to about 0.1 ms with recall@8 near 1.0 on embedding-like data.
  `chiton embed` builds the graph on every CPU and saves it beside the store as
  `hnsw.idx` (links only; the vectors stay in the store, and the file is
  encrypted when the store is). `serve` loads it in seconds instead of building
  at startup. When it is missing or out of date (embeddings changed since the
  last `chiton embed`), `serve` answers from the exact flat index at once,
  builds HNSW in the background, and switches over when it is ready, logging
  both steps. A one-shot `chiton search` stays exact under `auto`. Tune it with
  `vector.hnsw_threshold`, `vector.hnsw_m`, `vector.hnsw_ef_search`, and
  `vector.hnsw_ef_construction` (or `CHITON_VECTOR_*`). The graph is chiton's
  own pure-Go implementation (`internal/vector/hnswlib`); the default build
  stays cgo-free.
- **`chiton embed --link-similar --exact`.** `--link-similar` now uses the
  configured index, so large stores finish in minutes instead of an O(n²) scan;
  `--exact` restores the exact pass.

- **Claude Code plugin packaging.** The repository root doubles as a plugin:
  `.claude-plugin/plugin.json` (`chiton`, version `0.1.0`) and
  `.claude-plugin/marketplace.json` (`chiton-marketplace`, single plugin sourced
  from `./`). Both pass `claude plugin validate --strict`. Load it from a working
  tree with `claude --plugin-dir .`.
- **Three plugin skills**, namespaced `/chiton:graph`,
  `/chiton:subagent-tasks`, and `/chiton:setup`, living at
  `skills/{graph,subagent-tasks,setup}/SKILL.md`. `setup` is new: it walks a
  first-run workspace through locating or installing the binary, building the
  graph, and confirming the MCP server answers.
- **`skills/` is now the single source of truth for assistant-facing markdown.**
  A new root-level `embed.go` embeds each `SKILL.md` into the `chiton` package,
  and `internal/install/manifests` re-exports it, so the plugin and
  `chiton install` ship byte-identical text and cannot drift.
- **`chiton install` writes a second Claude Code skill.** Alongside
  `.claude/skills/chiton/SKILL.md` it now writes
  `.claude/skills/chiton-subagent-tasks/SKILL.md`; `chiton uninstall` and
  `chiton doctor` cover both. Skill frontmatter deliberately carries no `name:`
  — the directory name supplies it in both the plugin and the installed layout.
- **`bin/chiton` and `bin/chiton.cmd` PATH shims.** The plugin puts `bin/` on the
  assistant's PATH, and the shim resolves the real binary in order: `$CHITON_BIN`,
  then `chiton` / `chiton.exe` on PATH (skipping itself), then
  `$CLAUDE_PLUGIN_DATA/bin`, then `$HOME/.chiton/bin` — with install instructions
  on failure instead of a bare "not found".
- **`scripts/install-chiton.sh` and `scripts/install-chiton.ps1`.** They download
  the release asset for the host OS/arch, verify its SHA-256 against the
  release's `checksums.txt`, and stop on any mismatch. Nothing downloaded is
  executed by the scripts. Install location defaults to
  `$CLAUDE_PLUGIN_DATA/bin` under the plugin, otherwise `~/.chiton/bin`.
  Because the repository is private, assets are fetched through the REST API —
  either via an authenticated `gh`, or by per-asset API URL with a
  `GITHUB_TOKEN`. The `releases/download/` browser URL is deliberately not used:
  it is session-authenticated and returns 404 for a token.
- **`CHITON_PROJECT_DIR`.** When set, `internal/config` looks up `.chitonrc` in
  that directory and resolves a relative `graph_path` against it instead of
  against the process working directory. Unset, behaviour is unchanged. The
  plugin's MCP config sets it to `${CLAUDE_PROJECT_DIR}`.
- **New Makefile targets:** `build-msgraph`, `fmt`, `fmt-check`, `check-msgraph`,
  and `checksums`. `make release` now also writes `dist/checksums.txt`, which the
  install scripts require.
- **`.gitattributes`.** The tree is normalised LF-only, with CRLF kept for
  `*.cmd` / `*.bat` / `*.ps1` and explicit LF pins for `bin/chiton` and
  `scripts/*.sh` so the shim and installer survive being committed from Windows.
- **`SECURITY.md`, `CONTRIBUTING.md`, and this changelog** at the repository
  root. `SECURITY.md` is the private-reporting entry point GitHub surfaces; the
  threat model stays in `docs/SECURITY.md`.
- Tests for the new surfaces: `internal/config/config_test.go`,
  `cmd/chiton/netwarn_test.go`, a Teams webhook fail-closed test, and
  `internal/ingest/email/outlook_msgraph_test.go`.

### Changed

- **Fewer duplicate writes.** The Go extractor emits each node and edge once per file, and a `build`/`sync once` run writes each shared `pkg://` stub once; `build`'s node and edge counts drop accordingly.

- **`shortest_path` walks adjacency keys** and decodes only the edges on the returned path.
- **Incremental sync no longer rescans the whole graph per file.** The Badger store keeps source_uri and kind
  indexes (store layout v2), so `DeleteBySourceURI` and kind or source filters read only matching nodes. The
  first open after upgrading builds the indexes once, and an open after an unclean shutdown rebuilds them.
  `store.CountByKind` counts nodes and edges from keys alone.
- **The pre-commit hook now gates the commit, not the working tree.** It analyzes the staged blobs of the
  committed files in memory, with file rules only, since graph rules need the whole corpus; pre-push still
  scans the tree. Neither needs the graph store any more, and findings are recorded only when the store
  opens without prompting.

- **One ignore list for every walk.** `sync`, `sync watch`, `bench`, `analyze`, `analyze pending`, and the git hooks skip `.git`, `.hg`, `.svn`, `.chiton`, `chiton-out`, `node_modules`, `vendor`, `.venv`, `venv`, `__pycache__`, `target`, and `dist` at any depth, plus the entries in `.chitonignore`; `bench` previously skipped nothing. `analyze` also no longer follows symlinks.

- **Live sync with `--analyze` is linear again.** Re-analyzing a file used to load and decode every
  deterministic finding in the store to prune that file's; it now reads the file's findings through their
  AFFECTS edges before replacing its nodes, and an Analyzer writes each rule node once. `sync once` over
  Go's `src/net` (390 files) dropped from 10.1 s to 3.4 s.
- `chiton serve` with no transport flag now speaks stdio instead of starting the
  HTTP/SSE daemon on 127.0.0.1:7474. Pass `--http 127.0.0.1:7474` for the
  daemon; `--stdio` is still accepted.
- **Microsoft Graph is now behind the `msgraph` build tag, and the default build
  excludes it.** The Graph SDK links request builders for every top-level Graph
  resource and kiota registers its serialization types through interface tables,
  so the linker cannot strip the parts chiton never calls — about 117 MB of
  binary for two API calls. Excluding it took the stripped binary from
  **142.0 MB to 24.6 MB (-82.6%)**.
  - Affected commands: `chiton teams fetch`, `chiton teams serve-webhook`, and
    `chiton email outlook`. They still appear in `--help` and still parse their
    flags, but fail at runtime with a message telling you to rebuild with
    `-tags msgraph` (`make build-msgraph`).
  - `chiton email ingest` and `chiton email gmail` are pure Go and unaffected.
  - **Release artifacts are deliberately built without the tag.** If you rely on
    Teams or Outlook ingest, build your own binary with `make build-msgraph`.
  - `make check-msgraph` keeps the tagged path compiling and passing, since the
    default `make check` never compiles those files.
- **`make check` is now `fmt-check + vet + test`** (previously `vet + test`), so
  a non-gofmt-clean tree fails the pre-merge gate.
- **The root `.mcp.json` is now the plugin's MCP config, not a developer's.** It
  launches `chiton serve --stdio` by bare name — so the binary must be on PATH —
  and passes `CHITON_PROJECT_DIR=${CLAUDE_PROJECT_DIR}`. It previously hard-coded
  an absolute Windows path to `chiton.exe` and a `CHITON_GRAPH_PATH`. Anyone
  developing chiton itself should register their own build with
  `claude mcp add` at local scope rather than editing this file.
- **The always-on rules block written into `CLAUDE.md` / `AGENTS.md` and friends
  was rewritten.** It now names `semantic_search`, `get_node`, `diff_findings`,
  `search_meetings`, and `decisions_about`, explains `EXTRACTED` vs `INFERRED` vs
  `AMBIGUOUS` provenance, states that graph content (docs, transcripts, emails,
  page titles) is untrusted data rather than instructions, and mentions that the
  `pending_*` / `submit_*` flows need a writable server.
- `internal/install/manifests/SKILL.md` was removed; the package now re-exports
  the embedded content from `skills/` instead of holding its own copy.
- The wildcard-bind warning used by `chiton serve --http` moved into a shared
  `warnIfNonLoopback` helper (`cmd/chiton/netwarn.go`) so every listener warns
  the same way. It recognises an empty host and the unspecified IPv4/IPv6
  addresses, and leaves named hosts alone rather than resolving them at startup.
- Help text for `chiton teams` and `chiton email outlook` now states the
  `-tags msgraph` build requirement up front.
- The Outlook test whose subject takes a Microsoft Graph type moved behind the
  `msgraph` tag, so the default test binary no longer links the Graph SDK.

### Fixed

- **`.chiton/` state now follows `CHITON_PROJECT_DIR`.** An assistant-launched server started outside the workspace read `diff_findings` baselines and wrote LLM-rule state relative to its own working directory.
- **A malformed `.chitonrc` is now an error.** It used to be ignored, so a typo silently dropped `graph_path` back to the default and the command opened a different store.
- **`sync watch` recovers from event overflow.** When fsnotify reports dropped events, the watcher rescans the tree instead of silently missing changes.
- **`chiton build` runs the sync engine.** It removes symbols renamed or deleted since the last build, prunes files gone from disk (`--prune`, default true), skips ignored directories and `.chitonignore` paths, extracts on `GOMAXPROCS` workers, and batches store writes. The summary line is unchanged; per-file skip messages on stderr now start with `sync: skip`.
- **Windows path case no longer duplicates files.** `file://` URIs use an upper-case drive letter and on-disk case, so `chiton build c:\repo` and `chiton build .` index one copy. The next `build`/`sync once` drops entries an older build stored under a different case (those files are re-indexed once).
- **`sync once` survives unreadable directories and honours Ctrl-C.** A permission error below the root is logged and counted (`N unreadable` in the summary) instead of aborting the run, and `sync git`/`sync once` stop between files when cancelled.
- **Re-syncing a file no longer deletes meeting and concept links.** File sync and meeting/Slack/Teams/URL/email re-ingest replace a source by diff. Nodes the new extraction drops are deleted with their edges, but edges other sources attached to surviving nodes (`DECIDED_IN`, `DISCUSSED_IN`, `MENTIONED`, concept `DESCRIBES`, finding `AFFECTS`) are kept. An unchanged node keeps its `extraction_status` and embedding; a changed one loses its embedding so `chiton embed` redoes it. Decision and rationale nodes now use `decision://<meeting-uri>`, so re-ingesting a Slack channel no longer deletes them; older ones move on the next re-ingest.
- **The graph store no longer wedges with "Log truncate required".** On Windows, a writer opened while `chiton serve` had the store mapped could fail halfway and strand a 128 MiB memtable log that broke every later read-only open. chiton processes now coordinate on `chiton.lock` in the store directory and fail fast with `graph store is in use by another chiton process` instead of colliding. On Windows a writer is refused while any reader is open; on Linux/macOS writers may still open beside a read-only server. A read-only open now replays a crashed writer's log itself when nothing else holds the store.
- **Leiden no longer collapses the graph into a few giant communities.** Aggregation dropped each
  community's internal weight from the super-node degree and the total, so every merge looked profitable,
  and each level restarted from singletons instead of the unrefined partition. On this repo's graph the
  largest community fell from 87.5% of nodes to 15%, and Zachary's karate club now scores Q ≈ 0.42 (was 0).
  Re-run `chiton cluster` to refresh `cluster.json`.

- **Git hooks no longer block commits on infrastructure errors.** A locked store (a running `sync watch`),
  an encrypted store with no passphrase, a git error, or a malformed `hooks.yaml` used to fail
  `git commit`, and a failing `sync git --stage …` also failed `git checkout`. These now warn and exit 0.
  `timeout-seconds` is enforced (the analyzer checks its context per file and per graph rule), and an
  unreadable directory is skipped and counted instead of aborting `chiton analyze` or `analyze pending`.

- **An assistant-launched MCP server no longer risks reading a different store
  than `chiton build` wrote.** MCP servers are not reliably started in the
  workspace root, and the default `graph_path` is relative, so a server started
  elsewhere would resolve it against its own working directory and answer from an
  empty graph. `CHITON_PROJECT_DIR` (set by the plugin) now anchors both the
  `.chitonrc` lookup and relative `graph_path` resolution.

- **Live sync now prunes removed directories on Linux.** `fswatch` decided
  whether a removed path had been a directory by consulting fsnotify's
  `WatchList()`. On Linux fsnotify drops a deleted directory's watch from that
  list *while draining the kernel's event buffer* — `IN_DELETE_SELF` and
  `IN_IGNORED` are processed before the parent's `Remove` event reaches the
  Events channel — so by the time the watcher handled the event, the path was
  already gone from the list. The removal was misread as a file event,
  `OnDirRemove` never fired, and `chiton sync watch` left every node under the
  deleted directory in the graph. The watcher now tracks the directories it
  watches itself, which is unaffected by fsnotify's internal timing. Windows
  was never affected, which is why this only surfaced in CI.

- **`CHITON_EMBED_DIM` is no longer silently ignored.** `embed.dim` is the one
  setting with no default, so it never entered viper's key set and `Unmarshal`
  skipped it — the variable parsed fine and then did nothing. It is now bound
  explicitly, and a value in `.chitonrc` still works as before.

### Security

- **A repository's `.chitonrc` can no longer send your text off-device.** A non-loopback `embed.endpoint` set by the project `.chitonrc` is refused. Set `CHITON_EMBED_ENDPOINT` yourself to use a remote embedder.
- The graph tools (`get_node`, `query_graph`, `get_node_neighborhood`, `shortest_path`, `semantic_search`, `god_nodes`, `graph_summary`) now redact people's names, email addresses, speakers and assignees by default; `chiton serve --include-pii` restores them.
- `submit_extraction` rejects a `task_id` that isn't a doc_section or image and no longer overwrites concepts other tasks created, so a forged submit can't mark meetings as linked or rewrite concept descriptions.
- `diff_findings` baseline/target refs are validated and passed to `git rev-parse --verify --end-of-options`, so an option-like ref such as `--show-toplevel` can no longer leak the repository path.
- `add_url`'s SSRF guard now checks the IP it actually dials (closing the DNS-rebinding race) and blocks CGNAT, benchmarking, reserved, multicast, NAT64/6to4/Teredo and IPv4-mapped ranges; `add_url` no longer goes through `HTTP(S)_PROXY`.
- **Indexing no longer follows symlinks.** `build`, `sync`, and `sync watch` index regular files only; a symlink or junction (e.g. `notes.md -> ~/.ssh/id_rsa`) is skipped, so its target never lands in the graph or in `pending_extractions`.
- MCP `add_url` accepts only http/https URLs (the CLI still reads local files), and `pending_analyses` refuses paths outside the project root and no longer reads through symlinks, so an injected instruction can't make an MCP tool read `~/.aws/credentials`.
- `chiton serve --http` now requires a bearer token (per start, written to `<graph dir>/mcp-token`, or `CHITON_MCP_TOKEN`), refuses browser `Origin`s not listed with `--allow-origin`, requires `application/json` POSTs, and no longer sends `Access-Control-Allow-Origin: *`. Before this, any web page could call tools on the loopback server with a `text/plain` POST.
- **`chiton teams serve-webhook` now fails closed.** Two changes, both
  behaviour-visible:
  - The command **refuses to start** without a subscription `clientState`
    (`CHITON_TEAMS_CLIENT_STATE` or `--client-state`), rather than starting a
    receiver that cannot authenticate anything.
  - The handler itself now **rejects every notification** when no `clientState`
    is configured. It previously accepted them all — an unauthenticated receiver
    that trusted any caller, so any host that could reach the port could inject
    forged meeting content into the graph.

  If you were running the webhook receiver without `CHITON_TEAMS_CLIENT_STATE`
  set, it will now exit with an error until you configure the shared secret you
  registered with the Graph subscription.
- **Webhook and events listeners now default to loopback.** `chiton teams
  serve-webhook` defaults to `127.0.0.1:8444` and `chiton slack serve-events` to
  `127.0.0.1:8445`; both previously defaulted to `:8444` / `:8445`, which binds
  **every** network interface. Both commands now also print a warning when the
  configured address is a wildcard. Neither receiver is meant to face the
  internet directly — put a tunnel or reverse proxy in front. If you depended on
  the old wide bind, pass `--addr 0.0.0.0:<port>` explicitly and heed the
  warning.
