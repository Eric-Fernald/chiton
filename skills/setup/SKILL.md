---
description: Set up chiton in this workspace — install the chiton binary if it is missing, build the knowledge graph, and verify the MCP server is answering. Use when chiton's MCP tools are unavailable or erroring, when the graph is empty or stale, when the user asks how to get chiton running, or right after installing the chiton plugin.
---

# Setting chiton up in this workspace

Chiton is a single Go binary. A plugin installed from the marketplace bundles
it for linux and darwin (amd64, arm64) and windows/amd64, alongside the skills
and the MCP server registration. Work through these steps in order and stop as
soon as the graph answers.

## Step 1 — is the binary there?

```sh
chiton version
```

If that prints a version, skip to Step 2. With the plugin installed from the
marketplace on a supported platform it should: the plugin's `bin/chiton`
launcher is on the Bash tool's `PATH` and runs the bundled binary.

If it is not found, the plugin is running from a source checkout
(`claude --plugin-dir`) or on a platform with no bundled binary. The launcher
resolves the binary in this order and uses the first hit:

1. `$CHITON_BIN` — an explicit override, if the user has set one.
2. The binary bundled with the plugin.
3. `chiton` on `PATH`.
4. `$CLAUDE_PLUGIN_DATA/bin/chiton` (`chiton.exe` on Windows) — where this
   plugin's installer puts it.
5. `$HOME/.chiton/bin/chiton` (`%USERPROFILE%\.chiton\bin\chiton.exe`).

If none exist, install it. Ask the user which route they want rather than
picking for them — each has a different tradeoff:

**a. Prebuilt release** (no Go toolchain needed):

```sh
sh "$CLAUDE_PLUGIN_ROOT/scripts/install-chiton.sh"
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:CLAUDE_PLUGIN_ROOT\scripts\install-chiton.ps1"
```

The `-ExecutionPolicy Bypass` is not optional — Windows client SKUs default to
`Restricted` and refuse to run any `.ps1` without it.

The script detects OS/arch, downloads the matching release asset, verifies its
SHA-256 against the published `checksums.txt`, and installs into
`$CLAUDE_PLUGIN_DATA/bin` (or `~/.chiton/bin`). It never executes anything it
downloads.

It needs no credentials. The GitHub API allows 60 unauthenticated requests an
hour; an authenticated `gh` or a `GITHUB_TOKEN` lifts that limit. It fails if no
release has been published yet.

**b. From a local clone** (needs a Go toolchain):

```sh
make build
```

**c. `go install`** (needs a Go toolchain):

```sh
go install github.com/Eric-Fernald/chiton/cmd/chiton@latest
```

That lands in `$(go env GOPATH)/bin`, which is usually already on `PATH`. Any
Go new enough to honour the `toolchain` directive works — `go.mod` declares
`go 1.25.5` and pins `toolchain go1.26.4`, which is fetched automatically.

After installing, confirm `chiton` is on `PATH`. If it is not, tell the user
the exact line to add to their shell profile — do not edit their profile for
them. A source checkout's `.mcp.json` launches `chiton` by name, so there it
must be resolvable from `PATH` for the graph tools to appear.

## Step 2 — build the graph

From the workspace root:

```sh
chiton init          # once: scaffolds .chiton/ (policy files + encryption keyfile)
chiton build .       # walk the tree, extract the graph into chiton-out/
chiton cluster       # Leiden communities + betweenness, written to cluster.json
```

**Encryption, and what it means for the plugin.** `chiton init` encrypts the
store by default, deriving the key from a passphrase. An assistant-launched MCP
server has no terminal, so it cannot prompt for one — it exits immediately with
`no passphrase available: set CHITON_PASSPHRASE or run interactively`. The
error's second line names the fix. Two ways forward, and the user picks:

- Export `CHITON_PASSPHRASE` in the shell profile that starts Claude Code, so
  the server inherits it. The passphrase is never written to the graph, the
  logs, or any config file.
- Or, before the first `chiton init`, scaffold without encryption:
  `chiton init --encrypt=false`. Once a keyfile exists that flag changes
  nothing; turning encryption off then means removing `.chiton/keyfile.json`
  and `chiton-out/` and rebuilding, which discards ingested meetings, email
  and Slack. Reasonable for a graph over already-public code; not for one
  holding email, Slack, or meeting transcripts.

OS keychain support is not implemented; the environment variable is the only
non-interactive path.

Do not put the passphrase in the plugin's `.mcp.json` or in any file inside the
repository.

`chiton build` is incremental — a SHA-256 content cache skips unchanged files,
and a rebuild removes symbols and files deleted since the last one, so re-running
it is cheap.

Optional, in rough order of value:

```sh
chiton analyze .     # deterministic rule findings (free, inline, no LLM)
chiton embed         # embeddings for semantic_search (Ollama by default;
                     #   CHITON_EMBED_PROVIDER=hash works offline)
```

## Step 3 — restart and verify

Embeddings and cluster results load at `chiton serve` startup, so the MCP
server has to be restarted to see a freshly built graph. Run `/reload-plugins`,
or restart the session.

Then verify from the assistant side, not the shell:

- Call `graph_summary` — it should report a non-zero node and edge count.
- Call `query_graph` with a symbol you know exists in this repo.

If `graph_summary` reports an empty graph, the server is pointed at a different
store than `chiton build` wrote. Check for a `.chitonrc` with a custom
`graph_path`, and confirm `chiton-out/` sits at the workspace root.

## Step 4 — keep it current

Pick one:

```sh
chiton sync watch .                  # fsnotify watcher; re-indexes on save
chiton hooks install --stage post-commit,post-checkout   # sync on git operations
chiton sync once .                   # manual one-shot reconcile
```

## Optional extras

- **Semantic concepts and LLM rules** run through subagents, not an API key.
  See the `subagent-tasks` skill.
- **Meetings, email, browser history, Slack, Teams** are opt-in ingestion
  commands — nothing is read until the user runs them explicitly. See the
  privacy section of the plugin README before enabling any of them.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| MCP tools missing entirely | no bundled binary for this platform (or a source checkout) and `chiton` not on `PATH` when the server launched | install, then `/reload-plugins` |
| `no passphrase available` | encrypted store, no terminal to prompt on | export `CHITON_PASSPHRASE` in the shell profile that starts the assistant, then restart it |
| graph store is in use by another chiton process | a writer (sync watch, build, serve --writable) holds the store, or on Windows a running chiton serve blocks a writer | stop the other process (or the assistant's server) and retry |
| needs a one-time read-write pass | a chiton writer crashed, or the store predates this chiton | close other chiton processes and retry; the next open repairs it |
| Tools work but return nothing | graph never built, or built elsewhere | `chiton build .` at the workspace root, then restart |
| `submit_*` tools fail to write | server started read-only | restart with `chiton serve --writable`, or use the CLI submit path |
| `semantic_search` errors | no embeddings | `chiton embed`, then restart the server |
| serve warns "made with model …" | embedding model changed | `chiton embed`, then restart the server |
