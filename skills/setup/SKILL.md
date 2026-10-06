---
description: Set up chiton in this workspace — install the chiton binary if it is missing, build the knowledge graph, and verify the MCP server is answering. Use when chiton's MCP tools are unavailable or erroring, when the graph is empty or stale, when the user asks how to get chiton running, or right after installing the chiton plugin.
---

# Setting chiton up in this workspace

Chiton is a single Go binary. The plugin ships the skills and the MCP server
registration; the binary itself is installed once per machine. Work through
these steps in order and stop as soon as the graph answers.

## Step 1 — is the binary there?

```sh
chiton version
```

If that prints a version, skip to Step 2.

If it is not found, resolve it in this order and use the first hit:

1. `$CHITON_BIN` — an explicit override, if the user has set one.
2. `chiton` on `PATH`.
3. `$CLAUDE_PLUGIN_DATA/bin/chiton` (`chiton.exe` on Windows) — where this
   plugin's installer puts it.
4. `$HOME/.chiton/bin/chiton` (`%USERPROFILE%\.chiton\bin\chiton.exe`).

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

**The repository is private**, so the script needs credentials: either
`gh auth login` (simplest — it then downloads through the GitHub CLI) or
`GITHUB_TOKEN` set to a token with `repo:read`. Release assets are fetched
through the REST API, not the `releases/download/` browser URL, because that
URL does not serve private repositories.

**b. From a local clone** — no credentials beyond git access:

```sh
make build
```

**c. `go install`** — convenient, but it needs two extra things because the
module is private: git credentials for github.com, and `GOPRIVATE` set so the
Go module proxy is bypassed (without it you get a 404 from `proxy.golang.org`,
which cannot see the repository).

```sh
GOPRIVATE=github.com/Eric-Fernald/* go install github.com/Eric-Fernald/chiton/cmd/chiton@latest
```

That lands in `$(go env GOPATH)/bin`, which is usually already on `PATH`. Any
Go new enough to honour the `toolchain` directive works — `go.mod` declares
`go 1.25.5` and pins `toolchain go1.26.4`, which is fetched automatically.

After installing, confirm `chiton` is on `PATH`. If it is not, tell the user
the exact line to add to their shell profile — do not edit their profile for
them. The plugin's MCP server launches `chiton` by name, so it must be
resolvable from `PATH` for the graph tools to appear.

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
`no passphrase available: set CHITON_PASSPHRASE or run interactively`. Two ways
forward, and the user picks:

- Export `CHITON_PASSPHRASE` in the shell profile that starts Claude Code, so
  the server inherits it. The passphrase is never written to the graph, the
  logs, or any config file.
- Or scaffold without encryption: `chiton init --encrypt=false`. Reasonable for
  a graph over already-public code; not for one holding email, Slack, or
  meeting transcripts.

Do not put the passphrase in the plugin's `.mcp.json` or in any file inside the
repository.

`chiton build` is incremental — a SHA-256 content cache skips unchanged files,
so re-running it is cheap.

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
| MCP tools missing entirely | `chiton` not on `PATH` when the server launched | install, then `/reload-plugins` |
| `no passphrase available` | encrypted store, no terminal to prompt on | export `CHITON_PASSPHRASE`, or re-scaffold with `chiton init --encrypt=false` |
| graph store is in use by another chiton process | a writer (sync watch, build, serve --writable) holds the store, or on Windows a running chiton serve blocks a writer | stop the other process (or the assistant's server) and retry |
| needs a one-time read-write pass | a chiton writer crashed, or the store predates this chiton | close other chiton processes and retry; the next open repairs it |
| Tools work but return nothing | graph never built, or built elsewhere | `chiton build .` at the workspace root, then restart |
| `submit_*` tools fail to write | server started read-only | restart with `chiton serve --writable`, or use the CLI submit path |
| `semantic_search` errors | no embeddings | `chiton embed`, then restart the server |
