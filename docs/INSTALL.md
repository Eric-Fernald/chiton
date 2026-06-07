# chiton — Installation Guide

`chiton install` wires the chiton MCP server, an always-on rules block, and
(for Claude Code) a skill file into each AI assistant you use. This document
covers the quick start, the per-assistant config matrix, the three serve
transports, and troubleshooting.

> chiton is distributed under a proprietary license — see [`LICENSE`](../LICENSE).

## Prerequisites

Build or install the `chiton` binary first (see the README):

```sh
go build -o chiton ./cmd/chiton
# or install onto your PATH:
go install github.com/Eric-Fernald/chiton/cmd/chiton@latest
```

`chiton install` records the absolute path of the running `chiton` binary as the
MCP server's launch command, so the assistant can spawn it directly.

## Quick start

Run `install` from your repo root. It writes config files and an always-on rules
block, then prints a per-file report (`wrote` / `skip` / `note` lines).

```sh
# Auto-detect: install only for assistants whose footprint is found in this repo.
# This is the default (--assistant auto).
chiton install

# Every supported assistant, whether detected or not.
chiton install --assistant all

# One named assistant (or a comma-separated list).
chiton install --assistant claude-code
chiton install --assistant cursor,codex

# Write into the user-global config location instead of the repo
# (only where the assistant supports it — see the matrix below).
chiton install --assistant cursor --user-scope

# Report the install state across every supported assistant. Writes nothing.
chiton doctor
chiton doctor --json

# Remove chiton's entries again (only chiton-owned content is touched).
chiton uninstall --assistant claude-code
chiton uninstall --assistant all
```

If `--assistant auto` (the default) finds no assistant evidence in the
workspace, `install` errors out and asks you to pass `--assistant <name>` or
`--assistant all`.

### `install` / `uninstall` flags

| Flag | Default | Applies to | Meaning |
|---|---|---|---|
| `--assistant` | `auto` | install, uninstall | `auto`, `all`, a name (e.g. `claude-code`), or a comma-separated list. |
| `--dir` | `.` | install, uninstall, doctor | Workspace directory to operate on. Per-repo files are resolved relative to this. |
| `--user-scope` | `false` | install | Write into the user-global config location where supported; installers that have none return an "installer does not support --user-scope" error. |
| `--with-hooks` | `false` | install | Opt in to the assistant's PreToolUse-style hook integration (not yet implemented; currently emits a note for Claude Code). |
| `--graph` | *(empty)* | install | Graph store path to record in the MCP config so `chiton serve` uses the same store as `chiton build`. Empty = chiton's built-in default. |

`doctor` additionally accepts `--json` to emit the structured report.

### Auto-detect (`--assistant auto`)

Each installer's `Detect()` looks for that assistant's footprint in `--dir`:

| Assistant | Detected when any of these exist in the workspace |
|---|---|
| claude-code | `CLAUDE.md`, `.claude/` |
| cursor | `.cursor/mcp.json`, `.cursor/rules/`, `.cursorrules`, `.cursor/` |
| codex | `.codex/` |
| gemini-cli | `.gemini/settings.json`, `GEMINI.md`, `.geminiignore`, `.gemini/` |
| opencode | `opencode.json`, `opencode.jsonc`, `.opencode/opencode.json`, `.opencode/` |
| aider | `.aider.conf.yml`, `.aider.conf.yaml`, `CONVENTIONS.md`, `.aiderignore` |
| antigravity | `.agent/rules/` |
| copilot-vscode | `.github/copilot-instructions.md`, `.vscode/mcp.json`, `.vscode/` |
| copilot-cli | `.copilot/` |

Detection is workspace-scoped even for assistants whose MCP config is
user-global (codex, antigravity, copilot-cli): `Detect()` reports evidence in
*this repo* only.

## Per-assistant matrix

`--assistant` names, the config files written, and per-installer caveats. Paths
shown as `~/...` are user-global; the rest are relative to `--dir` (your repo
root). The MCP entry is always registered under the key `chiton`.

| Assistant | install name | MCP config file (project) | MCP config (user-scope) | Rules file | Notes |
|---|---|---|---|---|---|
| Claude Code (CLI + IDE extension) | `claude-code` | `.mcp.json` | *not supported* | `CLAUDE.md` | Also writes `.claude/skills/chiton/SKILL.md` (the `/chiton` slash-command reference). `--user-scope` is rejected. `--with-hooks` only prints a "not yet implemented" note. Restart Claude Code to pick up the server. |
| Cursor editor + CLI | `cursor` | `.cursor/mcp.json` | `~/.cursor/mcp.json` | `.cursor/rules/chiton.mdc` | Rule file uses MDC frontmatter with `alwaysApply: true`. The project rule is always written into the repo, even with `--user-scope` (only the MCP config moves). Reload the window to pick up the server. |
| OpenAI Codex CLI | `codex` | *(none — user-scope only)* | `~/.codex/config.toml` | `AGENTS.md` | MCP config is **always** user-scope (a single TOML file with a `[mcp_servers.chiton]` table); it applies to Codex in every project. Project-scope `.codex/config.toml` is honored only for trusted projects, so the installer does not target it. |
| Google Gemini CLI | `gemini-cli` | `.gemini/settings.json` | `~/.gemini/settings.json` | `GEMINI.md` | Rules go to `GEMINI.md` (Gemini CLI auto-loads it, not `AGENTS.md`). Restart Gemini CLI to pick up the server. |
| OpenCode terminal agent | `opencode` | `opencode.json` | `~/.config/opencode/opencode.json` | `AGENTS.md` | OpenCode's MCP servers live under a top-level `mcp` object (singular, **not** `mcpServers`); the user-scope path is the same on every platform. Restart OpenCode to pick up the server. |
| Aider pair-programmer | `aider` | *(none — no MCP support)* | *(none)* | `CONVENTIONS.md` | **No native MCP.** The installer only wires `CONVENTIONS.md` (rules block) and adds it to the `read:` list in `.aider.conf.yml`; the rules tell Aider to shell out to the `chiton` CLI instead. Run `chiton build .` first. `--user-scope` is rejected. |
| Google Antigravity IDE | `antigravity` | *(none — user-scope only)* | `~/.gemini/config/mcp_config.json` | `AGENTS.md` | MCP config is **always** user-scope (a separate file from Gemini CLI's `settings.json`, so they don't collide); it applies to Antigravity in every project. Rules go to `AGENTS.md` (not `GEMINI.md`) to avoid the shared `~/.gemini/GEMINI.md` collision. |
| GitHub Copilot Chat in VS Code | `copilot-vscode` | `.vscode/mcp.json` | *not supported* | `.github/copilot-instructions.md` | `--user-scope` is rejected. Reload the VS Code window to pick up the server. |
| GitHub Copilot CLI (terminal agent) | `copilot-cli` | *(none — user-scope only)* | `~/.copilot/mcp-config.json` | `AGENTS.md` | MCP config is **always** user-scope (or `$COPILOT_HOME/mcp-config.json` when `COPILOT_HOME` is set, which *replaces* the whole `~/.copilot` path); it applies to Copilot CLI in every project. The entry carries `type`/`env`/`tools` fields to match Copilot's documented local-server shape. |

### Shared `AGENTS.md` (codex, opencode, antigravity, copilot-cli)

These four installers share a single repo-root `AGENTS.md` rules block. On
`uninstall`, the block is **reference-counted**: it is only removed once no other
`AGENTS.md`-using assistant still has its chiton MCP entry registered. If another
is still installed, the block is kept and `uninstall` notes:
`AGENTS.md: kept chiton rules block (still used by another installed assistant)`.

## Serve transports

`chiton serve` exposes the identical MCP tool surface over three transports. The
installers always wire **stdio** (the assistant launches `chiton serve --stdio`);
the HTTP and Unix transports are for a shared daemon or debugging and are not
written into any assistant config.

| Transport | Flag | Default | Use |
|---|---|---|---|
| stdio | `--stdio` | off | Assistant-launched MCP server. This is what every installer records. |
| HTTP / SSE | `--http <addr>` | `:7474` | Shared daemon across sessions / debugging (default transport when neither `--stdio` nor `--unix` is given). |
| Unix socket | `--unix <path>` | *(empty)* | Low-latency shared daemon over a Unix domain socket (Linux/macOS). |

Other `serve` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--writable` | `false` | Open the store read-write so the `submit_*` write tools work over MCP. This takes the store's write lock, so concurrent `build`/`embed` cannot run. |
| `--graph` | *(empty)* | Graph store path to serve. Overrides the configured/built-in `graph_path`; this is what `chiton install --graph <path>` records so the server resolves the intended store regardless of its working directory. |

The exact argv the installers record is `serve --stdio` (plus `--graph <path>`
when `--graph` was passed to `install`):

```jsonc
// .mcp.json (Claude Code), .vscode/mcp.json (Copilot VS Code),
// .cursor/mcp.json (Cursor), .gemini/settings.json (Gemini CLI), etc.
{
  "mcpServers": {
    "chiton": {
      "command": "/abs/path/to/chiton",
      "args": ["serve", "--stdio"]
    }
  }
}
```

## Troubleshooting

### The MCP server won't start

- **Restart / reload the assistant.** Most installers print a reminder
  (`Restart Claude Code…`, `Reload the VS Code window…`, etc.); MCP configs are
  read at startup.
- **Check the binary path.** The MCP entry's `command` is the absolute path of
  the `chiton` binary captured at install time. If you moved or rebuilt the
  binary elsewhere, re-run `chiton install` so the path is refreshed, or pass the
  intended binary.
- **Run the server by hand** to see the error the assistant is swallowing:
  ```sh
  chiton serve --stdio
  ```
  At startup the server logs whether `semantic_search` is enabled and the
  embedding count; if no embeddings are indexed it notes
  `semantic_search disabled (run 'chiton embed')` — the structural tools still
  serve.

### Wrong `--graph` path (server serves an empty / stale graph)

If the assistant's tools return nothing, the server is likely opening a
different store than the one `chiton build` wrote.

- When `--graph` is omitted, both the entry and the server fall back to chiton's
  configured/built-in default `graph_path`. If you built into a non-default
  location, re-install with the explicit path:
  ```sh
  chiton install --assistant <name> --graph /abs/path/to/store
  ```
  This records `serve --stdio --graph /abs/path/to/store` in the MCP config so
  the resolved store is independent of the server's working directory.
- Verify with `chiton serve --graph /abs/path/to/store` and watch the startup
  log line for the embedding count.

### `submit_*` tools fail: read-only vs `--writable`

By default `chiton serve` opens the store **read-only** so `chiton build`,
`embed`, and `cluster` can run concurrently. The read tools work in this mode,
but the write tools (`submit_extraction`, `submit_analysis`,
`submit_meeting_links`, `add_url`) need a writable store. Either:

- start the server with `chiton serve --writable` (this takes the write lock, so
  concurrent `build`/`embed` cannot run), **or**
- use the CLI write path instead, e.g. `chiton extract submit <file>`,
  `chiton analyze submit <file>`, `chiton meeting submit <file>`.

### Reading `chiton doctor` output

`doctor` writes nothing. For each assistant it prints a status line followed by
`ok:` and `issue:` detail lines:

| Status | Meaning |
|---|---|
| `registered` | chiton's MCP entry **and** rules block are both present (for Aider: rules block + `CONVENTIONS.md` in `read:`). |
| `detected (not registered)` | the assistant's footprint exists in this repo, but chiton is not fully wired up. |
| `not detected` | no evidence of this assistant in the workspace. |

Each `issue:` line names the missing file and the command to fix it, e.g.
`.mcp.json: no chiton entry (run 'chiton install --assistant claude-code')`. Run
`doctor --json` for the machine-readable `{detected, registered, ok, issues}`
shape per installer.

### Re-running install is idempotent

Re-running `chiton install` is safe. The chiton block inside a rules file is
replaced in place (not appended), and the `chiton` MCP entry is upserted under
`mcpServers` (or the Codex `[mcp_servers.chiton]` table / OpenCode `mcp` object)
while every unrelated key, server, and note is preserved. When an entry already
existed, the report notes `refreshed existing chiton entry`. `uninstall` is
likewise surgical — it removes only chiton-owned content and deletes an MCP
config file only if doing so would leave it completely empty.
