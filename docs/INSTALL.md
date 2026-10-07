# chiton — Installation Guide

There are two ways to wire chiton into an assistant:

- **`chiton install`** — writes the MCP server registration, an always-on rules
  block, and (for Claude Code) two skill files into each assistant's own config
  files. Covers all nine supported assistants.
- **The Claude Code plugin** — this repository doubles as a Claude Code plugin,
  so one `claude plugin install` registers the MCP server and three skills, and
  its launcher downloads the prebuilt binary for the machine on first use.
  Claude Code only.

This document covers both, plus getting the binary, the per-assistant config
matrix, the three serve transports, the environment variables the ingest
commands read, and troubleshooting.

> chiton is licensed under the PolyForm Strict License 1.0.0 with an additional
> permission for commercial use: use it for any purpose, but do not modify or
> redistribute it. See [`LICENSE`](../LICENSE).

## Prerequisites: the `chiton` binary

The Claude Code plugin downloads the binary for linux and darwin (amd64, arm64)
and windows (amd64, and arm64 under emulation) on first use, so a plugin
install needs nothing from this section (see
[Claude Code plugin](#claude-code-plugin)). `chiton install`, and the plugin on
any other platform or on a machine that cannot reach GitHub, need a `chiton`
binary installed by one of the routes below.

Git 2.30 or newer is needed at runtime for `chiton sync git`, the git hooks, and
`analyze --diff` / `diff_findings` ref resolution (they pass `--end-of-options`).

**`go install`** (needs a Go toolchain):

```sh
go install github.com/Eric-Fernald/chiton/cmd/chiton@latest
```

**Build from source** (needs a Go toolchain):

```sh
git clone https://github.com/Eric-Fernald/chiton.git
cd chiton
make build              # or: go build ./cmd/chiton
```

**Prebuilt release binaries.** `make release` cross-compiles the Makefile's
platform matrix — `linux/amd64`, `linux/arm64`, `darwin/amd64`, `darwin/arm64`,
`windows/amd64` — into `dist/`, then writes `dist/checksums.txt` over the
artifacts it just built:

```sh
make release            # dist/chiton-<version>-<os>-<arch>[.exe] + dist/checksums.txt
```

`scripts/install-chiton.sh` (POSIX `sh`) and `scripts/install-chiton.ps1`
(Windows PowerShell 5.1) download the release asset for the host OS/arch,
`chiton-<tag>-<os>-<arch>[.exe]`, and verify its SHA-256 against that
`checksums.txt`; a binary not listed there is refused, so `checksums.txt` has
to ship with every release. Both files come straight from the release download
URL, `https://github.com/Eric-Fernald/chiton/releases/download/<tag>/`. The
scripts call no GitHub API and need no credentials. "latest" resolves by reading
where `https://github.com/Eric-Fernald/chiton/releases/latest` redirects. The
POSIX script needs `curl` or `wget`, plus `sha256sum` or `shasum`. Windows on
arm64 gets the windows/amd64 binary, which runs under emulation.

| Option | Environment | Meaning |
|---|---|---|
| `--version` / `-Version` | `CHITON_VERSION` | Release tag to install (default `latest`). |
| `--dir` / `-Dir` | `CHITON_INSTALL_DIR` | Install directory: default `$CLAUDE_PLUGIN_DATA/bin` when run as part of the plugin, otherwise `~/.chiton/bin`. |
| `--checksums` / `-Checksums` | | Verify against this local `checksums.txt` instead of downloading the release's own. The plugin launchers pass the copy shipped inside the plugin. |
| `--quiet` / `-Quiet` | | Print errors only. |

The scripts copy the binary beside its target and rename it into place, so a
concurrent launcher never runs a half-written file. They never execute anything
they download. No release has been cut yet, so until one is, use `go install`
or build from source.

**Teams and Outlook need a tagged build.** Microsoft Graph support is behind
`//go:build msgraph`, which the default build *and every release artifact*
deliberately omit: the Graph SDK adds roughly 117 MB of binary for two API
calls. `chiton teams fetch`, `chiton teams serve-webhook`, and
`chiton email outlook` still appear in `--help` in a default build, but fail
with a message telling you to rebuild. To get them:

```sh
make build-msgraph      # or: go build -tags msgraph ./cmd/chiton
```

Everything else is pure Go and always present, including `chiton email ingest`,
`chiton email gmail`, and both `chiton slack` subcommands. (The other optional
tag, `whisper_cgo`, only affects in-process audio transcription; without it
`chiton meeting transcribe` shells out to a whisper CLI on `PATH`.)

`chiton install` records the absolute path of the running `chiton` binary as the
MCP server's launch command, so the assistant can spawn it directly. The plugin
launches the binary its launcher downloaded instead — see
[Claude Code plugin](#claude-code-plugin) below.

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

## Claude Code plugin

The repository is also a Claude Code plugin: `.claude-plugin/plugin.json`
(plugin name `chiton`) and `.claude-plugin/marketplace.json` (marketplace name
`chiton-marketplace`). Installing the plugin is an *alternative* to
`chiton install --assistant claude-code`, not an addition to it — both register
the same MCP server and the same skill markdown.

```sh
claude plugin marketplace add Eric-Fernald/chiton
claude plugin install chiton@chiton-marketplace
```

### Where the published plugin comes from

`main` is the source tree. Its marketplace entry points Claude Code at the
`plugin-dist` branch (`{"source": "github", "repo": "Eric-Fernald/chiton",
"ref": "plugin-dist"}`). Each release (`.github/workflows/release.yml`,
dispatched by hand with a tag, e.g. `gh workflow run release.yml -f tag=v1.0.0`)
cross-compiles five binaries, publishes them with `checksums.txt` as a GitHub
release, then pushes one commit to `plugin-dist`, parented on the previous
release's so installed plugins fast-forward, holding the assembled plugin:

| Path | Contents |
|---|---|
| `.claude-plugin/plugin.json` | the manifest, with `version` set to the tag minus its leading `v` |
| `.mcp.json` | launches `${CLAUDE_PLUGIN_ROOT}/bin/chiton serve --stdio` |
| `skills/` | the three plugin skills |
| `bin/chiton`, `bin/chiton.cmd` | the POSIX `sh` launcher and the Windows launcher |
| `release/tag` | the release this plugin pins, e.g. `v1.0.0` |
| `release/checksums.txt` | that release's SHA-256 list |
| `scripts/install-chiton.{sh,ps1}` | the release installers the launchers call |
| `assets/icon.png`, `LICENSE`, `README.md` | the plugin icon, license, and readme |

The branch holds no binaries. `scripts/build-plugin-dist.sh`
(`make plugin-dist`) assembles it, and refuses a `checksums.txt` that lacks any
of the five assets the launchers download. Until the first release is cut,
`plugin-dist` does not exist and `claude plugin install` fails.

Because the plugin directory contains `bin/`, claude.ai and Cowork will not
install it. chiton needs a local filesystem, so Claude Code is the only target.

### The plugin downloads its binary on first use

The published plugin's `.mcp.json` (template: `packaging/plugin/.mcp.json`)
starts the server through the plugin's launcher:

```jsonc
// .mcp.json (published plugin root)
{
  "mcpServers": {
    "chiton": {
      "command": "${CLAUDE_PLUGIN_ROOT}/bin/chiton",
      "args": ["serve", "--stdio"],
      "env": { "CHITON_PROJECT_DIR": "${CLAUDE_PROJECT_DIR}" }
    }
  }
}
```

On Windows, Claude Code resolves that command to `bin/chiton.cmd`, which runs
`scripts/install-chiton.ps1` through `powershell -ExecutionPolicy Bypass`. On
Linux and macOS it runs the `bin/chiton` sh launcher, which runs
`scripts/install-chiton.sh`. Either way, the first launch:

1. reads the pinned tag from `release/tag`;
2. downloads `chiton-<tag>-<os>-<arch>[.exe]` (about 25 MB) from
   `https://github.com/Eric-Fernald/chiton/releases/download/<tag>/`;
3. verifies its SHA-256 against `release/checksums.txt`, which arrived with the
   plugin through git rather than from the same place as the binary, and
   refuses the binary on a mismatch;
4. caches it as `~/.chiton/versions/<tag>/chiton`
   (`%USERPROFILE%\.chiton\versions\<tag>\chiton.exe` on Windows) and runs it.

Later launches find the cached binary and start it directly. A plugin update
that moves to a new tag downloads that tag's binary once. The binary therefore
always comes from the same release as the plugin's skills. Windows on arm64
gets the amd64 binary, which runs under emulation. Linux and macOS need `curl`
or `wget`, plus `sha256sum` or `shasum`; Windows needs only the built-in
PowerShell. Launcher and installer messages go to stderr, because stdout
carries the MCP protocol.

The first MCP start waits for the download before the server answers. On a
slow link that can exceed Claude Code's MCP startup timeout. If it does, wait
for the download to finish (the file appears under `~/.chiton/versions/<tag>/`),
then reconnect with `/mcp` or `/reload-plugins`; the next launch uses the cache.

chiton does not clean up `~/.chiton/versions/`. Directories for earlier tags
stay after an update, and uninstalling the plugin leaves the whole directory
behind. Delete it by hand to reclaim the space.

`bin/` is also on the Bash tool's `PATH` while the plugin is enabled. Both
launchers (`bin/chiton` and `bin/chiton.cmd`) resolve the real binary in order:

1. `$CHITON_BIN`;
2. in a published plugin only, the cached pinned binary, downloading it first
   if it is missing;
3. `chiton` on `PATH`;
4. `$CLAUDE_PLUGIN_DATA/bin/chiton`;
5. `$HOME/.chiton/bin/chiton`.

If the download fails, for example offline, the launcher moves on to step 3. On
a machine that cannot reach GitHub, or on a platform with no release binary,
install one by a route under [Prerequisites](#prerequisites-the-chiton-binary);
the `/chiton:setup` skill walks a user through this.

The repo-root `.mcp.json` on `main` still launches `chiton` by bare name. It
serves development of chiton itself (`claude --plugin-dir .` from a working
tree). A working tree has no `release/tag`, so the launcher skips step 2 and
`chiton` has to resolve on `PATH`.

Unlike the installer's absolute-path entry, the plugin's command has no idea
which workspace it belongs to — hence `CHITON_PROJECT_DIR`. With it set, chiton looks
up `.chitonrc` in that directory and resolves a relative `graph_path` against
it, instead of against whatever working directory the assistant happened to
launch the server in. Unset (the plain CLI case), behaviour is unchanged.

### The three plugin skills

Skills are namespaced by plugin name, so they are invoked as `/chiton:<skill>`:

| Skill | Source file | Covers |
|---|---|---|
| `/chiton:graph` | `skills/graph/SKILL.md` | Graph queries — which tool answers which question, provenance, known limitations. |
| `/chiton:subagent-tasks` | `skills/subagent-tasks/SKILL.md` | The `pending_*` → `submit_*` flows: semantic extraction, LLM rules, meeting→code linkage. |
| `/chiton:setup` | `skills/setup/SKILL.md` | First run — locate or install the binary, build the graph, verify the server answers. |

`chiton install --assistant claude-code` writes the first two of those from the
same embedded bytes (see [SKILL.md](SKILL.md)); it does not write `setup`, since
anyone running `chiton install` already has a working binary.

## Per-assistant matrix

`--assistant` names, the config files written, and per-installer caveats. Paths
shown as `~/...` are user-global; the rest are relative to `--dir` (your repo
root). The MCP entry is always registered under the key `chiton`.
Config writes keep an existing file's permissions and write through a symlink, so the link survives and a dangling link is an error. A newly created MCP config is owner-only (`0600`) because these files often hold other servers' API keys.

| Assistant | install name | MCP config file (project) | MCP config (user-scope) | Rules file | Notes |
|---|---|---|---|---|---|
| Claude Code (CLI + IDE extension) | `claude-code` | `.mcp.json` | *not supported* | `CLAUDE.md` | Also writes **two** skills: `.claude/skills/chiton/SKILL.md` (graph queries, the `/chiton` command) and `.claude/skills/chiton-subagent-tasks/SKILL.md` (the `pending_*`/`submit_*` flows). `uninstall` removes both directories and `doctor` reports each one. `--user-scope` is rejected. `--with-hooks` only prints a "not yet implemented" note. Restart Claude Code to pick up the server. |
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
| stdio | `--stdio` | default | Assistant-launched MCP server; the default when no transport flag is given. This is what every installer records. |
| HTTP / SSE | `--http <addr>` | unset | Shared daemon across sessions / debugging (only when `--http <addr>` is given). Every request needs the bearer token (`Authorization: Bearer <token>`). |
| Unix socket | `--unix <path>` | *(empty)* | Low-latency shared daemon over a Unix domain socket (Linux/macOS). |

Other `serve` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--writable` | `false` | Open the store read-write so the `submit_*` write tools work over MCP. The server then holds the store exclusively: no other chiton process (`build`, `embed`, `cluster`, `query`) can open it until the server exits. |
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

## Environment variables

Nothing here is required for a local-only install: the graph, the rule engine,
and the MCP tools all work with no environment at all. These matter when you
point chiton at an external source, or when the server needs to find its way
back to the right workspace.

**Wiring and config**

| Variable | Used by | Meaning |
|---|---|---|
| `CHITON_PROJECT_DIR` | every command | Anchor directory: `.chitonrc` is read from here and a relative `graph_path` resolves against it. The plugin sets it to `${CLAUDE_PROJECT_DIR}`. `.chiton/` state (baselines, LLM-rule state, keyfile lookup) resolves here too. Unset = the process working directory. |
| `CHITON_GRAPH_PATH` | every command | Overrides `graph_path` from `.chitonrc` (the config layer maps any `CHITON_*` name onto its key, e.g. `CHITON_EMBED_PROVIDER`, `CHITON_EMBED_DIM`, `CHITON_SEMANTIC_CONFIDENCE_THRESHOLD`, `CHITON_VECTOR_INDEX`). |
| `CHITON_PASSPHRASE` | encrypted stores | Passphrase for an encrypted graph (`chiton init` encrypts by default; `--encrypt=false` opts out). Without it, commands prompt interactively — which an assistant-launched server cannot do. `chiton init` prints this reminder. |
| `CHITON_PLAINTEXT_OK` | every command that opens a store | Set to `1` to hide the notice printed when a store has no keyfile. |
| `CHITON_MCP_TOKEN` | `chiton serve --http` | Pins the bearer token (at least 16 characters) instead of a fresh one per start written to `<graph dir>/mcp-token`. |
| `CHITON_BIN` | plugin `bin/chiton` / `bin/chiton.cmd` launchers | Explicit path to the real binary, checked before the plugin's pinned download and `PATH`. Setting it also skips the first-run download. |
| `CHITON_VERSION`, `CHITON_INSTALL_DIR` | `scripts/install-chiton.{sh,ps1}` | Release tag to fetch (default `latest`) and install directory. |

**Slack** (`chiton slack fetch`, `chiton slack serve-events`)

| Variable | Meaning |
|---|---|
| `CHITON_SLACK_TOKEN` | Bot/user OAuth token (`xoxb-…`). Needs `channels:history` / `groups:history`, plus `users:read` to resolve author names. Required by both subcommands. |
| `CHITON_SLACK_SIGNING_SECRET` | Slack app signing secret, used to verify every Events API request. `chiton slack serve-events` **refuses to start** without it (or a file named by `--signing-secret-file`). |

`serve-events` binds `127.0.0.1:8445` by default and warns loudly if `--addr`
points at a wildcard address; Slack needs a public HTTPS Request URL, so put a
tunnel or reverse proxy in front rather than binding wide. The receiver acks each request immediately and ingests on a single background worker, so Slack's 3-second ack deadline is always met; on Ctrl-C it finishes queued ingests for up to 10 seconds, then cancels what is still running.

**Microsoft Teams** (`chiton teams …` — requires a `-tags msgraph` build)

| Variable | Meaning |
|---|---|
| `CHITON_TEAMS_TENANT_ID`, `CHITON_TEAMS_CLIENT_ID`, `CHITON_TEAMS_CLIENT_SECRET` | App-only Graph credentials. |
| `CHITON_TEAMS_USER_ID` | The meeting organizer whose meetings to read (app-only auth has no `/me`). |
| `CHITON_TEAMS_CLIENT_STATE` | Shared secret validating Graph change notifications. `chiton teams serve-webhook` **refuses to start** without it (or a file named by `--client-state-file`) rather than accepting unauthenticated notifications. |

`serve-webhook` binds `127.0.0.1:8444` by default, with the same wildcard
warning as the Slack receiver. The receiver acks each request immediately and ingests on a single background worker, so Graph's 3-second ack deadline is always met; on Ctrl-C it finishes queued ingests for up to 10 seconds, then cancels what is still running.

**Email**

| Variable | Meaning |
|---|---|
| `CHITON_GMAIL_TOKEN` | OAuth2 access token with the `gmail.readonly` scope, for `chiton email gmail`. Pure Go — no build tag needed. |
| `CHITON_OUTLOOK_TENANT_ID`, `CHITON_OUTLOOK_CLIENT_ID`, `CHITON_OUTLOOK_CLIENT_SECRET`, `CHITON_OUTLOOK_USER_ID` | App-only Graph credentials and the mailbox to read, for `chiton email outlook` (requires a `-tags msgraph` build). |

`chiton email ingest` (local `.eml` / maildir / `.mbox`) and `chiton browser
ingest` read only local files and need no environment at all.

## Troubleshooting

### The MCP server won't start

- **Restart / reload the assistant.** Most installers print a reminder
  (`Restart Claude Code…`, `Reload the VS Code window…`, etc.); MCP configs are
  read at startup.
- **Check the binary path.** The MCP entry's `command` is the absolute path of
  the `chiton` binary captured at install time. If you moved or rebuilt the
  binary elsewhere, re-run `chiton install` so the path is refreshed, or pass the
  intended binary.
- **Plugin, first start timed out.** The first launch of each plugin version
  downloads about 25 MB before the server answers, which can outlast Claude
  Code's MCP startup timeout. Once `~/.chiton/versions/<tag>/` holds the binary,
  reconnect with `/mcp` or `/reload-plugins`. If the download itself fails
  (offline, a proxy blocking `github.com`, a checksum mismatch, or no `curl` /
  `wget` / `sha256sum` / `shasum` on Linux or macOS), the launcher falls back to
  `chiton` on `PATH`; install one by a route under
  [Prerequisites](#prerequisites-the-chiton-binary), or download the release
  asset yourself and point `CHITON_BIN` at it.
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
- With the **plugin**, there is no recorded `--graph` — the entry is
  `${CLAUDE_PLUGIN_ROOT}/bin/chiton serve --stdio`. What locates the store instead is
  `CHITON_PROJECT_DIR`, which the plugin sets to `${CLAUDE_PROJECT_DIR}`: the
  server reads `.chitonrc` from that directory and resolves a relative
  `graph_path` against it. If the tools come back empty under the plugin, check
  that the variable reached the server, or set an absolute `graph_path` in
  `.chitonrc` (or `CHITON_GRAPH_PATH`) to remove the ambiguity.

### `submit_*` tools fail: read-only vs `--writable`

By default `chiton serve` opens the store **read-only** so other read-only
commands can run beside it (and, on Linux/macOS, writers such as
`build`/`embed`; on Windows a writer fails fast while the server runs). The read
tools work in this mode, but the write tools (`submit_extraction`,
`submit_analysis`, `submit_meeting_links`, `add_url`) need a writable store.
Either:

- start the server with `chiton serve --writable` (the server then holds the
  store exclusively), **or**
- use the CLI write path instead, e.g. `chiton extract submit <file>`,
  `chiton analyze submit <file>`, `chiton meeting submit <file>`.

On Windows the CLI path needs the read-only server stopped first.

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
