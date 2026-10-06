# chiton Security & Privacy

chiton reads source code, documentation, images, meeting transcripts and meeting
audio, Slack conversations, email bodies, local browser history, and web pages
and PDFs you point it at — any of which can contain secrets, PII, customer data,
or NDAed material. This document describes how chiton handles that data, what is
encrypted, what stays on the device, what can leave it, and the threat model it
is (and is not) designed to defend against.

Throughout, **shipped** means the behaviour is implemented in the current
codebase; **planned** means it is described in the design (`docs/PLAN.md`) but is
not yet wired into the binary. We are explicit about the distinction so you can
make an informed trust decision.

> **License note:** chiton is proprietary software (see `LICENSE`: "All rights
> reserved"; the repository is private). This document describes its security
> posture; it is not a warranty.

---

## 0. What a default install actually does

Most of this document describes optional commands you have to run deliberately.
This section describes the baseline — an assistant that has chiton installed and
has not been asked to ingest anything.

- The Claude Code plugin's `.mcp.json` launches **`chiton serve --stdio`**. It
  invokes the binary by **bare name**, so `chiton` must be on `PATH`, and sets
  `CHITON_PROJECT_DIR=${CLAUDE_PROJECT_DIR}` so the server reads the same store
  `chiton build` wrote regardless of its working directory.
- **stdio binds no port.** The process talks to the assistant over its own
  stdin/stdout. Nothing is listening.
- The store is opened **read-only** unless you pass `--writable`.
- **19 MCP tools** are registered. 15 are read-only queries against the local
  (normally encrypted) graph. The 4 write tools — `submit_extraction`,
  `submit_analysis`, `submit_meeting_links`, `add_url` — are still *listed*, but
  fail against a read-only store; `add_url` refuses **before** it fetches
  anything (`internal/ingest/url/ingest.go`).
- **Exactly one tool causes any socket activity: `semantic_search`.** It POSTs
  the query text to the configured embedding provider —
  `http://localhost:11434/api/embed` by default — to turn the question into a
  vector. Every other read tool is pure local I/O. If you never ran
  `chiton embed`, `semantic_search` is disabled outright and even that call does
  not happen.
- chiton **calls no LLM of its own**, in any mode.

Everything else in this document — Slack, email, Teams, browser history, URL
ingest — happens only when you run the corresponding command.

---

## 1. Data-handling posture

chiton is a single, statically-linked Go binary that runs **on-device**. The
graph it builds — code symbols, doc sections, findings, meetings, decisions,
emails, URLs, and embeddings — is stored locally in a BadgerDB store at
`graph_path`, **`./chiton-out/graph.kuzu` by default** (the name is historical).

**In the default subagent mode, chiton calls no LLM itself.** Semantic
extraction, LLM rules, and meeting→code linkage all run through the host
assistant via the `pending_*` / `submit_*` task flow: chiton emits a task spec,
the assistant (which you already trust with the code) does the reasoning, and the
result is written back. chiton does not open outbound LLM connections to perform
this work.

| Property | Status |
|---|---|
| Single binary, on-device graph | Shipped |
| No built-in LLM calls (subagent mode) | Shipped |
| No telemetry / phone-home / update check | Shipped — see §1.4 |
| Embeddings can stay local | Shipped — `chiton embed` defaults to local Ollama (`http://localhost:11434`); `CHITON_EMBED_PROVIDER=hash` is fully offline |
| All network egress is command-triggered except `semantic_search` | Shipped — see §1.1 |

### 1.1 Complete egress inventory

chiton can reach seven destinations. Four are direct `net/http` call sites, two
go through the Microsoft Graph SDK and its Azure identity library, and one is a
subprocess. Nothing here fires on its own schedule — each is triggered by the
command or tool named in the third column.

| Destination | Code | Triggered by | Data sent |
|---|---|---|---|
| `http://localhost:11434/api/embed` (configurable) | `internal/embed/ollama.go` | `chiton embed`, `chiton search`, `chiton sync … --embed`, and the **`semantic_search` MCP tool** | Node text / the query string to be embedded |
| `https://slack.com/api/*` | `internal/ingest/slack/web_client.go` | `chiton slack fetch`, and **automatically, per verified inbound event, while `chiton slack serve-events` runs** | Bot/user OAuth token + channel/thread ids; receives message content |
| `https://gmail.googleapis.com` | `internal/ingest/email/gmail.go` | `chiton email gmail` | OAuth2 access token + your query; receives message bodies |
| `https://graph.microsoft.com` | `internal/ingest/email/outlook.go`, `internal/ingest/teams/graph_client.go` | `chiton email outlook`, `chiton teams fetch`, and **automatically, per verified inbound notification, while `chiton teams serve-webhook` runs** | Graph token; receives mail / meeting transcripts |
| `https://login.microsoftonline.com` | `azidentity` (vendored) | The same three commands, to mint a token | **Your Teams/Outlook tenant id, client id, and client secret** |
| **Any host you name** | `internal/ingest/url/fetch.go` | `chiton add-url`, and the **`add_url` MCP tool on a `--writable` server** | An HTTP GET with a chiton user-agent; receives the page/PDF |
| `youtube.com` (via the `yt-dlp` subprocess) | `internal/ingest/url/fetch.go` | `chiton add-url` / `add_url` with a YouTube URL | Whatever `yt-dlp` sends; receives the transcript |

Notes that materially change the picture:

- **`semantic_search` is the only egress reachable from the default MCP surface
  without an explicit user command.** It is loopback by default, and
  `CHITON_EMBED_PROVIDER=hash` removes it entirely — but
  `embed.endpoint` / `CHITON_EMBED_ENDPOINT` can redirect it off-device, at
  which point node text and queries go wherever you pointed it.
- **Microsoft Graph is behind the `msgraph` build tag and is absent from default
  and release binaries.** `chiton teams fetch`, `chiton teams serve-webhook`,
  and `chiton email outlook` still appear in `--help`, but fail with a
  "rebuild with `-tags msgraph`" error. Build them yourself with
  `make build-msgraph` if you want them. `chiton email ingest` and
  `chiton email gmail` are unaffected.
- **`AZURE_AUTHORITY_HOST` is read by the vendored Azure SDK, not by chiton.**
  If it is set in your environment, it redirects where the client secret above
  is sent. chiton neither reads nor validates it.
- `hostAllowed` (§5.3) performs a **DNS lookup** for any host passed to
  `add-url`, which is itself observable traffic.

### 1.2 Subprocesses

chiton is CGo-free by default, but it is not hermetic — it shells out to three
external programs:

| Program | When | Notes |
|---|---|---|
| `git` | `chiton sync git`, `chiton analyze --since`, repo detection | Read-only plumbing (`rev-parse`, log/diff) |
| `yt-dlp` | `chiton add-url` / `add_url` on a YouTube URL | Looked up on `PATH`; downloads subtitles into a temp dir that is removed afterwards. **This branch bypasses the `hostAllowed` SSRF check** (§5.3) |
| A whisper CLI | `chiton meeting transcribe` | **Auto-detected from `PATH`**, trying `whisper`, `whisper-cli`, `whisper-cpp`, `whisper.cpp` in order (`internal/ingest/audio/shellout.go`) |

The whisper auto-detection is the one worth a second look before you trust a
shared machine: chiton runs the first matching name on `PATH` and hands it your
meeting audio. It does not verify what that binary is or where it sends data. An
in-process whisper.cpp backend exists but requires a `-tags whisper_cgo` build.

### 1.3 Where data is written

| Path | Contents | Encrypted? |
|---|---|---|
| `chiton-out/graph.kuzu/` (default `graph_path`) | The BadgerDB graph: all node text, transcripts, email bodies, embeddings | **Yes**, when a keyfile is present (§2) |
| `chiton-out/hnsw.idx` | Saved HNSW index: node IDs and graph links, no vectors or text | **Yes**, when the store is encrypted: AES-256-GCM under a key derived from the store's (§2) |
| `chiton-out/cluster.json`, `chiton-out/cache.json` | Cluster sidecar and content-hash cache | **No — plaintext** |
| `.chiton/keyfile.json` | Argon2id salt + KDF parameters only (no key, no passphrase) | n/a — no secret in it |
| `.chiton/hooks.yaml`, `sources.yaml`, `schema_version`, `.gitignore` | Policy files | **No — plaintext**, and `hooks.yaml`/`sources.yaml` are meant to be committed |
| `.chiton/baselines/<commit>.json`, `.chiton/llm-analyzed.json` | Finding baselines and LLM-rule state | **No — plaintext** |
| `chiton export --out <file>` | Full graph snapshot, embeddings inlined | **No — plaintext JSON** |
| OS temp dir | A copy of the browser History DB (`chiton-hist-*`), yt-dlp scratch (`chiton-yt-*`) | **No** — removed after use |

**`.chiton/` is not the graph.** Earlier revisions of this document told you to
protect `.chiton/`; that was wrong. The encrypted data lives under
`chiton-out/`, and `.chiton/` holds plaintext policy plus the (secret-free)
keyfile. `chiton init` writes a `.chiton/.gitignore` that keeps everything but
`hooks.yaml` and `sources.yaml` out of git; **`chiton-out/` is not
auto-ignored** — add it to your repo's `.gitignore` yourself.

**Outside the project directory:** `chiton install` and `chiton uninstall` write
assistant configuration into your **home directory**, not just the repo —
`~/.codex/config.toml`, `~/.config/opencode/opencode.json`,
`~/.gemini/settings.json` and `~/.gemini/config/mcp_config.json`,
`~/.cursor/mcp.json`, and `~/.copilot/` (or `$COPILOT_HOME`). These are MCP
server registrations and rules files; they contain paths, not secrets. `chiton
install` for Claude Code additionally writes `.claude/skills/chiton/SKILL.md`
and `.claude/skills/chiton-subagent-tasks/SKILL.md` inside the project.

`.chitonrc` is read from `$CHITON_PROJECT_DIR` (or the working directory) **and
from `$HOME`**, so a home-directory config can influence any project.

### 1.4 Telemetry

There is none, and this is checkable rather than merely asserted:

- No analytics, crash-reporting, or metrics SDK is imported anywhere
  (`go.mod` has no such dependency).
- There is no version or update check. `chiton version` reads only ldflags and
  the Go build info embedded in the binary — it is entirely offline.
- OpenTelemetry appears in `go.mod` **only** as an indirect dependency of the
  Microsoft Graph SDK, and no exporter is ever configured — so nothing is
  emitted even in an `msgraph` build.
- The seven destinations in §1.1 are the complete set: grepping `internal/` and
  `cmd/` for `http.Client`, `http.Get`, `http.Post`, `http.NewRequest`, and
  `net.Dial` finds no others.

### 1.5 Generated HTML reports

`chiton dashboard` and `chiton visualize` write self-contained HTML files with
the graph data inlined. **Both load the vis-network library from
`https://unpkg.com`.** The binary makes no such request — your *browser* does,
when you open the file, which discloses to a CDN that you opened it.

Both pages carry the same restrictive CSP, quoted here in full:

```
default-src 'none'; script-src https://unpkg.com 'unsafe-inline';
style-src 'unsafe-inline'; img-src data:; connect-src 'none';
base-uri 'none'; object-src 'none'
```

`connect-src 'none'` is the load-bearing part: the page cannot exfiltrate the
graph data it embeds even if the CDN script were hostile. Note that `style-src`
and `img-src` are allowed separately, so `default-src 'none'` does not govern
them. Still don't open either file on a machine where the inlined graph would be
sensitive if the CDN were compromised — a CSP constrains the page, not what the
CDN script can read inside it.

> **Planned, not shipped:** the design calls for a per-source allowlist
> (`sources.yaml` with `allow_remote_llm` / `allow_remote_embed`), a
> gitleaks-based secret-scrubbing pass before any remote call, and an
> `audit.jsonl` log of every remote call. `chiton init` writes a `sources.yaml`
> template, but **there is no enforcement code, no scrubbing pass, and no audit
> log in the current binary** — nothing in `internal/` or `cmd/` even reads
> `sources.yaml` back. Treat those as roadmap items, not active controls.

---

## 2. Encryption at rest

### What is encrypted

The **BadgerDB graph store is encrypted at rest** when a keyfile is present. The
store is opened with Badger's native encryption (`WithEncryptionKey`, plus the
required non-zero `IndexCacheSize` of 64 MiB) — see
`internal/graph/store/badger/store.go` (`applyEncryption`). This covers the LSM
tree, value log, and all node/edge/embedding data Badger persists — which, once
you have ingested email, Slack, or transcripts, is where the sensitive text
lives.

> **Scope caveat (shipped reality):** only the Badger store and the saved HNSW
> index (`hnsw.idx`) are encrypted today. The index is sealed with
> `EncryptBytes` under `FileKey`, an HKDF-SHA256 derivation of the store key, so
> it never shares a key with the store; without the key it is not loaded and
> `serve` rebuilds the index instead. The package doc in
> `internal/crypto/keyfile.go` describes the same wrapping for other artifacts
> (`cache.json`, `cluster.json`, `audit.jsonl`), but that is **not yet wired in**.
> Everything in the "No — plaintext" rows of §1.3 stays plaintext: the cluster
> sidecar, the content cache, `.chiton/baselines/`, `.chiton/llm-analyzed.json`,
> and `chiton export` snapshots. The design's "export archive is encrypted with
> the same passphrase" is **planned, not shipped.** Protect exported snapshots
> and sidecars yourself.

### Key derivation and handling

Encryption keys are derived from a **user passphrase** via **Argon2id**; the
passphrase itself is **never written to disk**. The on-disk
`.chiton/keyfile.json` holds only the salt and KDF parameters — without the
passphrase it reveals nothing.

| Parameter | Default (`DefaultKeyFile`) | Notes |
|---|---|---|
| KDF | `argon2id` | Only `argon2id` is accepted |
| Memory | 64 MiB (`memory_kib = 65536`) | Floor enforced at 8 MiB |
| Iterations | 3 | Floor 1 |
| Parallelism | 4 | Floor 1 |
| Salt | 16 random bytes (hex) | Floor 16 bytes |
| Key length | 32 bytes | Must be exactly 32 (AES-256) |
| Cipher | `AES-256-GCM` | |

The derivation is deterministic: the same passphrase + same keyfile always
produce the same 32-byte key, so the encrypted store re-opens reliably. A loaded
keyfile is validated against the floors above so a corrupt or tampered keyfile
fails with a clear error rather than silently deriving a weak key. The keyfile is
written `0o600` and `Save` refuses to overwrite an existing keyfile unless forced
(regenerating the salt would lock you out of existing encrypted data).

### Enabling and disabling

Encryption is **on by default**:

```sh
chiton init            # encryption enabled; prompts you to choose a passphrase
                       #   (entered twice and confirmed)
```

To opt out (e.g. CI runners against synthetic test data):

```sh
chiton init --encrypt=false   # no keyfile; graph stored in plaintext
```

After `init`, subsequent commands obtain the passphrase from the
`CHITON_PASSPHRASE` environment variable, or — if unset and a terminal is
attached — by prompting interactively. The interactive prompt uses
`golang.org/x/term`'s `ReadPassword`, so **the passphrase is never echoed** and
never reaches the terminal's scrollback. In CI / non-interactive environments you
**must** set `CHITON_PASSPHRASE` or the command fails with a clear error:

```sh
export CHITON_PASSPHRASE='…'   # read before any interactive prompt
chiton build .
```

Key resolution is automatic: when chiton opens a graph it looks for a keyfile
next to the store, one directory up in `.chiton/`, and finally in `./.chiton/`
(`internal/crypto/resolve.go`). If a keyfile is found, it derives the key and
opens the store encrypted; if none is found, the store opens in plaintext.

### Threats this addresses

Encryption at rest is aimed squarely at **data-at-rest exposure**:

- **Laptop theft / lost device:** an attacker with the powered-off disk cannot
  read graph contents without the passphrase.
- **Backup leakage:** cloud or full-disk backups (Time Machine, OneDrive,
  Backblaze) that include `chiton-out/` remain unintelligible without the
  passphrase.
- **Stolen store directory:** copying `chiton-out/graph.kuzu/` off a machine or
  out of a backup yields ciphertext only.

It does **not** protect a running, unlocked process (see out-of-scope below), and
it does **not** protect the plaintext sidecars or exported snapshots noted above.

### Operational caveat: hard kill / WAL recovery

chiton writers (notably `chiton sync watch`, the first long-running read-write
holder of the store) should be stopped cleanly with **Ctrl-C** so BadgerDB closes
properly. A hard kill (`SIGKILL` / `taskkill /F`) can leave Badger's
write-ahead log needing recovery on the next open. This is a pre-existing
property of any chiton writer; it is not specific to encryption, but it applies
equally to the encrypted store.

---

## 3. PII handling

### What is tagged

Three ingest paths create `person` nodes, and all three tag them:

- **Meetings.** Speakers become `person` nodes with `Attrs: {"pii": true}`
  (`internal/extract/meeting/graph.go`, `internal/extract/meeting/linkage.go`),
  and a shared `person://` identity URI keeps one person to one node across
  meetings. Slack ingest reuses this same meeting subgraph, so Slack
  participants land in the same place.
- **Email.** `chiton email ingest` / `gmail` / `outlook` create `person` nodes
  for the sender and every recipient, also `pii: true`, **keyed on the
  lowercased literal email address** and sharing the meeting `person://` scheme
  (`internal/ingest/email/email.go`). The address is additionally stored in
  `attrs.email`. So the graph holds real addresses, not just display names.

### Where the `include_pii` gate actually applies — read this part

**PII gating is enforced in exactly one file: `internal/mcp/meetings.go`.** Only
two of the nineteen MCP tools honour it:

- `search_meetings` returns each meeting with its decisions and linked code
  symbols, but **participant names are omitted** unless `include_pii: true` is
  passed; by default only a participant *count* is returned. When `include_pii`
  is false, a query is also **not** matched against the participant list —
  returning a meeting because a queried name attended would itself disclose
  attendance (`matches` / `public`).
- `decisions_about` drops **action-item assignee names** unless `include_pii` is
  set. (There is no separate "action-item tool"; the flag lives here.)

**Every other read tool has no PII gating at all.** `get_node`, `query_graph`,
`get_node_neighborhood`, `shortest_path`, `graph_summary`, `god_nodes`, and
`semantic_search` will return `person` nodes — display names *and* the
`attrs.email` addresses — and will return browser-history `url` nodes, which are
not PII-tagged in the first place. A caller that wants the names does not need
`include_pii`; it just needs to ask a different tool.

Treat the gate as *a courtesy default on the two meeting tools*, not as an
access control over the graph. If the graph must not contain a person's
identifiers, don't ingest the source.

> **Also out of scope today:** PII tagging covers `person` nodes only. Free-text
> PII inside transcript bodies, email bodies, and doc sections is **never
> detected or redacted** — meeting transcript content, titles, decision
> summaries, and email bodies are always searchable, which is the point of the
> tool. The design mentions detecting email addresses and phone numbers in body
> text; that is not shipped.

### Browser history

`chiton browser ingest` reads **only page metadata** — `url`, `title`,
`visit_count`, `last_visit_time` from the `urls` table — and it reads them from
a **copy** of the browser's `History` SQLite file placed in a temp directory and
deleted afterwards (`internal/ingest/browser/history.go`), using a pure-Go
driver. Cookies, `Login Data`, saved credentials, and form autofill stores are
**never opened**, and **no page is fetched** — the graph gets the title you saw,
not the page you saw. Chrome, Edge, and Brave profiles are auto-detected.

That still means your browsing history — every URL and title, including private
ones — is in the graph as `url` nodes, ungated, retrievable by any read tool.

---

## 4. Credentials and secrets

Every secret chiton uses is read **from the environment only**. None is
persisted to disk, written to a config file, or logged.

| Variable | Used by | Sent to |
|---|---|---|
| `CHITON_PASSPHRASE` | Every command that opens an encrypted store | Nowhere — derived locally into an AES key |
| `CHITON_SLACK_TOKEN` | `chiton slack fetch`, `slack serve-events` | `slack.com` as `Authorization: Bearer …` |
| `CHITON_SLACK_SIGNING_SECRET` | `chiton slack serve-events` | Nowhere — used locally to verify inbound request signatures |
| `CHITON_GMAIL_TOKEN` | `chiton email gmail` | `gmail.googleapis.com` as `Authorization: Bearer …` |
| `CHITON_OUTLOOK_CLIENT_SECRET` (with `_TENANT_ID`, `_CLIENT_ID`, `_USER_ID`) | `chiton email outlook` (`msgraph` builds only) | `login.microsoftonline.com` to mint a token |
| `CHITON_TEAMS_CLIENT_SECRET` (with `_TENANT_ID`, `_CLIENT_ID`, `_USER_ID`, `_CLIENT_STATE`) | `chiton teams fetch`, `teams serve-webhook` (`msgraph` builds only) | `login.microsoftonline.com` to mint a token; `CLIENT_STATE` stays local and validates inbound notifications |

Handling properties worth verifying yourself:

- Each token/secret is sent **only to its own vendor's API**, as an
  `Authorization` header — never as a URL parameter, never to a third party.
- No secret is written to stdout, stderr, or any log. Error paths name the
  **variable** ("an OAuth token is required (set `CHITON_SLACK_TOKEN`)"), never
  its value.
- The passphrase is read with `term.ReadPassword` (no echo) and, when
  `CHITON_PASSPHRASE` is set, straight from the environment.

The residual risks are the ordinary ones for environment-variable secrets: they
are visible to anything that can read the process environment, they land in
shell history if you type `export` inline, and `AZURE_AUTHORITY_HOST` (read by
the Azure SDK, not by chiton) can redirect where the Microsoft secrets go.
chiton does not use an OS keychain.

---

## 5. Network listeners and the MCP surface

Three commands can open a listening socket. None of them authenticates the
caller by identity.

| Command | Default bind | Auth |
|---|---|---|
| `chiton serve` **with no flags** | `127.0.0.1:7474`, HTTP + SSE | **None** |
| `chiton serve --stdio` | *no socket at all* | n/a |
| `chiton serve --unix <path>` | filesystem socket (Linux/macOS) | filesystem permissions only |
| `chiton teams serve-webhook` | `127.0.0.1:8444` | Graph `clientState` shared secret |
| `chiton slack serve-events` | `127.0.0.1:8445` | Slack request-signature verification |

### 5.1 `chiton serve`

**A bare `chiton serve` with no flags starts the HTTP/SSE transport** — `--http`
is not opt-in, it is the default, and `--stdio` is what you must ask for. The
assistant installers and the plugin all pass `--stdio`, so this matters mainly
when you start a daemon by hand.

The HTTP/SSE transport has **no authentication, no TLS, and no origin check**.
Anything that can reach the port can read the entire graph — code, findings,
meetings, emails, browser history, PII. It binds loopback by default and prints
a loud stderr warning when asked to bind a wildcard address (`0.0.0.0`, `::`, or
an empty host), but it will not stop you. Do not expose it, and be aware that a
loopback bind is still reachable by every local process and, in a browser
context, by any page that can talk to `127.0.0.1`.

### 5.2 The two receivers

- **`chiton slack serve-events`** verifies Slack's request signature using
  `CHITON_SLACK_SIGNING_SECRET` and **refuses to start without it** rather than
  running a server that rejects everything. Each verified event triggers an
  outbound Slack API call to re-ingest the channel or thread (§1.1).
- **`chiton teams serve-webhook`** now **refuses to start without
  `CHITON_TEAMS_CLIENT_STATE`.** It previously failed open and accepted every
  notification; that is fixed.
- Both previously defaulted to `:8444` / `:8445`, which binds **all**
  interfaces. They now default to `127.0.0.1` and warn on a wildcard bind. Both
  are designed to sit behind a tunnel or reverse proxy that terminates TLS.

### 5.3 `add_url` — assistant-driven fetching

On a `--writable` server, `add_url` lets the assistant make chiton fetch **any
http/https URL**, and also lets it read **an arbitrary local file path or
`file://` URL** and put the contents in the graph. This is the widest capability
chiton exposes over MCP.

Shipped mitigations (`internal/ingest/url/`):

- The **read-only check runs before the fetch**, so a default (non-`--writable`)
  stdio server makes no request at all — it returns an error instead.
- `hostAllowed` rejects loopback, private, link-local, and unspecified addresses
  — both as literals *and* after DNS resolution — blocking SSRF at cloud
  metadata endpoints such as `169.254.169.254`.
- `CheckRedirect` re-applies `hostAllowed` on **every** hop, with a 10-redirect
  cap, so a public host cannot redirect into the internal network.
- Responses are capped at 32 MiB and the client has a 30 s timeout.

Acknowledged residual gaps, stated plainly:

- **DNS-rebinding TOCTOU.** `hostAllowed` resolves the name, then the HTTP
  client resolves it again to dial. A determined attacker can race the two. The
  code documents this itself; closing it needs a custom `DialContext`, which is
  not implemented.
- **The `yt-dlp` branch bypasses `hostAllowed` entirely.** A YouTube-shaped host
  goes straight to the subprocess without the SSRF check.
- **Local file reads are not path-restricted.** `add_url` with a bare path or a
  `file://` URL will read anything the chiton process can read.

If you run `chiton serve --writable`, you are trusting the assistant with those
capabilities. The read-only default exists precisely so that you have to opt in.

---

## 6. Threat model

Adapted from `docs/PLAN.md` ("Security, Privacy & Threat Model"). Mitigation
status is labelled **shipped** or **planned**.

### In scope

| Threat | Mitigation | Status |
|---|---|---|
| Laptop theft / lost device reading the store off disk | AES-256-GCM store encryption keyed from an Argon2id passphrase | **Shipped** |
| Backup leakage (`chiton-out/` in cloud/disk backups) | Same store encryption | **Shipped** |
| Stolen `chiton-out/graph.kuzu/` yields readable data | Same store encryption | **Shipped** |
| Plaintext sidecars / exports leaking the same data | — | **Not mitigated:** `cluster.json`, `cache.json`, `.chiton/baselines/`, `.chiton/llm-analyzed.json`, and `chiton export` output are plaintext |
| Meeting/email PII surfaced by the two meeting tools | `person` nodes tagged `pii=true`, hidden unless `include_pii` | **Shipped, but narrow** — see §3; the other read tools have no gate |
| Accidental upload of secrets to a remote LLM | Subagent mode = no built-in LLM calls; per-source allowlist + secret scrubbing | **Partly:** no built-in LLM egress (shipped); allowlist/scrubbing **planned** |
| SSRF via assistant-driven `add_url` | Read-only default; `hostAllowed` on the host and on every redirect; 10-redirect cap | **Shipped, with known gaps** (§5.3) |
| Unauthenticated reader on the MCP HTTP port | Loopback default + wildcard-bind warning | **Partial:** there is no authentication at all (§5.1) |
| Unauthenticated Teams/Slack webhook notifications | `clientState` and signing-secret checks now required to start | **Shipped** (previously fail-open for Teams) |
| Secrets in environment variables | Env-only, header-only, never logged or persisted | **Shipped**; no OS keychain integration |
| Cross-project leakage (query in repo A pulling repo B nodes) | Query scoping defaults to current repo, `--scope all` gated | **Planned** |
| Malicious MCP client exfiltrating the whole graph | Per-tool token caps, per-session call caps, elicitation for broad scope | **Planned** |
| Prompt injection via ingested content | Structured tool schemas instead of free-form prompts; `input` treated as data | **Design principle:** the subagent task specs are schema-pinned and instruct treating `input` as data, never instructions. Slack messages, emails, transcripts, browser page titles, and fetched pages are all untrusted input |

### Out of scope (v1)

- **Multi-tenant isolation.** chiton is single-user, single-machine.
- **Network attackers on the MCP socket.** See §5.1 — loopback default, wildcard
  warning, and **no authentication whatsoever**. Do not expose it.
- **Malicious or substituted subprocesses.** The whisper CLI is auto-detected
  from `PATH`; `yt-dlp` and `git` are resolved the same way. chiton does not pin
  or verify them.
- **Live-memory attacks** against a running, unlocked process (cold-boot,
  `/proc/<pid>/mem` by a privileged process, debugger attach). The design
  mentions mlock'ing the key; **mlock is not implemented today.** A kernel-level
  attacker is the OS's problem, not chiton's.
- **EDR bypass, kernel escalation, supply-chain attacks** against the Go
  toolchain or dependencies. Standard hygiene only (pinned `go.mod`, checksum
  verification). `make release` writes a `dist/checksums.txt`, and the install
  scripts refuse to install a binary that is not listed in it.
- **Side-channel attacks** on the AES-NI implementation. chiton relies on the Go
  standard library's constant-time crypto.

### Resource & cost controls — planned

`docs/PLAN.md` specifies per-call / per-session / lifetime LLM budget limits,
disk and memory caps, and automatic fallback from daemon to subagent mode when a
budget is hit. **None of these are enforced in the current binary** — they are
unnecessary today precisely because chiton makes no LLM calls of its own. Treat
the budget knobs as roadmap, relevant only if/when a daemon LLM mode ships.

### Configuration that exists today

The shipped config (`internal/config/config.go`, `.chitonrc` TOML + `CHITON_*`
env overrides) covers the graph path, log level, embedding provider/model/
endpoint/dim, and the semantic confidence threshold. `CHITON_PROJECT_DIR`
anchors `.chitonrc` lookup and resolves a relative `graph_path` against that
directory instead of the process working directory (the Claude Code plugin sets
it); unset, behaviour is unchanged. The richer `config.yaml`/`sources.yaml`
policy surface described in the design is **not** parsed for enforcement yet.

---

## 7. Reporting a vulnerability

If you discover a security issue in chiton, please report it **privately** —
**do not** open a public issue.

- Preferred: open a private **GitHub Security Advisory** on the repository
  (Security → Advisories → Report a vulnerability), or
- Contact the maintainer directly.

Please include: affected version / commit, a description of the issue, and
reproduction steps or a proof of concept if available. We will acknowledge the
report and coordinate a fix and disclosure timeline with you.
