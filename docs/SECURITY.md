# chiton Security & Privacy

chiton reads source code, documentation, and meeting transcripts — any of which
can contain secrets, PII, customer data, or NDAed material. This document
describes how chiton handles that data, what is encrypted, what stays on the
device, and the threat model it is (and is not) designed to defend against.

Throughout, **shipped** means the behaviour is implemented in the current
codebase; **planned** means it is described in the design (`docs/PLAN.md`) but is
not yet wired into the binary. We are explicit about the distinction so you can
make an informed trust decision.

> **License note:** chiton is proprietary software (see `LICENSE`). This document
> describes its security posture; it is not a warranty.

---

## 1. Data-handling posture

chiton is a single, self-contained pure-Go binary that runs **on-device**. The
graph it builds — code symbols, doc sections, findings, meetings, decisions, and
embeddings — is stored locally under `.chiton/` next to the graph store.

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
| No telemetry / phone-home | Shipped (nothing in the binary contacts a remote endpoint on its own) |
| Embeddings can stay local | Shipped — `chiton embed` defaults to local Ollama (`http://localhost:11434`); `CHITON_EMBED_PROVIDER=hash` is fully offline |

The one outbound network path chiton *can* take today is the **embedding
provider**. By default it talks to a local Ollama instance on `localhost`. If you
reconfigure `embed.endpoint` / `CHITON_EMBED_ENDPOINT` to point at a remote
service, node text will be sent there — this is the only egress under your
control, and it is opt-in by configuration.

> **Planned, not shipped:** the design calls for a per-source allowlist
> (`sources.yaml` with `allow_remote_llm` / `allow_remote_embed`), a
> gitleaks-based secret-scrubbing pass before any remote call, and an
> `audit.jsonl` log of every remote call. `chiton init` writes a `sources.yaml`
> template, but **there is no enforcement code, no scrubbing pass, and no audit
> log in the current binary.** Treat those as roadmap items, not active controls.

---

## 2. Encryption at rest

### What is encrypted

The **BadgerDB graph store is encrypted at rest** when a keyfile is present. The
store is opened with Badger's native encryption (`WithEncryptionKey`, plus the
required non-zero `IndexCacheSize` of 64 MiB) — see
`internal/graph/store/badger/store.go` (`applyEncryption`). This covers the LSM
tree, value log, and all node/edge/embedding data Badger persists.

> **Scope caveat (shipped reality):** only the Badger store is encrypted today.
> The package doc in `internal/crypto/keyfile.go` describes AES-256-GCM wrapping
> for ad-hoc artifacts (`cache.json`, `cluster.json`, `audit.jsonl`) via
> `EncryptBytes`/`DecryptBytes` in `internal/crypto/file.go`, but those helpers
> are **not yet wired into the application** (they are exercised only by tests).
> Sidecar files such as the cluster results and content cache are written in
> plaintext. `chiton export` also writes a **plaintext JSON snapshot**
> (`internal/.../Snapshot`) — the design's "export archive is encrypted with the
> same passphrase" is **planned, not shipped.** Protect exported snapshots and
> sidecars yourself.

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
attached — by prompting interactively (no echo). In CI / non-interactive
environments you **must** set `CHITON_PASSPHRASE` or the command fails with a
clear error:

```sh
export CHITON_PASSPHRASE='…'   # read before any interactive prompt
chiton build .
```

Key resolution is automatic: when chiton opens a graph it looks for a keyfile in
the conventional `.chiton/` locations next to the store. If a keyfile is found,
it derives the key and opens the store encrypted; if none is found, the store
opens in plaintext (`internal/crypto/resolve.go`).

### Threats this addresses

Encryption at rest is aimed squarely at **data-at-rest exposure**:

- **Laptop theft / lost device:** an attacker with the powered-off disk cannot
  read graph contents without the passphrase.
- **Backup leakage:** cloud or full-disk backups (Time Machine, OneDrive,
  Backblaze) that include `.chiton/` remain unintelligible without the
  passphrase.
- **Stolen `.chiton/` directory:** copying the store directory off a machine or
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

## 3. PII handling for meetings

Meeting ingest creates `person` nodes for speakers. **These are tagged as PII**
at creation time — `internal/extract/meeting/graph.go` sets
`Attrs: {"pii": true}` on every `person` node — and a shared `person://` identity
URI keeps one person to one node across meetings.

PII is **hidden by default** in the meeting MCP tools and only surfaced when the
caller explicitly opts in:

- `search_meetings` returns each meeting with its decisions and linked code
  symbols, but **participant names are omitted** unless `include_pii: true` is
  passed; by default only a participant *count* is returned. Crucially, when
  `include_pii` is false, a query is **not** matched against the participant list
  — returning a meeting because a queried name attended would itself disclose
  attendance (`internal/mcp/meetings.go`, `matches` / `public`).
- The action-item tool likewise drops assignee names (also PII) unless
  `include_pii` is set.

Meeting **transcript content, titles, and decision summaries are always
searchable** — that is the point of the tool — but the personal identifiers
attached to them are gated.

> **Out of scope today:** chiton's meeting PII gating covers **speaker /
> assignee names** on `person` nodes. The design also mentions detecting email
> addresses and phone numbers; only the speaker-name `pii=true` tagging is
> confirmed shipped. Do not assume free-text PII inside transcript bodies is
> detected or redacted.

---

## 4. Threat model

Adapted from `docs/PLAN.md` ("Security, Privacy & Threat Model"). Mitigation
status is labelled **shipped** or **planned**.

### In scope

| Threat | Mitigation | Status |
|---|---|---|
| Laptop theft / lost device reading the store off disk | AES-256-GCM store encryption keyed from an Argon2id passphrase | **Shipped** |
| Backup leakage (`.chiton/` in cloud/disk backups) | Same store encryption | **Shipped** |
| Stolen `.chiton/` directory yields readable data | Same store encryption | **Shipped** |
| Meeting PII surfaced to colleagues with different access | `person` nodes tagged `pii=true`, hidden unless `include_pii` | **Shipped** |
| Accidental upload of secrets to a remote LLM | Subagent mode = no built-in LLM calls; per-source allowlist + secret scrubbing | **Partly:** no built-in LLM egress (shipped); allowlist/scrubbing **planned** |
| Cross-project leakage (query in repo A pulling repo B nodes) | Query scoping defaults to current repo, `--scope all` gated | **Planned** |
| Malicious MCP client exfiltrating the whole graph | Per-tool token caps, per-session call caps, elicitation for broad scope | **Planned** |
| Prompt injection via ingested content | Structured tool schemas instead of free-form prompts; `input` treated as data | **Design principle:** the subagent task specs are schema-pinned and instruct treating `input` as data, never instructions |

### Out of scope (v1)

- **Multi-tenant isolation.** chiton is single-user, single-machine.
- **Network attackers on the MCP socket.** `chiton serve` binds `--http` to
  **loopback by default (`127.0.0.1:7474`)** and prints a stderr warning when
  asked to bind a non-loopback address (`0.0.0.0`, `::`, or an empty host —
  all of which listen on every interface). The HTTP transport still has **no
  authentication**, so even with the warning, do not expose it on untrusted
  networks.
- **Live-memory attacks** against a running, unlocked process (cold-boot,
  `/proc/<pid>/mem` by a privileged process, debugger attach). The design
  mentions mlock'ing the key; **mlock is not implemented today.** A kernel-level
  attacker is the OS's problem, not chiton's.
- **EDR bypass, kernel escalation, supply-chain attacks** against the Go
  toolchain or dependencies. Standard hygiene only (pinned `go.mod`, checksum
  verification).
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
endpoint/dim, and the semantic confidence threshold. The richer
`config.yaml`/`sources.yaml` policy surface described in the design is **not**
parsed for enforcement yet.

---

## 5. Reporting a vulnerability

If you discover a security issue in chiton, please report it **privately** —
**do not** open a public issue.

- Preferred: open a private **GitHub Security Advisory** on the repository
  (Security → Advisories → Report a vulnerability), or
- Contact the maintainer directly.

Please include: affected version / commit, a description of the issue, and
reproduction steps or a proof of concept if available. We will acknowledge the
report and coordinate a fix and disclosure timeline with you.
