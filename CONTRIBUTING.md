# Contributing to chiton

This guide covers how to build, test, and extend the project. Read
[Contribution status](#contribution-status) first. This document grants no
license.

## Contribution status

chiton is licensed under the PolyForm Strict License 1.0.0 with additional
permissions (see [LICENSE](LICENSE)). Anyone may use it for any purpose and
install or build it by the documented routes, but the license does not permit
making changes or new works based on it, or distributing it beyond the plugin
marketplaces and mirrors it names.

Bug reports and feature requests are welcome as
[issues](https://github.com/Eric-Fernald/chiton/issues). Pull requests are not
accepted: a change to chiton is a new work the license does not permit, and
there is no contributor agreement through which outside code could be taken in.
[NOTICE.md](NOTICE.md) records project provenance, and is where any change to
contributor status will be recorded.

What follows is the working guide for the maintainer.

## Prerequisites

**Go.** `go.mod` declares `go 1.25.5` and pins `toolchain go1.26.4`, so any Go
new enough to honour the toolchain directive fetches and uses 1.26.4
automatically. If you have set `GOTOOLCHAIN=local`, install Go 1.26.4 yourself.

**No C toolchain is required.** The default build is pure Go and cross-compiles
to Windows, macOS, and Linux without CGo. Only the optional `whisper_cgo` tag
needs a C compiler.

**`make` is optional.** The Makefile is written for a POSIX shell (`mkdir -p`,
`command -v`, `[ -n ... ]`), so on Windows run it from Git Bash with GNU make,
or skip it and run the underlying `go` commands listed beside each target below.

Optional, per feature:

| Tool | Needed for | Fallback |
|---|---|---|
| [Ollama](https://ollama.com) on `localhost:11434` | `chiton embed` / `chiton search`. The default provider is `ollama` with model `nomic-embed-text` | `CHITON_EMBED_PROVIDER=hash` is fully offline, and is what CI should use |
| A whisper CLI on `PATH` (`whisper`, `whisper-cli`, `whisper-cpp`, `whisper.cpp`) | `chiton meeting transcribe` | `--whisper-bin <path>`, or build with `-tags whisper_cgo` for the in-process backend |
| `yt-dlp` on `PATH` | YouTube URLs passed to `chiton add-url` | none, since other URL kinds (HTML, PDF) need nothing extra |

## Build and test

```sh
make build            # go build -ldflags "-s -w -X main.version=..." -o chiton ./cmd/chiton
make check            # the pre-merge gate: fmt-check + vet + test
make race             # go test -race ./...
make test             # go test ./...
make check-msgraph    # vet + test the tag-gated Microsoft Graph path
```

Without `make`:

```sh
go build -o chiton ./cmd/chiton
gofmt -l .            # must print nothing
go vet ./...
go test ./...
```

`make check` is the gate to run before merging anything. `make race` is worth
running when you touch the watcher, the job queue, the MCP server, or anything
else that spawns goroutines. There are no `testdata` directories. Tests build
their inputs inline or in `t.TempDir()`, so the suite runs anywhere.

### Build tags

Two optional tags exist. Neither is part of the default build, and neither is
part of any release artifact.

**`msgraph`** enables Microsoft Graph support: `chiton teams fetch`,
`chiton teams serve-webhook`, and `chiton email outlook`. The Graph SDK links
request builders for every top-level Graph resource, and kiota registers its
serialization types through interface tables, so the linker cannot strip the
parts chiton never calls. That costs roughly 117 MB of binary for two API calls,
which is why the default build excludes it:

```sh
make build-msgraph    # go build -tags msgraph -o chiton ./cmd/chiton
```

Without the tag those commands still appear in `--help` and still parse their
flags; they fail at runtime with a message telling you to rebuild. The stubs
(`internal/ingest/teams/graph_stub.go`, `internal/ingest/email/outlook_stub.go`)
mirror the real types exactly and assert the same interfaces, so both build
configurations compile identically from the outside.

**Run `make check-msgraph` whenever you touch anything under
`internal/ingest/teams/` or `internal/ingest/email/`.** The default `make check`
never compiles the tagged files, so otherwise they break without anyone noticing. A test whose
subject takes a Graph type belongs in a `//go:build msgraph` file (see
`internal/ingest/email/outlook_msgraph_test.go`), not in the default test
binary, or it drags the whole SDK into a test binary the product binary does not
link.

`chiton email ingest` (local `.eml` / `.mbox` / maildir) and `chiton email gmail`
are pure Go and are unaffected by the tag.

**`whisper_cgo`** enables in-process whisper.cpp transcription. Needs a C toolchain.
Without it, `chiton meeting transcribe` shells out to a whisper CLI on `PATH`.
There is no `make` target; build it with
`go build -tags whisper_cgo ./cmd/chiton`.

## Repository layout

```
chiton/
├── cmd/chiton/        thin cobra wrapper — flag parsing and wiring, no domain logic
├── internal/
│   ├── extract/       per-language AST extractors, Markdown, images, subagent task specs
│   ├── graph/         model, GraphStore interface + BadgerDB impl, clustering, metrics, retrieval
│   ├── ingest/        incremental updater, fsnotify, git, and the external sources
│   │                    (meeting, url, audio, browser, email, slack, teams)
│   ├── analyze/       rule engine, findings, diff/baselines, LLM-rule dispatch
│   │   └── rules/     the rules themselves, one package per category
│   ├── mcp/           MCP server and tool handlers (stdio / SSE / unix)
│   ├── install/       per-assistant installers + the shared markdown manifests
│   ├── embed/ vector/ embedding providers and the flat-cosine index
│   └── crypto/ cache/ config/ hooks/ jobs/ report/ bench/
├── skills/            assistant-facing skill markdown — single source of truth
├── .claude-plugin/    plugin.json + marketplace.json
├── bin/ scripts/      PATH shim and the prebuilt-binary installers
├── embed.go           //go:embed of skills/*/SKILL.md
└── docs/              PLAN.md, INSTALL.md, MCP.md, SKILL.md, SECURITY.md, BENCHMARK.md
```

Keep `cmd/chiton` thin: it parses flags, opens a store, and calls into
`internal/`. Depend on the `GraphStore` interface, not on the BadgerDB
implementation behind it.

### `skills/` is shared by both distribution paths

`skills/graph/SKILL.md`, `skills/subagent-tasks/SKILL.md`, and
`skills/setup/SKILL.md` are the only copies of that markdown. Root-level
`embed.go` embeds them into the `chiton` package and
`internal/install/manifests` re-exports them, so one edit changes **both**
distribution paths at once:

- the Claude Code plugin, which loads `skills/<name>/SKILL.md` directly and
  publishes them as `/chiton:graph`, `/chiton:subagent-tasks`, and
  `/chiton:setup`; and
- `chiton install`, which writes `.claude/skills/chiton/SKILL.md` and
  `.claude/skills/chiton-subagent-tasks/SKILL.md` from the embedded bytes.

Two consequences. First, the two paths cannot drift apart. Do not add a second
copy of a skill next to the installer. Second, the frontmatter carries a
`description:` but deliberately **no `name:`**: in both layouts the directory
name supplies the skill's name, so hard-coding one would break the other layout.
After editing a skill, rebuild. The content is compiled in, and a stale binary
keeps writing the old text.

`chiton install` writes only the graph and subagent-tasks skills. `setup` is
plugin-only, on the reasoning that anyone running `chiton install` already has a
binary.

## Adding a deterministic rule

Rules live in `internal/analyze/rules/<category>/`, one file per rule, where
`<category>` is one of the eight fixed buckets: `architecture`, `security`,
`bugs`, `performance`, `reliability`, `database`, `quality`, `style`. There are
75 deterministic rules and 24 LLM rules today.
`internal/analyze/rules/architecture/dot_import.go` is a short, complete model to
copy.

1. **Declare a type and its `Meta()`.** A rule is a struct, usually empty,
   whose `Meta() analyze.Meta` returns its registration data. Every field is
   required except `Languages` (nil means every language) and `Fix`:

   ```go
   func (DotImport) Meta() analyze.Meta {
       return analyze.Meta{
           ID:         "architecture/dot-import",  // "<category>/<kebab-name>", unique
           Category:   analyze.CategoryArchitecture,
           Severity:   analyze.SeverityMedium,     // info | low | medium | high | critical
           Scope:      analyze.FileScope,          // or analyze.GraphScope
           Languages:  []string{"go"},             // file extensions, without the dot
           Summary:    "...",                      // required; the registry test enforces it
           Fix:        "...",                      // optional hint
           Provenance: analyze.ProvenanceDeterministic,
       }
   }
   ```

2. **Implement the matcher for your scope.** `FileScope` rules implement
   `MatchFile(ctx, analyze.FileTarget) []analyze.Finding` and run once per
   matching file; the target carries `Path`, `URI`, raw `Content`, the `FileNode`
   already in the graph, and the symbols and edges belonging to that file.
   `GraphScope` rules implement `MatchGraph(ctx, analyze.GraphTarget)` and run
   once over the whole extracted graph. Fill each `Finding` in with `RuleID`,
   `Severity`, `Category`, `Message`, `TargetURI`, `TargetNodeID`, and a `Span`
   when you know the line.

3. **Register it in `init()`.** `func init() { analyze.Register(DotImport{}) }`.
   `Register` panics on an empty or duplicate ID, so a collision fails loudly at
   startup instead of silently shadowing an existing rule.

4. **Only if you added a new category package**, add a blank import to
   `internal/analyze/rules/rules.go`. That barrel package is what every caller
   (CLI, MCP server, tests) imports to trigger the `init()` registrations, so a
   package missing from it registers nothing at all.

5. **Write the tests, including the false-positive cases.** Tests live in the
   same package as the rule. `FileScope` tests build a target from raw source
   with the `fileTarget(path, content)` helper (see
   `internal/analyze/rules/architecture/newrules_test.go`); `GraphScope` tests
   build nodes and edges with the `file()` / `sym()` / `edge()` helpers (see
   `architecture_test.go`) and assert the cycle or hub the rule should find.

   The house pattern is **one positive test plus at least one negative test that
   encodes a real false positive**, meaning a case that naively matches but must
   not fire. Existing examples: an `import "fake"` inside a comment must not count
   toward the import threshold; a package flagged `external: true` must not be
   reported as a circular import; a threshold rule gets both an over- and an
   under-threshold case. Comment-induced false positives are common, so reuse the
   existing comment strippers and literal maskers instead of writing new ones.
   `internal/analyze/lex.go` has `StripLineComment`, `StripPyComment`, `MaskCLiterals`,
   and `MaskGoLiterals`. Call `MaskedCLiterals` or `MaskedGoLiterals` to get a mask that
   every rule shares for the file. `stripArchLineComment` in
   `internal/analyze/rules/architecture/too_many_imports.go` also handles line comments.

   Precision matters more than recall here: re-analysing a file prunes findings
   that no longer fire, so a noisy rule does not add noise once, it churns the
   graph on every sync.

6. **`internal/analyze/rules/registry_test.go` is the backstop.** It asserts
   that no two rules share an ID, that every rule has a non-empty `Severity` and
   `Summary`, and that all eight categories still have at least one rule. It runs
   as part of `make check`.

LLM rules are different: they are data, not Go logic. Add one to the `rules`
slice in `internal/analyze/rules/llm/llm.go` using the `def()` helper. Their
`Meta` sets `Provenance: analyze.ProvenanceLLM` and `Scope: analyze.FileScope`,
they implement `Prompt()` instead of a matcher, and the prompt must describe what
to look for **without** naming a file or embedding input. The task payload
supplies the code under inspection separately, and the findings come back tagged
`INFERRED` (or `AMBIGUOUS` below the confidence threshold).

## Line endings and file modes

The repository is **LF-only**. `.gitattributes` sets `* text=auto eol=lf`, with
deliberate exceptions for `*.cmd`, `*.bat`, and `*.ps1` (CRLF, because Windows
shells want it) and explicit LF pins for `bin/chiton` and `scripts/*.sh`.

If you work on Windows:

- Do not commit CRLF into anything that is not one of those three extensions.
  Leave `git config core.autocrlf` alone. `.gitattributes` already normalises on
  commit, and the setting only conflicts with it.
- `gofmt -l .` **must print nothing.** `make fmt` (`gofmt -w .`) fixes it, and
  `make fmt-check` is what fails the build.

**The executable bit matters.** `bin/chiton` (the POSIX shim placed on the
plugin's PATH) and `scripts/install-chiton.sh` must stay executable, and Windows
does not record that bit. When adding a file that needs it, set it explicitly:

```sh
git update-index --chmod=+x bin/chiton
git update-index --chmod=+x scripts/install-chiton.sh
```

Verify with `git ls-files -s <path>`: mode `100755`, not `100644`. A shim that
lands as `100644` fails silently for every POSIX user.

## Plugin development

The repository root doubles as a Claude Code plugin for development:
`.claude-plugin/plugin.json` declares it, and `skills/` plus the root `.mcp.json`
are its components. What users install is different: a release build of it,
pinned to that release, that the release workflow commits to the `plugin-dist`
branch. `.claude-plugin/marketplace.json` points at that branch.

```sh
claude --plugin-dir .                   # load the plugin from this working tree
make plugin-validate                    # strict-validate a clean clone of HEAD
make plugin-dist VERSION=v1.2.3         # build dist/plugin/ as a release would
claude plugin validate dist/plugin --strict
claude --plugin-dir dist/plugin         # try the release build
```

`make plugin-validate` validates a fresh clone because a working tree can hold
untracked files the published plugin never contains; a local `CLAUDE.md` at the
root, for one, raises a warning that `--strict` turns into a failure.
`claude plugin validate` on a directory containing **both** manifests resolves to
the marketplace manifest. Both manifests and the assembled `dist/plugin/` pass
strict validation today, and they should stay that way.

`scripts/build-plugin-dist.sh` decides what the published plugin contains. Its
`.mcp.json` comes from `packaging/plugin/.mcp.json` and launches
`${CLAUDE_PLUGIN_ROOT}/bin/chiton`: Claude Code on Windows resolves that
extensionless path to `bin/chiton.cmd`, and Linux and macOS run the `bin/chiton`
sh launcher. The tree carries no binaries. It carries `release/tag` and
`release/checksums.txt` instead, and on first use the launcher runs
`scripts/install-chiton.{sh,ps1}` to download that tag's binary for the machine,
verify it against the shipped checksums, and cache it in
`~/.chiton/versions/<tag>/`. Every message on that path must go to stderr,
because stdout is the MCP protocol stream.

Trying `dist/plugin/` locally therefore needs the tag in
`dist/plugin/release/tag` to be published on GitHub: the launcher downloads
from the real release, not from `dist/`. To test unreleased binaries, point
`CHITON_BIN` at one, or pre-seed `~/.chiton/versions/<tag>/` with it.

## Working on chiton itself

**The root `.mcp.json` is shared, not a personal dev config.** It launches the
server by bare name:

```json
{ "command": "chiton", "args": ["serve", "--stdio"],
  "env": { "CHITON_PROJECT_DIR": "${CLAUDE_PROJECT_DIR}" } }
```

That suits a source checkout loaded with `--plugin-dir .`: it requires `chiton`
on `PATH`, and lets `${CLAUDE_PROJECT_DIR}` anchor the server to the workspace.
It is wrong for development, where you want *your* freshly built binary rather
than whichever one is on `PATH`. Do not edit it to point at a local path; that
change would be committed for everyone.

Register a dev server at local scope instead. It writes to your own settings and
never to the repository:

```sh
make build      # or: go build -o chiton ./cmd/chiton
claude mcp add chiton-dev -s local -- "$PWD/chiton" serve --stdio
```

`-s local` is the default scope and can be omitted. Keep the `--` so that
chiton's own flags are not consumed by `claude mcp add`.

`CHITON_PROJECT_DIR` is honoured by `internal/config`: when it is set, `.chitonrc`
is looked up in that directory and a relative `graph_path` is resolved against it
rather than against the process working directory. Unset, behaviour is
unchanged, and a relative default stays relative. It exists because
assistant-launched MCP servers do not reliably start in the workspace root, and a
server that starts elsewhere would read a different store than `chiton build`
wrote. Pass it explicitly (`-e CHITON_PROJECT_DIR=/path/to/repo`) if a dev server
ever seems to be answering from an empty graph.

The HTTP transport requires a per-start bearer token and refuses browser origins that are not allowlisted. `chiton serve --http` and the webhook
receivers all bind loopback by default and print a warning when pointed at a
wildcard address. Keep it that way, and read [SECURITY.md](SECURITY.md) and
[docs/SECURITY.md](docs/SECURITY.md) before changing anything in that area.

## Documentation

Docs are part of the change, not a follow-up. If you add a CLI command, an MCP
tool, a node kind, or a rule category, update `README.md`, `docs/MCP.md`, and the
relevant file under `skills/` in the same change. Record user-visible changes in
[CHANGELOG.md](CHANGELOG.md) under `## [Unreleased]`.

