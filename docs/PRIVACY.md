# chiton Privacy Policy

*Effective 2026-10-06. Applies to the chiton binary and the chiton Claude Code
plugin.*

chiton runs entirely on your machine. The author operates no server, collects
no data, and receives nothing from your installation. This page summarises what
chiton does with the data it touches; [SECURITY.md](SECURITY.md) is the
detailed, code-referenced version and wins where the two differ.

## What chiton collects about you

Nothing. There is no telemetry, analytics, crash reporting, account, licence
check, or update check. `chiton version` is offline. No chiton-operated
endpoint exists for data to be sent to.

## What chiton reads

By default, only the files in the workspace you point it at: source code,
Markdown and images, which `chiton build` turns into a local knowledge graph.

Everything else is opt-in and happens only when you run the command for it:

| Source | Command | What is read |
|---|---|---|
| Meeting transcripts / audio | `chiton meeting ingest`, `chiton meeting transcribe` | The files you name |
| Slack | `chiton slack fetch`, `chiton slack serve-events` | The channels/threads you name, with your token |
| Email | `chiton email ingest`, `chiton email gmail`, `chiton email outlook` | The mailbox files or the query you give, with your token |
| Browser history | `chiton browser ingest`, `chiton browser watch` | Page title, URL, visit count and last-visit time from the local History database. Query strings and fragments are dropped unless you pass `--keep-query`. No page is fetched |
| Web pages | `chiton add-url`, or the `add_url` MCP tool on a server you started with `--writable` | The URL you name |

The Claude Code plugin starts a **read-only** MCP server; it cannot ingest
anything on its own.

## Where data is stored

In a local store under `chiton-out/` in your workspace, encrypted at rest
(AES-256, passphrase-derived key) when you run `chiton init` with its default
settings. Snapshots written by `chiton export` are plaintext. Names of meeting speakers, email participants and Slack users are
tagged as personal data and hidden from assistant queries unless you opt in per
call. Deleting `chiton-out/` and `.chiton/` removes everything chiton stored.

## What leaves your machine

Only traffic you trigger, to destinations you choose:

- **Embeddings.** `chiton embed` and the `semantic_search` tool send text to the
  embedding provider, which is a local Ollama server
  (`http://localhost:11434`) by default. Pointing `CHITON_EMBED_ENDPOINT` at a
  remote host sends node text and queries there. `CHITON_EMBED_PROVIDER=hash`
  sends nothing.
- **Ingest commands** contact the service they name (Slack, Gmail, Microsoft
  Graph, the URL you pass, YouTube through `yt-dlp`) using credentials you
  supply.
- **Your AI assistant.** Results of chiton's MCP tools are returned to the
  assistant that called them, and from there are subject to that assistant's
  own privacy policy. chiton itself calls no LLM.
- **HTML reports** from `chiton dashboard` / `chiton visualize` load a script
  from `unpkg.com` when you open them in a browser.

## Children

chiton is a developer tool and is not directed at children.

## Changes and contact

Changes to this policy are made in this file and are visible in the
repository's history. Questions or concerns: open an issue at
<https://github.com/Eric-Fernald/chiton/issues>, or follow
[SECURITY.md](../SECURITY.md) for anything security-sensitive.
