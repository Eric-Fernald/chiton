# The `/chiton` skill

`/chiton` is a slash-command skill that gives an AI assistant a small, fixed
vocabulary of commands for querying the chiton knowledge graph. Each command
maps to a chiton MCP tool (with a `chiton` CLI fallback when the MCP server is
unavailable), so the assistant answers architecture and navigation questions
from typed graph relationships instead of grepping the tree.

The canonical skill content is embedded in the chiton binary
(`internal/install/manifests/SKILL.md`, exposed as `manifests.SkillMarkdown`)
and is the single source of truth. This document mirrors that content.

## Installation

The skill is written to disk by `chiton install`. The Claude Code installer
writes the embedded skill markdown to:

```
.claude/skills/chiton/SKILL.md
```

`chiton uninstall` removes that directory, and `chiton doctor` reports whether
the skill is installed. See [INSTALL.md](INSTALL.md) for the full install flow
and per-assistant layout.

> **Scope note:** Among the installers in this package, only the Claude Code
> installer writes `SkillMarkdown` (the `/chiton` skill). The other assistant
> installers (Copilot, Cursor, Codex, Gemini CLI, OpenCode, Aider, Antigravity,
> Copilot CLI) receive only the shorter always-on `RulesPrompt` appended to
> their rules file — they do not get the slash-command skill. See "Rules
> prompt" below.

## Commands

The skill exposes the following `/chiton` subcommands (verbatim from the
manifest):

| Command | What it does |
|---|---|
| `/chiton` | Show status — graph stats, last build time. |
| `/chiton build [path]` | Build or refresh the graph for this workspace. |
| `/chiton query <label>` | Look up a node by label substring. |
| `/chiton neighborhood <label> [depth]` | Return a node and its 1–5 hop neighbors. |
| `/chiton path <from> <to>` | Shortest path between two named nodes. |
| `/chiton godnodes` | List highest-betweenness nodes (architectural hubs). |
| `/chiton summary [scope]` | Token-budgeted overview of a module or the repo. |
| `/chiton findings [filters]` | List rule findings; filter by severity, category, or scope. |
| `/chiton explain <finding-id>` | Finding + rule + related findings on the same target. |

## Command → MCP tool → CLI fallback

The skill maps each command to an MCP tool. When the MCP server is unavailable,
the assistant shells out to the equivalent `chiton` CLI subcommand. The
tool/argument mappings below are taken directly from the manifest's "How to
call the underlying tools" table.

| Command | MCP tool | Required / optional args | CLI fallback |
|---|---|---|---|
| `/chiton` | _(status)_ | — | — |
| `/chiton build` | _(build)_ | `path` (optional) | `chiton build .` |
| `/chiton query` | `query_graph` | `query` | `chiton query <label>` |
| `/chiton neighborhood` | `get_node_neighborhood` | `label` (depth optional) | — |
| `/chiton path` | `shortest_path` | `from`, `to` | — |
| `/chiton godnodes` | `god_nodes` | optional `limit`, `community_id` | — |
| `/chiton summary` | `graph_summary` | optional `scope`, `max_tokens` | — |
| `/chiton findings` | `list_findings` | optional `category`, `min_severity`, `scope` | `chiton findings list` |
| `/chiton explain` | `explain_finding` | `id` | — |

Notes:

- `/chiton` (status) and `/chiton build` are listed in the manifest's command
  list but not in its MCP-tool mapping table; the manifest does not pin them to
  a specific MCP tool. `/chiton build` corresponds to the `chiton build [path]`
  CLI subcommand.
- The manifest's CLI fallback section names only `chiton query <label>`,
  `chiton findings list`, and `chiton report`: "Shell out to the CLI … Same
  outputs, slower path, but no requirement that the assistant has MCP wired up."
  The dashes above mark commands for which the manifest gives no explicit CLI
  fallback. (Full underlying subcommands are listed in the README CLI reference.)

## Per-assistant invocation differences

The embedded skill content uses the `/chiton` prefix throughout and does **not**
describe any alternate invocation syntax (e.g. `$chiton`) or per-assistant
differences. There is nothing in the manifest to document on that front.

## Response style

The manifest instructs the assistant to:

- Cite every claim with the `source_uri` and line span from the returned node.
- Prefer the lean JSON in the tool result; do not paraphrase fields the user can
  read themselves.
- If the graph is empty or stale, suggest running `chiton build .` first.

## When to use chiton (vs. grep)

Call a `/chiton` command first when the question is about:

- "What does X depend on / call / import?"
- "Where is Y defined? Which file contains Z?"
- "How are A and B connected?"
- "What are the architectural hubs / god nodes here?"
- "What findings / rule violations exist in this module?"

Use grep or file reads when you need the actual source content — to make an
edit, locate a bug, or understand an implementation in detail.

## Fallback if the MCP server is unavailable

Per the manifest, shell out to the CLI:

```sh
chiton query <label>
chiton findings list
chiton report
```

Same outputs, slower path, but no requirement that the assistant has MCP wired
up. For the complete set of underlying `chiton` subcommands these commands wrap,
see the CLI reference in [README.md](../README.md).

## Rules prompt (always-on, all assistants)

Independent of the slash-command skill, every assistant the installer touches
also gets a short always-on instruction (`manifests.RulesPrompt`) appended to
its rules file (`CLAUDE.md`, `AGENTS.md`, `.github/copilot-instructions.md`,
etc.). It tells the assistant to prefer chiton's MCP tools over grep for
architecture/navigation questions and references these tools directly:
`get_node_neighborhood`, `query_graph`, `shortest_path`, `graph_summary`,
`god_nodes`, and `list_findings` / `explain_finding`. It closes with: build the
graph with `chiton build .` and run `chiton analyze .` for findings.
