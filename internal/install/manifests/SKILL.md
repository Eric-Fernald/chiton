# chiton

Knowledge-graph context for this workspace. Use the MCP tools below before
falling back to grep or file reads for architecture, navigation, or findings
questions — they return typed graph relationships in far fewer tokens.

## When to use chiton

Call a chiton tool first when the question is about:

- "What does X depend on / call / import?"
- "Where is Y defined? Which file contains Z?"
- "How are A and B connected?"
- "What are the architectural hubs / god nodes here?"
- "What findings / rule violations exist in this module?"

Use grep or file reads when you need the actual source content (to make an
edit, locate a bug, or understand an implementation in detail).

## Commands

- `/chiton` — show status (graph stats, last build time)
- `/chiton build [path]` — build or refresh the graph for this workspace
- `/chiton query <label>` — look up a node by label substring
- `/chiton neighborhood <label> [depth]` — return a node and its 1–5 hop neighbors
- `/chiton path <from> <to>` — shortest path between two named nodes
- `/chiton godnodes` — list highest-betweenness nodes (architectural hubs)
- `/chiton summary [scope]` — token-budgeted overview of a module or the repo
- `/chiton findings [filters]` — list rule findings; filter by severity, category, or scope
- `/chiton explain <finding-id>` — finding + rule + related findings on same target

## How to call the underlying tools

| Command | MCP tool | Required args |
|---|---|---|
| `/chiton query` | `query_graph` | `query` |
| `/chiton neighborhood` | `get_node_neighborhood` | `label` |
| `/chiton path` | `shortest_path` | `from`, `to` |
| `/chiton godnodes` | `god_nodes` | (optional `limit`, `community_id`) |
| `/chiton summary` | `graph_summary` | (optional `scope`, `max_tokens`) |
| `/chiton findings` | `list_findings` | (optional `category`, `min_severity`, `scope`) |
| `/chiton explain` | `explain_finding` | `id` |

## Response style

- Cite every claim with the `source_uri` and line span from the returned node.
- Prefer the lean JSON in the tool result; do not paraphrase fields that the
  user can read themselves.
- If the graph is empty or stale, suggest running `chiton build .` first.

## Fallback if the MCP server is unavailable

Shell out to the CLI: `chiton query <label>`, `chiton findings list`,
`chiton report`. Same outputs, slower path, but no requirement that the
assistant has MCP wired up.
