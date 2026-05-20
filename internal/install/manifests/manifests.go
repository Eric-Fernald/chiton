// Package manifests carries the markdown templates shared across installers.
// Files in this directory are embedded via go:embed so the chiton binary is
// the single source of truth for skill content.
package manifests

import _ "embed"

// SkillMarkdown is the canonical /chiton skill reference, written into each
// assistant's skills directory by the per-platform installer.
//
//go:embed SKILL.md
var SkillMarkdown string

// RulesPrompt is the short always-on instruction appended to rules files
// (CLAUDE.md, AGENTS.md, .github/copilot-instructions.md, etc.). It tells the
// assistant when to call chiton instead of grepping. Kept short to limit the
// per-turn token cost.
const RulesPrompt = `# chiton — knowledge graph context

This workspace has a chiton MCP server registered. Before using grep or file
reads to answer architecture or navigation questions, call the chiton tools:

- ` + "`get_node_neighborhood`" + ` — "what does X call / depend on?", "show X and its neighbors"
- ` + "`query_graph`" + ` — "what symbols exist named like Y?"
- ` + "`shortest_path`" + ` — "how are A and B connected?"
- ` + "`graph_summary`" + ` — token-budgeted overview of a module or the repo
- ` + "`god_nodes`" + ` — architectural hubs ranked by betweenness centrality
- ` + "`list_findings`" + ` / ` + "`explain_finding`" + ` — rule-engine results

Still use grep or file reads when you need actual source content (to make an
edit, locate a bug, or understand an implementation). Chiton is for "what
exists" and "how things are connected," not for reading code.

Build the graph with ` + "`chiton build .`" + ` and run ` + "`chiton analyze .`" + ` for findings.
`
