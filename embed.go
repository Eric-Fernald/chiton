// Package chiton embeds the assistant-facing skill files that ship with this
// repository.
//
// The files under skills/ are the single source of truth for what chiton tells
// an assistant. They are consumed two ways, and both read the same bytes:
//
//   - As a Claude Code plugin, where skills/<name>/SKILL.md is loaded directly
//     from the plugin directory.
//   - By `chiton install`, which writes the same markdown into each assistant's
//     own skills directory for people who wire chiton up without the plugin.
//
// Embedding them here — rather than keeping a second copy next to the installer —
// makes it impossible for the two distribution paths to drift apart.
package chiton

import _ "embed"

// GraphSkillMarkdown is the graph-query skill: when to reach for chiton's MCP
// tools instead of grep, which tool backs which question, and how to read
// provenance. Published as /chiton:graph by the plugin.
//
//go:embed skills/graph/SKILL.md
var GraphSkillMarkdown string

// SubagentTasksSkillMarkdown is the subagent task-flow skill: semantic
// extraction, LLM rule analysis, and meeting→code linkage, all of which run
// through the host assistant's subagents rather than an API key of chiton's own.
// Published as /chiton:subagent-tasks by the plugin.
//
//go:embed skills/subagent-tasks/SKILL.md
var SubagentTasksSkillMarkdown string

// SetupSkillMarkdown is the first-run skill: locate or install the binary,
// build the graph, and verify the MCP server is answering. Published as
// /chiton:setup by the plugin. It is deliberately not written by
// `chiton install` — anyone running that command already has the binary.
//
//go:embed skills/setup/SKILL.md
var SetupSkillMarkdown string
