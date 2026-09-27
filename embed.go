// Package chiton embeds skills/, the single source of truth for what chiton tells an assistant. The plugin
// loads those files directly and `chiton install` writes the same bytes, so the two always match.
package chiton

import _ "embed"

// GraphSkillMarkdown is the graph-query skill, published as /chiton:graph by the plugin.
//
//go:embed skills/graph/SKILL.md
var GraphSkillMarkdown string

// SubagentTasksSkillMarkdown is the subagent task-flow skill, published as /chiton:subagent-tasks.
//
//go:embed skills/subagent-tasks/SKILL.md
var SubagentTasksSkillMarkdown string

// SetupSkillMarkdown is the first-run skill (/chiton:setup). `chiton install` never writes it.
//
//go:embed skills/setup/SKILL.md
var SetupSkillMarkdown string
