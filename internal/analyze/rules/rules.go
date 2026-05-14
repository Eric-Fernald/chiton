// Package rules is a barrel import: importing it brings in every rule
// sub-package, which registers its rules with the default registry via init().
//
// Anywhere that needs rule discovery (CLI commands, MCP server, tests) should
// blank-import this package once at startup.
package rules

import (
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/architecture"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/bugs"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/quality"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/security"
)
