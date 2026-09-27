// Package rules is a barrel import: blank-import it once wherever rule discovery is needed, and
// every rule sub-package registers its rules with the default registry via init().
package rules

import (
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/architecture"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/bugs"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/database"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/llm"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/performance"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/quality"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/reliability"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/security"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules/style"
)
