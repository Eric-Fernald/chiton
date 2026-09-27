package security

import (
	"bytes"
	"context"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
)

// WeakRandom flags non-test Go files importing math/rand instead of crypto/rand; tests are skipped
// because they legitimately want determinism.
type WeakRandom struct{}

func (WeakRandom) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/weak-random",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go"},
		Summary:    "math/rand is not cryptographically secure.",
		Fix:        "Use crypto/rand for tokens, IDs, and other security-sensitive values.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (WeakRandom) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	if strings.HasSuffix(t.Path, "_test.go") {
		return nil
	}
	if !containsImport(t.Content, "math/rand") {
		return nil
	}
	return []analyze.Finding{{
		RuleID:       "security/weak-random",
		Severity:     analyze.SeverityMedium,
		Category:     analyze.CategorySecurity,
		Message:      "math/rand imported in production code; use crypto/rand for anything security-sensitive",
		Fix:          "Replace math/rand with crypto/rand.",
		TargetURI:    t.URI,
		TargetNodeID: t.FileNode.ID,
	}}
}

// containsImport scans a Go file for `import "pkg"` or a grouped import of pkg.
func containsImport(content []byte, pkg string) bool {
	quoted := `"` + pkg + `"`
	return bytes.Contains(content, []byte("import "+quoted)) ||
		bytes.Contains(content, []byte("\t"+quoted)) ||
		bytes.Contains(content, []byte(" "+quoted+"\n"))
}

func init() { analyze.Register(WeakRandom{}) }
