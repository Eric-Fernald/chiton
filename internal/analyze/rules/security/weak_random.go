package security

import (
	"context"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// WeakRandom flags Go files importing math/rand. crypto/rand should be used
// for anything security-sensitive. Test files are skipped because they
// legitimately want determinism.
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
	// The Go extractor records imports as IMPORTS edges with the package label
	// on the destination node. Scan edges to detect the import.
	for _, e := range t.Edges {
		if e.Kind != model.EdgeImports {
			continue
		}
		// Destination is a "pkg://<pkg>" node — we don't have that node in the
		// file's local Nodes slice, but the edge's Dst ID is derived from the
		// label. Easier: scan the raw content for the canonical import line.
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
	src := string(content)
	quoted := `"` + pkg + `"`
	// Match either `import "pkg"` form or a grouped import line.
	return strings.Contains(src, "import "+quoted) ||
		strings.Contains(src, "\t"+quoted) ||
		strings.Contains(src, " "+quoted+"\n")
}

func init() { analyze.Register(WeakRandom{}) }
