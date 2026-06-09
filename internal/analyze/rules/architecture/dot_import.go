package architecture

import (
	"bufio"
	"bytes"
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DotImport flags Go dot imports (`import . "pkg"` and dot entries inside an
// `import ( ... )` block). Dot imports merge another package's exported names
// into the file scope, making it impossible to tell where an identifier comes
// from and risking silent shadowing.
type DotImport struct{}

func (DotImport) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "architecture/dot-import",
		Category:   analyze.CategoryArchitecture,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go"},
		Summary:    "Go dot import merges another package's names into file scope.",
		Fix:        "Use a normal qualified import so identifiers stay traceable.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var (
	// Single-line:  import . "path"
	goDotImportLineRe = regexp.MustCompile(`^import\s+\.\s+"[^"]+"`)
	// Inside a block:  . "path"
	goDotBlockEntryRe = regexp.MustCompile(`^\.\s+"[^"]+"`)
)

func (DotImport) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	inBlock := false
	for sc.Scan() {
		line++
		text := strings.TrimSpace(stripArchLineComment("go", sc.Text()))
		switch {
		case inBlock:
			if strings.HasPrefix(text, ")") {
				inBlock = false
				continue
			}
			if goDotBlockEntryRe.MatchString(text) {
				out = append(out, dotImportFinding(t, line))
			}
		case strings.HasPrefix(text, "import ("):
			inBlock = true
		case goDotImportLineRe.MatchString(text):
			out = append(out, dotImportFinding(t, line))
		}
	}
	return out
}

func dotImportFinding(t analyze.FileTarget, line int) analyze.Finding {
	return analyze.Finding{
		RuleID:       "architecture/dot-import",
		Severity:     analyze.SeverityMedium,
		Category:     analyze.CategoryArchitecture,
		Message:      "dot import merges package names into file scope",
		Fix:          "Use a normal qualified import.",
		TargetURI:    t.URI,
		TargetNodeID: t.FileNode.ID,
		Span:         &model.SourceSpan{StartLine: line, EndLine: line},
	}
}

func init() { analyze.Register(DotImport{}) }
