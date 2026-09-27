package architecture

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// TooManyImportsThreshold is the import-statement count above which TooManyImports flags a file. It is
// set high so only import-heavy files are reported.
const TooManyImportsThreshold = 25

// TooManyImports flags a file with more than TooManyImportsThreshold import statements (a coupling
// smell), counted per language from the leading token of each comment-stripped line.
type TooManyImports struct{}

func (TooManyImports) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "architecture/too-many-imports",
		Category:   analyze.CategoryArchitecture,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "rs", "java"},
		Summary:    fmt.Sprintf("File has more than %d import statements.", TooManyImportsThreshold),
		Fix:        "Reduce coupling: split the file, or depend on fewer modules.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var (
	pyImportRe = regexp.MustCompile(`^(import\s+\S|from\s+\S+\s+import\s)`)
	// jsImportRe matches an ES import statement. The first char after `import` may not be `(` or `.`, so
	// dynamic import(...) and import.meta don't count.
	jsImportRe   = regexp.MustCompile(`^import\b\s*[^(.]`)
	jsRequireRe  = regexp.MustCompile(`\brequire\s*\(\s*['"]`)
	rustUseRe    = regexp.MustCompile(`^use\s+\S`)
	javaImportRe = regexp.MustCompile(`^import\s+\S`)
	goImportRe   = regexp.MustCompile(`^import\s+(\(|"|[A-Za-z_.])`)
	// A line inside a Go import block is either `"path"` or `alias "path"`
	// or `. "path"` or `_ "path"`.
	goBlockEntryRe = regexp.MustCompile(`^(\.|_|[A-Za-z][\w.]*)?\s*"[^"]+"`)
)

func (TooManyImports) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	ext := analyze.FileExt(t.Path)
	count := countImports(ext, t.Content)
	if count <= TooManyImportsThreshold {
		return nil
	}
	return []analyze.Finding{{
		RuleID:       "architecture/too-many-imports",
		Severity:     analyze.SeverityMedium,
		Category:     analyze.CategoryArchitecture,
		Message:      fmt.Sprintf("file has %d import statements (limit %d)", count, TooManyImportsThreshold),
		Fix:          "Reduce coupling: split the file or depend on fewer modules.",
		TargetURI:    t.URI,
		TargetNodeID: t.FileNode.ID,
		Span:         &model.SourceSpan{StartLine: 1, EndLine: 1},
	}}
}

func countImports(ext string, content []byte) int {
	count := 0
	inGoBlock := false
	for _, raw := range analyze.Lines(content) {
		line := strings.TrimSpace(stripArchLineComment(ext, raw))
		if line == "" {
			continue
		}
		switch ext {
		case "go":
			if inGoBlock {
				if strings.HasPrefix(line, ")") {
					inGoBlock = false
					continue
				}
				if goBlockEntryRe.MatchString(line) {
					count++
				}
				continue
			}
			if strings.HasPrefix(line, "import (") {
				inGoBlock = true
				continue
			}
			if goImportRe.MatchString(line) {
				count++
			}
		case "py":
			if pyImportRe.MatchString(line) {
				count++
			}
		case "js", "ts", "jsx", "tsx":
			if jsImportRe.MatchString(line) || jsRequireRe.MatchString(line) {
				count++
			}
		case "rs":
			if rustUseRe.MatchString(line) {
				count++
			}
		case "java":
			if javaImportRe.MatchString(line) {
				count++
			}
		}
	}
	return count
}

// stripArchLineComment cuts s at the first `//` (`#` for Python) unless the quotes before it are
// unbalanced. It does not handle block comments.
func stripArchLineComment(ext, s string) string {
	marker := "//"
	if ext == "py" {
		marker = "#"
	}
	if idx := strings.Index(s, marker); idx >= 0 {
		// Unbalanced quotes mean the first marker sits inside a string, so keep the line whole.
		before := s[:idx]
		if strings.Count(before, `"`)%2 == 0 && strings.Count(before, "'")%2 == 0 {
			return before
		}
	}
	return s
}

func init() { analyze.Register(TooManyImports{}) }
