package architecture

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"path"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// TooManyImportsThreshold is the import-statement count above which a file is
// flagged. Calibrated high so only genuinely import-heavy files (a sign of a
// module doing too much / poor cohesion) are reported.
const TooManyImportsThreshold = 25

// TooManyImports flags a source file that pulls in more than
// TooManyImportsThreshold distinct import statements — a coupling smell.
//
// Counting is language-aware and conservative:
//   - Go: lines inside an `import ( ... )` block, plus single-line
//     `import "x"` / `import alias "x"` statements.
//   - Python: `import x` and `from x import y` lines.
//   - JS/TS: ES `import ... from "x"` / bare `import "x"` lines and
//     CommonJS `require("x")` calls.
//   - Rust: top-level `use ...;` lines.
//   - Java: `import ...;` lines.
//
// Comments and strings that merely contain the keyword do not count because we
// anchor on the leading token of a (trimmed) code line.
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
	// ES `import` statement, but NOT a dynamic `import(...)` call or the
	// `import.meta` meta-property — i.e. the char right after `import` must not
	// be `(` or `.` (a real statement has whitespace or a quote there).
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
	ext := strings.ToLower(strings.TrimPrefix(path.Ext(t.Path), "."))
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
	sc := bufio.NewScanner(bytes.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	count := 0
	inGoBlock := false
	for sc.Scan() {
		line := strings.TrimSpace(stripArchLineComment(ext, sc.Text()))
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

// stripArchLineComment removes a trailing single-line comment so a comment that
// merely mentions an import keyword does not inflate the count. Conservative:
// only handles `//` (and `#` for Python); does not parse block comments.
func stripArchLineComment(ext, s string) string {
	marker := "//"
	if ext == "py" {
		marker = "#"
	}
	if idx := strings.Index(s, marker); idx >= 0 {
		// Avoid cutting inside an obvious string by only stripping when the
		// marker is not preceded by a quote on the same line segment. This is a
		// cheap heuristic; import lines rarely contain quoted "//".
		before := s[:idx]
		if strings.Count(before, `"`)%2 == 0 && strings.Count(before, "'")%2 == 0 {
			return before
		}
	}
	return s
}

func init() { analyze.Register(TooManyImports{}) }
