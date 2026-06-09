package reliability

import (
	"bufio"
	"bytes"
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// AbruptExit flags abrupt process termination from code that is not an obvious
// entrypoint: `os.Exit(` in Go, or `process.exit(` in JS/TS. Calling these from
// library code skips deferred cleanup (Go) and prevents callers from handling
// the failure — exits belong in main/CLI top-level wiring, not in reusable code.
type AbruptExit struct{}

func (AbruptExit) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "reliability/abrupt-exit",
		Category:   analyze.CategoryReliability,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "js", "ts"},
		Summary:    "Abrupt process exit (os.Exit / process.exit) outside an entrypoint.",
		Fix:        "Return an error and let the entrypoint decide how to terminate.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var (
	osExitRe      = regexp.MustCompile(`\bos\.Exit\s*\(`)
	processExitRe = regexp.MustCompile(`\bprocess\.exit\s*\(`)
)

func (AbruptExit) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	ext := langExt(t.Path)
	if isGoTestFile(t.Path) || isJSTestFile(t.Path) {
		return nil
	}
	// Entrypoints are allowed to exit. For Go, any `package main` file (not just
	// main.go — cmd/<tool>/ files often use other names); for JS/TS, the common
	// entry basenames, bin/ paths, and files with a shebang on line one.
	if isEntrypointFile(t.Path, t.Content) {
		return nil
	}

	var re *regexp.Regexp
	var masked []byte
	switch ext {
	case "go":
		re = osExitRe
		masked = maskGoLiteralsAndComments(t.Content)
	case "js", "ts":
		re = processExitRe
		masked = maskLiteralsAndComments(t.Content)
	default:
		return nil
	}

	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(masked))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
		if re.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "reliability/abrupt-exit",
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryReliability,
				Message:      "abrupt process exit from non-entrypoint code",
				Fix:          "Return an error and let the entrypoint terminate the process.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

// isJSTestFile recognises the common JS/TS test-file naming conventions.
func isJSTestFile(path string) bool {
	base := basename(path)
	for _, suf := range []string{".test.js", ".test.ts", ".test.jsx", ".test.tsx",
		".spec.js", ".spec.ts", ".spec.jsx", ".spec.tsx"} {
		if len(base) > len(suf) && base[len(base)-len(suf):] == suf {
			return true
		}
	}
	return false
}

// isEntrypointFile recognises conventional program entrypoints where exiting is
// expected. Go: any file whose package clause is `package main` (covers main.go
// and arbitrarily-named files under cmd/<tool>/). JS/TS: the common entry
// basenames (index/main/cli/server/app/start), files under a bin/ directory, and
// files that begin with a `#!` shebang.
func isEntrypointFile(path string, content []byte) bool {
	switch langExt(path) {
	case "go":
		return hasGoMainPackage(content)
	case "js", "ts", "jsx", "tsx", "mjs", "cjs":
		switch jsBasenameStem(path) {
		case "index", "main", "cli", "server", "app", "start":
			return true
		}
		if isUnderBinDir(path) || hasShebang(content) {
			return true
		}
	}
	return false
}

// hasGoMainPackage reports whether the file's package clause is `package main`,
// scanning the masked content so a `package main` inside a string/comment does
// not count.
func hasGoMainPackage(content []byte) bool {
	return goMainPkgRe.Match(maskGoLiteralsAndComments(content))
}

var goMainPkgRe = regexp.MustCompile(`(?m)^\s*package\s+main\b`)

// isUnderBinDir reports whether any path segment is exactly "bin".
func isUnderBinDir(path string) bool {
	p := "/" + filepathSlashed(path) + "/"
	return strings.Contains(p, "/bin/")
}

// hasShebang reports whether the file's first line begins with `#!`.
func hasShebang(content []byte) bool {
	return len(content) >= 2 && content[0] == '#' && content[1] == '!'
}

// jsBasenameStem returns the basename with a single trailing extension removed
// and lowercased (e.g. "Server.js" -> "server"). It is intentionally simple:
// only one extension is stripped, which suffices for the entry-name check.
func jsBasenameStem(path string) string {
	base := basename(path)
	for i := len(base) - 1; i >= 0; i-- {
		if base[i] == '.' {
			base = base[:i]
			break
		}
	}
	return strings.ToLower(base)
}

func basename(path string) string {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' || path[i] == '\\' {
			return path[i+1:]
		}
	}
	return path
}

func init() { analyze.Register(AbruptExit{}) }
