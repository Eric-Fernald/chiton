package reliability

import (
	"bytes"
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// AbruptExit flags `os.Exit(` (Go) or `process.exit(` (JS/TS) outside an entrypoint. Exiting from library
// code skips deferred cleanup and denies callers the chance to handle the failure.
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
	goMainPkgRe   = regexp.MustCompile(`(?m)^\s*package\s+main\b`)
)

func (AbruptExit) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	if isGoTestFile(t.Path) || isJSTestFile(t.Path) {
		return nil
	}
	// Entrypoints may exit. That covers any `package main` Go file (cmd/<tool>/ files are often not main.go)
	// and the JS/TS entrypoints that isJSEntrypoint recognises.
	var re *regexp.Regexp
	var masked []byte
	switch analyze.FileExt(t.Path) {
	case "go":
		// The prefilter is exact since masking writes only 'x' and never masks a lone byte between letters.
		if !bytes.Contains(t.Content, []byte("os.Exit")) {
			return nil
		}
		masked = analyze.MaskedGoLiterals(t)
		if goMainPkgRe.Match(masked) {
			return nil
		}
		re = osExitRe
	case "js", "ts":
		if isJSEntrypoint(t.Path, t.Content) || !bytes.Contains(t.Content, []byte("process.exit")) {
			return nil
		}
		re = processExitRe
		masked = analyze.MaskedCLiterals(t, false)
	default:
		return nil
	}

	var out []analyze.Finding
	for line, text := range analyze.Lines(masked) {
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

var jsTestSuffixes = []string{".test.js", ".test.ts", ".test.jsx", ".test.tsx",
	".spec.js", ".spec.ts", ".spec.jsx", ".spec.tsx"}

// isJSTestFile recognises the common JS/TS test-file naming conventions.
func isJSTestFile(path string) bool {
	base := analyze.PathBase(path)
	for _, suf := range jsTestSuffixes {
		if len(base) > len(suf) && strings.HasSuffix(base, suf) {
			return true
		}
	}
	return false
}

// isJSEntrypoint recognises JS/TS entrypoints: common entry basenames (index/main/cli/server/app/start),
// files under a bin/ directory, and files starting with a `#!` shebang.
func isJSEntrypoint(path string, content []byte) bool {
	switch jsBasenameStem(path) {
	case "index", "main", "cli", "server", "app", "start":
		return true
	}
	return isUnderBinDir(path) || hasShebang(content)
}

// isUnderBinDir reports whether any path segment is exactly "bin".
func isUnderBinDir(path string) bool {
	p := "/" + filepathSlashed(path) + "/"
	return strings.Contains(p, "/bin/")
}

// hasShebang reports whether the file's first line begins with `#!`.
func hasShebang(content []byte) bool {
	return len(content) >= 2 && content[0] == '#' && content[1] == '!'
}

// jsBasenameStem returns the lowercased basename without one trailing extension, so "Server.js" becomes
// "server". Stripping only one extension is enough for the entry-name check.
func jsBasenameStem(path string) string {
	base := analyze.PathBase(path)
	if i := strings.LastIndexByte(base, '.'); i >= 0 {
		base = base[:i]
	}
	return strings.ToLower(base)
}

func init() { analyze.Register(AbruptExit{}) }
