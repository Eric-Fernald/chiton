package reliability

import (
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// LogFatalInCode flags log.Fatal* / log.Panic* in non-main Go code: Fatal calls os.Exit(1), skipping
// deferred cleanup, and Panic panics the goroutine; both take control away from callers.
type LogFatalInCode struct{}

func (LogFatalInCode) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "reliability/log-fatal-in-code",
		Category:   analyze.CategoryReliability,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go"},
		Summary:    "log.Fatal*/log.Panic* in library code exits or panics the whole process.",
		Fix:        "Return an error to the caller; reserve log.Fatal/Panic for main().",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// Anchoring on `log.` keeps unrelated `.Fatal(` methods on other receivers from matching.
var logFatalRe = regexp.MustCompile(`\blog\.(Fatal|Fatalf|Fatalln|Panic|Panicf|Panicln)\s*\(`)

func (LogFatalInCode) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	if isGoTestFile(t.Path) || isGoMainFile(t.Path) {
		return nil
	}
	if !bytes.Contains(t.Content, []byte("log.")) {
		return nil
	}
	// Per-file masking removes log.Fatal/Panic tokens inside string literals and
	// //, /* */ (incl. multi-line) comments, so commented-out code is not flagged.
	masked := analyze.MaskedGoLiterals(t)
	var out []analyze.Finding
	for line, text := range analyze.Lines(masked) {
		if logFatalRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "reliability/log-fatal-in-code",
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryReliability,
				Message:      "log.Fatal/Panic in library code terminates the whole process",
				Fix:          "Return an error; reserve log.Fatal/Panic for main().",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(LogFatalInCode{}) }
