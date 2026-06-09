package reliability

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// LogFatalInCode flags log.Fatal / Fatalf / Fatalln and log.Panic / Panicf /
// Panicln in non-entrypoint Go code. log.Fatal* calls os.Exit(1) (skipping
// deferred cleanup) and log.Panic* panics the goroutine — both rip control away
// from callers. They belong in main(), not in library code.
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

// Matches a `log` (or aliased *log) selector calling Fatal/Fatalf/Fatalln or
// Panic/Panicf/Panicln. Anchoring on `log.` keeps it from matching unrelated
// `.Fatal(` methods on non-logger receivers.
var logFatalRe = regexp.MustCompile(`\blog\.(Fatal|Fatalf|Fatalln|Panic|Panicf|Panicln)\s*\(`)

func (LogFatalInCode) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	if isGoTestFile(t.Path) || isGoMainFile(t.Path) {
		return nil
	}
	// Per-file masking removes log.Fatal/Panic tokens inside string literals and
	// //, /* */ (incl. multi-line) comments, so commented-out code is not flagged.
	masked := maskGoLiteralsAndComments(t.Content)
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(masked))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
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
