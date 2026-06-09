package reliability

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// PanicInSource flags an explicit `panic(` in non-test Go source. A panic
// unwinds the stack and, if unrecovered, crashes the process — library code
// should prefer returning an error so callers can decide how to handle it.
type PanicInSource struct{}

func (PanicInSource) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "reliability/panic-in-source",
		Category:   analyze.CategoryReliability,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go"},
		Summary:    "Explicit panic() in non-test source; prefer returning an error.",
		Fix:        "Return an error to the caller instead of panicking.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// `panic(` as a call. The leading word boundary keeps it from matching the
// `.panic(` of an unrelated method or an identifier ending in "panic".
var panicCallRe = regexp.MustCompile(`(^|[^.\w])panic\s*\(`)

func (PanicInSource) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	if isGoTestFile(t.Path) {
		return nil
	}
	// Per-file masking removes panic( tokens inside string literals and //, /* */
	// (incl. multi-line) comments, so commented-out code is not flagged.
	masked := maskGoLiteralsAndComments(t.Content)
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(masked))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
		if panicCallRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "reliability/panic-in-source",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryReliability,
				Message:      "explicit panic() in source; prefer returning an error",
				Fix:          "Return an error to the caller instead of panicking.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(PanicInSource{}) }
