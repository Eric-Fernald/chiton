// Package reliability contains deterministic rules in the reliability category.
package reliability

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// SwallowedPanic flags Go `recover()` results that are assigned to the blank
// identifier — the recovered panic is silently dropped without logging,
// re-raising, or surfacing.
type SwallowedPanic struct{}

func (SwallowedPanic) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "reliability/swallowed-panic",
		Category:   analyze.CategoryReliability,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.FileScope,
		Languages:  []string{"go"},
		Summary:    "recover() result discarded — the panic is silently swallowed.",
		Fix:        "Log, instrument, or re-panic the recovered value.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var swallowedRecoverRe = regexp.MustCompile(`(^|\s)_\s*[:=]=\s*recover\s*\(\s*\)`)

func (SwallowedPanic) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		if swallowedRecoverRe.MatchString(sc.Text()) {
			out = append(out, analyze.Finding{
				RuleID:       "reliability/swallowed-panic",
				Severity:     analyze.SeverityHigh,
				Category:     analyze.CategoryReliability,
				Message:      "recover() result discarded — panic lost",
				Fix:          "Capture the value, log it, or re-panic.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(SwallowedPanic{}) }
