// Package reliability contains deterministic rules in the reliability category.
package reliability

import (
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// SwallowedPanic flags Go `recover()` results assigned to the blank identifier: the recovered panic is
// dropped without logging, re-raising, or surfacing.
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
	if !bytes.Contains(t.Content, []byte("recover")) {
		return nil
	}
	var out []analyze.Finding
	for line, text := range analyze.FileLines(t) {
		if swallowedRecoverRe.MatchString(text) {
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
