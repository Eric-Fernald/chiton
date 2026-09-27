package bugs

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// UncheckedError flags Go `_ = call()`: `_, err := ...` is fine, but discarding a call's sole result
// swallows its error, the most common silent-failure pattern in Go.
type UncheckedError struct{}

func (UncheckedError) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "bugs/unchecked-error",
		Category:   analyze.CategoryBugs,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go"},
		Summary:    "Error from a call assigned to blank identifier (`_ = foo()`).",
		Fix:        "Capture the error, handle it, or document why it can be ignored.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var uncheckedErrorRe = regexp.MustCompile(`(^|\s)_\s*[:=]=\s*\w[\w.]*\s*\(`)

func (UncheckedError) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, raw := range analyze.FileLines(t) {
		text := stripGoComment(raw)
		if uncheckedErrorRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "bugs/unchecked-error",
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryBugs,
				Message:      "error possibly swallowed via blank identifier",
				Fix:          "Replace `_ =` with proper error handling.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(UncheckedError{}) }
