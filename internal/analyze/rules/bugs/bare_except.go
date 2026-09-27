package bugs

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// BareExcept flags Python `except:` (no exception class) and JS/TS `catch () {`: both swallow every
// exception, including KeyboardInterrupt / TypeError, and hide real bugs.
type BareExcept struct{}

func (BareExcept) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "bugs/bare-except",
		Category:   analyze.CategoryBugs,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.FileScope,
		Languages:  []string{"py", "js", "ts", "jsx", "tsx"},
		Summary:    "Bare except: / empty catch swallows all errors, hiding real bugs.",
		Fix:        "Catch a specific exception class, or re-raise after logging.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var (
	bareExceptPy = regexp.MustCompile(`^\s*except\s*:\s*(#.*)?$`)
	bareCatchJS  = regexp.MustCompile(`\bcatch\s*\(\s*\)\s*\{`)
)

func (BareExcept) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, text := range analyze.Lines(t.Content) {
		if bareExceptPy.MatchString(text) || bareCatchJS.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "bugs/bare-except",
				Severity:     analyze.SeverityHigh,
				Category:     analyze.CategoryBugs,
				Message:      "bare except / empty catch swallows all errors",
				Fix:          "Catch a specific exception class.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(BareExcept{}) }
