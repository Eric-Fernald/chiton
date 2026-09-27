package bugs

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// UncheckedTypeAssertion flags single-value Go type assertions (`v := x.(T)`), which panic on a type
// mismatch; the comma-ok form and `.(type)` switches are not flagged.
type UncheckedTypeAssertion struct{}

func (UncheckedTypeAssertion) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "bugs/unchecked-type-assertion",
		Category:   analyze.CategoryBugs,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go"},
		Summary:    "Single-value type assertion x.(T) panics on mismatch; prefer comma-ok.",
		Fix:        "Use the two-result form: v, ok := x.(T) and handle !ok.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// A whole statement `v := expr.(T)` / `v = expr.(T)`; the single-identifier LHS excludes the comma-ok form.
var uncheckedAssertRe = regexp.MustCompile(
	`^\s*[A-Za-z_]\w*\s*:?=\s*[\w.()*\[\]]*\.\(\s*[*\[\]\w./]+\s*\)\s*$`)

func (UncheckedTypeAssertion) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, raw := range analyze.FileLines(t) {
		text := stripGoComment(raw)
		if !uncheckedAssertRe.MatchString(text) {
			continue
		}
		// Exclude the type-switch guard form: `x := y.(type)`.
		if strings.Contains(text, ".(type)") {
			continue
		}
		out = append(out, analyze.Finding{
			RuleID:       "bugs/unchecked-type-assertion",
			Severity:     analyze.SeverityMedium,
			Category:     analyze.CategoryBugs,
			Message:      "single-value type assertion panics on mismatch; use comma-ok",
			Fix:          "Use v, ok := x.(T) and handle the !ok case.",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		})
	}
	return out
}

func init() { analyze.Register(UncheckedTypeAssertion{}) }
