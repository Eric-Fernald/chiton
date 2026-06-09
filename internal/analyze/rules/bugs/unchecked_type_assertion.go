package bugs

import (
	"bufio"
	"bytes"
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// UncheckedTypeAssertion flags single-value Go type assertions of the form
// `v := x.(T)` or `v = x.(T)`. A single-value assertion PANICS when the dynamic
// type doesn't match; the comma-ok form `v, ok := x.(T)` is safe. The `.(type)`
// switch form is excluded.
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

// A single LHS target (no comma), `:=` or `=`, then an expression ending in a
// type assertion `.(T)` for the whole RHS. Excludes the comma-ok form (the
// leading group forbids a `,` in the LHS) and the type-switch `.(type)` form.
var uncheckedAssertRe = regexp.MustCompile(
	`^\s*[A-Za-z_]\w*\s*:?=\s*[\w.()*\[\]]*\.\(\s*[*\[\]\w./]+\s*\)\s*$`)

func (UncheckedTypeAssertion) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripGoComment(sc.Text())
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
