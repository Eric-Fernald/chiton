package bugs

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// SelfAssignment flags `x = x` (a no-op, almost always a typo for `x = y` or `this.x = x`). It fires only
// when both sides are textually identical simple lvalues, keeping false positives low.
type SelfAssignment struct{}

func (SelfAssignment) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "bugs/self-assignment",
		Category:   analyze.CategoryBugs,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "java", "rs"},
		Summary:    "Variable assigned to itself (x = x); a no-op, likely a typo.",
		Fix:        "Assign the intended value, or remove the redundant statement.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// A whole statement `lhs = rhs` (or `:=`) of identifiers/dotted members. RE2 has no backreferences, so
// MatchFile checks lhs == rhs itself.
var selfAssignRe = regexp.MustCompile(
	`^\s*([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*:?=\s*([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*;?\s*$`)

func (SelfAssignment) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, raw := range analyze.Lines(t.Content) {
		if strings.IndexByte(raw, '=') < 0 {
			continue
		}
		// Strip every supported comment style; a trailing /* */ would defeat the end-of-line anchor.
		text := analyze.StripPyComment(stripInlineBlockComment(stripGoComment(raw)))
		m := selfAssignRe.FindStringSubmatch(text)
		if m != nil && m[1] == m[2] {
			out = append(out, analyze.Finding{
				RuleID:       "bugs/self-assignment",
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryBugs,
				Message:      "variable assigned to itself (no-op); likely a typo",
				Fix:          "Assign the intended value or remove the statement.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(SelfAssignment{}) }
