package bugs

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// AssignmentInCondition flags `if (x = y)` / `while (x = y)` in JS/TS/Java, where a lone `=` in the
// condition is almost always a typo for `==` / `===`.
type AssignmentInCondition struct{}

func (AssignmentInCondition) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "bugs/assignment-in-condition",
		Category:   analyze.CategoryBugs,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.FileScope,
		Languages:  []string{"js", "ts", "jsx", "tsx", "java"},
		Summary:    "Assignment (=) inside an if/while condition; likely meant == / ===.",
		Fix:        "Use == / === for comparison, or move the assignment outside the condition.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// The identifier/member-only lvalue keeps compound operators (`+=`, `<=`) from matching; `([^=]|$)`
// rejects `==`.
var assignInCondBareRe = regexp.MustCompile(
	`\b(if|while)\s*\(\s*[A-Za-z_$][\w$.\[\]]*\s*=([^=]|$)`)

func (AssignmentInCondition) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, raw := range analyze.FileLines(t) {
		if assignInCondBareRe.MatchString(analyze.StripLineComment(raw, false)) {
			out = append(out, analyze.Finding{
				RuleID:       "bugs/assignment-in-condition",
				Severity:     analyze.SeverityHigh,
				Category:     analyze.CategoryBugs,
				Message:      "assignment (=) inside if/while condition; likely meant ==",
				Fix:          "Use == / === for comparison.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(AssignmentInCondition{}) }
