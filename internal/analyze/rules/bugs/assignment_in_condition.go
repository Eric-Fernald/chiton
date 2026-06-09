package bugs

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// AssignmentInCondition flags `if (x = y)` / `while (x = y)` in C-family
// languages (JS/TS/Java) where a single `=` inside the condition is almost
// always a typo for `==` / `===`. Compound assignment operators (+=, ==, etc.)
// are excluded.
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

// `if (` or `while (` followed by an identifier/member-access, then a lone `=`
// (not ==, +=, -=, etc.) and a value. The char before `=` is constrained to the
// identifier/whitespace tail so compound operators (`+=`, `<=`) don't match.
var assignInCondBareRe = regexp.MustCompile(
	`\b(if|while)\s*\(\s*[A-Za-z_$][\w$.\[\]]*\s*=([^=]|$)`)

func (AssignmentInCondition) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripJSComment(sc.Text())
		if assignInCondBareRe.MatchString(text) {
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
