package bugs

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// SelfAssignment flags `x = x` — assigning a variable (or member expression) to
// itself. It is a no-op and almost always a typo (e.g. `x = x` where `x = y`
// or `this.x = x` was intended). The rule only fires when the left- and
// right-hand sides are textually identical simple lvalues, keeping it
// low false-positive.
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

// Capture an lvalue (identifier or simple member/index access), a bare `=`
// (also matches Go's `:=`), then the SAME text via a backreference. Anchored to
// statement start (leading whitespace only) so we don't catch keyword args or
// comparisons. The trailing boundary forbids a continuation (e.g. `x = x + 1`).
//
// Go's regexp (RE2) has no backreferences, so identity is verified in code.
var selfAssignRe = regexp.MustCompile(
	`^\s*([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*:?=\s*([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*;?\s*$`)

func (SelfAssignment) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		// Apply both line-comment styles plus inline /* */ block comments;
		// harmless across the supported languages. Stripping a trailing block
		// comment keeps `x = x /* note */` matching the end-of-line anchor.
		text := stripPyComment(stripInlineBlockComment(stripGoComment(sc.Text())))
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
