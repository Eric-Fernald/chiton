package quality

import (
	"bufio"
	"bytes"
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// MagicNumber flags integer literals >= 100 that don't sit on a const/let/=
// declaration. Heuristic, conservative — only fires when the line looks like
// "expression with a multi-digit literal that isn't bound to a name."
type MagicNumber struct{}

func (MagicNumber) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "quality/magic-number",
		Category:   analyze.CategoryQuality,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Summary:    "Unnamed integer literal in code (>= 100); consider naming it.",
		Fix:        "Extract the value into a named constant explaining its meaning.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var (
	magicNumberRe = regexp.MustCompile(`\b\d{3,}\b`)
	declTokens    = []string{
		"const ", "var ", "let ", "final ", "static ", "= ", " : ",
	}
)

func (MagicNumber) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
		trimmed := strings.TrimSpace(text)
		if trimmed == "" || strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "#") || strings.HasPrefix(trimmed, "*") {
			continue
		}
		// Lines that already declare a name with the literal are fine.
		if hasDecl(text) {
			continue
		}
		if magicNumberRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "quality/magic-number",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryQuality,
				Message:      "unnamed numeric literal — consider extracting to a constant",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func hasDecl(line string) bool {
	for _, tok := range declTokens {
		if strings.Contains(line, tok) {
			return true
		}
	}
	return false
}

func init() { analyze.Register(MagicNumber{}) }
