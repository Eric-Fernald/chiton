package security

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// JWTNoneAlg flags an alg/algorithm(s) key set to the literal "none", which produces unsigned
// tokens that anyone can forge.
type JWTNoneAlg struct{}

func (JWTNoneAlg) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/jwt-none-alg",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "java"},
		Summary:    "JWT 'none' algorithm — tokens are unsigned and forgeable.",
		Fix:        "Use a real signing algorithm (e.g. HS256/RS256) and reject 'none'.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// Match "alg":"none" / 'alg': 'none' / algorithm: "none" / algorithms: ["none"].
var jwtNoneRe = regexp.MustCompile(`(?i)["']?\b(alg|algorithm|algorithms)\b["']?\s*[:=]\s*[\[]?\s*["']none["']`)

func (JWTNoneAlg) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, text := range analyze.Lines(t.Content) {
		if !containsAny(text, ":", "=") {
			continue
		}
		text = stripSlashCommentKeepURL(analyze.StripPyComment(text))
		if jwtNoneRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "security/jwt-none-alg",
				Severity:     analyze.SeverityHigh,
				Category:     analyze.CategorySecurity,
				Message:      "JWT 'none' algorithm — tokens are unsigned and forgeable",
				Fix:          "Use a real signing algorithm and reject 'none'.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(JWTNoneAlg{}) }
