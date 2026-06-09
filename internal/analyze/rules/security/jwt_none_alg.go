package security

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// JWTNoneAlg flags a JWT configured with the "none" algorithm, which produces
// unsigned tokens that anyone can forge. Matches an alg/algorithm key whose
// value is the literal string "none".
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
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripSlashComment(stripPyComment(sc.Text()))
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
