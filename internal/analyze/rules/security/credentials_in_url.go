package security

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// CredentialsInURL flags scheme://user:pass@host URLs, which leak secrets into logs, history, and
// process listings.
type CredentialsInURL struct{}

func (CredentialsInURL) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/credentials-in-url",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "java"},
		Summary:    "Credentials embedded in a URL (scheme://user:pass@host).",
		Fix:        "Pass credentials out of band (env var / secret manager), not in the URL.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// credsInURLRe requires a non-empty user:pass before '@', so "host:port" and "git@host" don't match.
var credsInURLRe = regexp.MustCompile(`\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@"']+:[^\s:/@"']+@[\w.\-]`)

func (CredentialsInURL) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, text := range analyze.FileLines(t) {
		if strings.Contains(text, "://") && credsInURLRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "security/credentials-in-url",
				Severity:     analyze.SeverityHigh,
				Category:     analyze.CategorySecurity,
				Message:      "credentials embedded in a URL",
				Fix:          "Keep credentials out of URLs; pass them via env var or secret manager.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(CredentialsInURL{}) }
