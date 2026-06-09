package security

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// CredentialsInURL flags a URL with embedded credentials of the form
// scheme://user:pass@host. Such URLs leak secrets into logs, history, and
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

// scheme://user:pass@host. The userinfo must contain a colon (user:pass) so a
// bare "host:port" path component does not match. Scheme, user, and pass are
// restricted to URL-credential-safe characters to keep this conservative.
// A non-empty password segment is required to avoid matching e.g. "git@host".
var credsInURLRe = regexp.MustCompile(`\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@"']+:[^\s:/@"']+@[\w.\-]`)

func (CredentialsInURL) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
		if credsInURLRe.MatchString(text) {
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
