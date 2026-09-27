package security

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// InsecureTLSVerify flags code that disables TLS certificate verification (Go, Python, Node);
// critical because it re-enables active MITM attacks.
type InsecureTLSVerify struct{}

func (InsecureTLSVerify) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/insecure-tls-verify",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityCritical,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx"},
		Summary:    "TLS certificate verification disabled (MITM exposure).",
		Fix:        "Enable certificate verification; only disable it in isolated test code with a documented reason.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var insecureTLSPatterns = []*regexp.Regexp{
	regexp.MustCompile(`InsecureSkipVerify\s*:\s*true`),                     // Go
	regexp.MustCompile(`ssl\._create_unverified_context\s*\(`),              // Python ssl
	regexp.MustCompile(`(?i)\brejectUnauthorized\s*:\s*false`),              // Node/JS
	regexp.MustCompile(`NODE_TLS_REJECT_UNAUTHORIZED["']?\s*[:=]\s*["']?0`), // Node env
}

// Python `verify=False` counts only with an HTTP-client call on the same line, so unrelated kwargs
// like pandas read_csv(verify=False) don't match.
var (
	insecureTLSPyVerify  = regexp.MustCompile(`(?i)\bverify\s*=\s*False\b`)
	insecureTLSPyHTTPCtx = regexp.MustCompile(`(?i)\b(?:requests|httpx|session)\s*\.|\.(?:get|post|put|request)\s*\(`)
)

func (InsecureTLSVerify) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, text := range analyze.Lines(t.Content) {
		if !containsAny(text, ":", "=", "_create_unverified_context") {
			continue
		}
		text = stripSlashCommentKeepURL(analyze.StripPyComment(text))
		matched := false
		for _, re := range insecureTLSPatterns {
			if re.MatchString(text) {
				matched = true
				break
			}
		}
		// Python verify=False only counts inside an HTTP-client context.
		if !matched && insecureTLSPyVerify.MatchString(text) && insecureTLSPyHTTPCtx.MatchString(text) {
			matched = true
		}
		if matched {
			out = append(out, analyze.Finding{
				RuleID:       "security/insecure-tls-verify",
				Severity:     analyze.SeverityCritical,
				Category:     analyze.CategorySecurity,
				Message:      "TLS certificate verification disabled",
				Fix:          "Re-enable certificate verification.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

// stripSlashCommentKeepURL truncates s at a `//` outside a quoted string, keeping string contents and
// `://` URL separators; block comments are left intact.
func stripSlashCommentKeepURL(s string) string {
	var inStr byte
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inStr != 0 {
			if c == '\\' && i+1 < len(s) {
				i++
				continue
			}
			if c == inStr {
				inStr = 0
			}
			continue
		}
		switch c {
		case '"', '\'', '`':
			inStr = c
		case '/':
			if i+1 < len(s) && s[i+1] == '/' && (i == 0 || s[i-1] != ':') {
				return s[:i]
			}
		}
	}
	return s
}

func init() { analyze.Register(InsecureTLSVerify{}) }
