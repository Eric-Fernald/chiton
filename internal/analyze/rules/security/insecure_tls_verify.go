package security

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// InsecureTLSVerify flags code that disables TLS certificate verification:
//   - Go:  InsecureSkipVerify: true
//   - Py:  verify=False  /  ssl._create_unverified_context
//   - JS:  rejectUnauthorized: false
//
// Disabling verification re-enables active MITM attacks, so this is critical.
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

// insecureTLSPyVerify matches a Python `verify=False` only when an HTTP-client
// context is present on the same line (requests/httpx/session receivers, or one
// of the common request-method calls). This avoids matching unrelated keyword
// arguments such as pandas read_csv(verify=False) or a marshmallow Bool field.
var (
	insecureTLSPyVerify  = regexp.MustCompile(`(?i)\bverify\s*=\s*False\b`)
	insecureTLSPyHTTPCtx = regexp.MustCompile(`(?i)\b(?:requests|httpx|session)\s*\.|\.(?:get|post|put|request)\s*\(`)
)

func (InsecureTLSVerify) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripSlashComment(stripPyComment(sc.Text()))
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

// stripSlashComment removes a trailing `//` line comment that is not inside a
// quoted string. Block comments are left intact (rare on a single matched line)
// and `://` URL separators are preserved because the function only treats `//`
// as a comment when it is preceded by something other than `:`. Shared by the
// security package's C-family rules.
func stripSlashComment(s string) string {
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
