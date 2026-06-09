package security

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// CORSWildcard flags an Access-Control-Allow-Origin set to "*". A wildcard
// origin lets any site read responses; combined with credentials it is an
// account-takeover vector. Matches both literal header writes and the common
// framework config keys.
type CORSWildcard struct{}

func (CORSWildcard) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/cors-wildcard",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "java"},
		Summary:    "CORS Access-Control-Allow-Origin set to '*' (any origin allowed).",
		Fix:        "Restrict the allowed origin to a specific trusted host list.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var corsWildcardPatterns = []*regexp.Regexp{
	// Literal header name/value pair on one line, e.g.
	//   "Access-Control-Allow-Origin", "*"  or  "Access-Control-Allow-Origin": "*"
	regexp.MustCompile(`(?i)Access-Control-Allow-Origin["']?\s*[,:]\s*["']\*["']`),
	// Framework config keys: origin/origins/allowedOrigins/cors_allowed_origins = "*"
	regexp.MustCompile(`(?i)\b(allowed_?origins?|cors_allowed_origins|origins?)\s*[:=]\s*["']\*["']`),
}

func (CORSWildcard) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripSlashComment(stripPyComment(sc.Text()))
		for _, re := range corsWildcardPatterns {
			if re.MatchString(text) {
				out = append(out, analyze.Finding{
					RuleID:       "security/cors-wildcard",
					Severity:     analyze.SeverityMedium,
					Category:     analyze.CategorySecurity,
					Message:      "CORS allows any origin ('*')",
					Fix:          "Allow only specific trusted origins.",
					TargetURI:    t.URI,
					TargetNodeID: t.FileNode.ID,
					Span:         &model.SourceSpan{StartLine: line, EndLine: line},
				})
				break
			}
		}
	}
	return out
}

func init() { analyze.Register(CORSWildcard{}) }
