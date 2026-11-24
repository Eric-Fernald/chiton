package security

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// InsecureHTTP flags non-test source that hard-codes an http:// URL. Loopback
// (127.0.0.1, localhost) is allowed since those are common in dev setups.
type InsecureHTTP struct{}

func (InsecureHTTP) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/insecure-http",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  analyze.SourceLangs,
		Summary:    "Hard-coded http:// URL — traffic is unencrypted in transit.",
		Fix:        "Switch to https:// or document the loopback/dev-only use.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var insecureHTTPRe = regexp.MustCompile(`\bhttp://[\w.-]+`)

func (InsecureHTTP) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	if isTestFile(t.Path) {
		return nil
	}
	var out []analyze.Finding
	for line, text := range analyze.FileLines(t) {
		if !strings.Contains(text, "http://") {
			continue
		}
		match := insecureHTTPRe.FindString(text)
		if match == "" {
			continue
		}
		host := strings.TrimPrefix(match, "http://")
		if host == "" {
			continue
		}
		if strings.HasPrefix(host, "127.0.0.1") || strings.HasPrefix(host, "localhost") || strings.HasPrefix(host, "0.0.0.0") {
			continue
		}
		out = append(out, analyze.Finding{
			RuleID:       "security/insecure-http",
			Severity:     analyze.SeverityMedium,
			Category:     analyze.CategorySecurity,
			Message:      "non-TLS URL: " + match,
			Fix:          "Use https:// in production code paths.",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		})
	}
	return out
}

// isTestFile detects Go, JS/TS, and Python test files by path. Unlike analyze.IsTestSource it also
// matches test/ directories, test_*.py, and *_test.py.
func isTestFile(path string) bool {
	lp := strings.ToLower(path)
	switch {
	case strings.HasSuffix(lp, "_test.go"):
		return true
	case strings.Contains(lp, ".test."), strings.Contains(lp, ".spec."):
		return true
	case strings.Contains(lp, "/tests/"), strings.Contains(lp, "/test/"):
		return true
	case strings.HasPrefix(analyze.PathBase(lp), "test_"), strings.HasSuffix(lp, "_test.py"):
		return true
	}
	return false
}

// containsAny is a cheap exact pre-check for a regex set in which every pattern requires one of
// lits case-sensitively. It skips the regex work on most lines.
func containsAny(s string, lits ...string) bool {
	for _, l := range lits {
		if strings.Contains(s, l) {
			return true
		}
	}
	return false
}

func init() { analyze.Register(InsecureHTTP{}) }
