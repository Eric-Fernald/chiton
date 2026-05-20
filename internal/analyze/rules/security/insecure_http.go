package security

import (
	"bufio"
	"bytes"
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
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		match := insecureHTTPRe.FindString(sc.Text())
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

// isTestFile is the heuristic that covers Go (_test.go), JS/TS (.test., .spec.),
// and Python (test_*.py / *_test.py) test files. Centralised here so multiple
// rules can call it.
func isTestFile(path string) bool {
	lp := strings.ToLower(path)
	switch {
	case strings.HasSuffix(lp, "_test.go"):
		return true
	case strings.Contains(lp, ".test."), strings.Contains(lp, ".spec."):
		return true
	case strings.Contains(lp, "/tests/"), strings.Contains(lp, "/test/"):
		return true
	case strings.HasPrefix(filenameOf(lp), "test_"), strings.HasSuffix(lp, "_test.py"):
		return true
	}
	return false
}

func filenameOf(p string) string {
	if i := strings.LastIndexAny(p, "/\\"); i >= 0 {
		return p[i+1:]
	}
	return p
}

func init() { analyze.Register(InsecureHTTP{}) }
