// Package security contains deterministic rules in the security category.
package security

import (
	"context"
	"fmt"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// HardcodedSecret flags lines that look like inline credentials, using only vendor-specific
// token and key patterns so the rule produces few false positives.
type HardcodedSecret struct{}

func (HardcodedSecret) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/hardcoded-secret",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityCritical,
		Scope:      analyze.FileScope,
		Summary:    "Inline credential pattern detected (AWS key, JWT, bearer token, etc.)",
		Fix:        "Move the value to an environment variable or secret manager.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// Patterns that are specific enough to be reliable signals.
// Each pattern's name is reported in the finding message.
var secretPatterns = []struct {
	name string
	re   *regexp.Regexp
}{
	{"aws-access-key", regexp.MustCompile(`AKIA[0-9A-Z]{16}`)},
	{"aws-secret-key", regexp.MustCompile(`(?i)aws[_-]?secret[_-]?(access[_-]?)?key[ \t]*[:=][ \t]*["']?[A-Za-z0-9/+=]{40}`)},
	{"google-api-key", regexp.MustCompile(`AIza[0-9A-Za-z\-_]{35}`)},
	{"github-token", regexp.MustCompile(`gh[pousr]_[A-Za-z0-9]{36,}`)},
	{"slack-token", regexp.MustCompile(`xox[baprs]-[A-Za-z0-9-]{10,}`)},
	{"private-key-block", regexp.MustCompile(`-----BEGIN (RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----`)},
	{"jwt", regexp.MustCompile(`eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}`)},
}

func (HardcodedSecret) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	// Test files routinely carry fixture keys (this repo's own analyzer tests
	// included); a critical there is pure noise.
	if analyze.IsTestSource(t.URI) {
		return nil
	}
	var out []analyze.Finding
	for line, text := range analyze.FileLines(t) {
		// aws-secret-key is (?i) but still needs its literal ':' or '='.
		if !containsAny(text, "AKIA", ":", "=", "AIza", "gh", "xox", "-----BEGIN ", "eyJ") {
			continue
		}
		for _, p := range secretPatterns {
			if p.re.MatchString(text) {
				out = append(out, analyze.Finding{
					RuleID:       "security/hardcoded-secret",
					Severity:     analyze.SeverityCritical,
					Category:     analyze.CategorySecurity,
					Message:      fmt.Sprintf("possible %s in source", p.name),
					Fix:          "Move the credential to an environment variable or secret manager.",
					TargetURI:    t.URI,
					TargetNodeID: t.FileNode.ID,
					Span:         &model.SourceSpan{StartLine: line, EndLine: line},
				})
				break // one finding per line
			}
		}
	}
	return out
}

func init() { analyze.Register(HardcodedSecret{}) }
