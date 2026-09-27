package security

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// WeakHash flags MD5/SHA-1, broken for any security purpose; medium severity because
// non-security uses (e.g. cache keys) exist.
type WeakHash struct{}

func (WeakHash) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/weak-hash",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "java"},
		Summary:    "Weak hash (MD5/SHA-1) used — unsuitable for security.",
		Fix:        "Use SHA-256 or stronger (bcrypt/scrypt/argon2 for passwords).",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var weakHashPatterns = []*regexp.Regexp{
	regexp.MustCompile(`\b(md5|sha1)\.New\s*\(`),                                     // Go: md5.New / sha1.New
	regexp.MustCompile(`\b(md5|sha1)\.Sum\s*\(`),                                     // Go: md5.Sum / sha1.Sum
	regexp.MustCompile(`\bhashlib\.(md5|sha1)\s*\(`),                                 // Python
	regexp.MustCompile(`(?i)createHash\s*\(\s*["'](md5|sha1)["']`),                   // Node crypto
	regexp.MustCompile(`(?i)MessageDigest\.getInstance\s*\(\s*["'](MD5|SHA-?1)["']`), // Java
}

func (WeakHash) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, text := range analyze.FileLines(t) {
		// The (?i) patterns still need a literal digit before the closing quote.
		if !containsAny(text, "md5", "sha1", `5"`, `5'`, `1"`, `1'`) {
			continue
		}
		text = stripSlashCommentKeepURL(analyze.StripPyComment(text))
		for _, re := range weakHashPatterns {
			if re.MatchString(text) {
				out = append(out, analyze.Finding{
					RuleID:       "security/weak-hash",
					Severity:     analyze.SeverityMedium,
					Category:     analyze.CategorySecurity,
					Message:      "weak hash (MD5/SHA-1) is unsuitable for security",
					Fix:          "Use SHA-256+ (or bcrypt/scrypt/argon2 for passwords).",
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

func init() { analyze.Register(WeakHash{}) }
