package security

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DebugModeEnabled flags web-framework debug mode left on outside tests; in Flask the Werkzeug
// debugger console is effectively RCE.
type DebugModeEnabled struct{}

func (DebugModeEnabled) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/debug-mode-enabled",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"py", "js", "ts", "jsx", "tsx"},
		Summary:    "Debug mode enabled (debug=True / DEBUG = True).",
		Fix:        "Disable debug mode in production; drive it from an environment variable.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var debugModePatterns = []*regexp.Regexp{
	regexp.MustCompile(`\bdebug\s*=\s*True\b`),       // Flask app.run(debug=True), config
	regexp.MustCompile(`\bDEBUG\s*=\s*True\b`),       // Django settings.DEBUG = True
	regexp.MustCompile(`["']debug["']\s*:\s*true\b`), // JS config object {"debug": true}
}

func (DebugModeEnabled) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	if isTestFile(t.Path) {
		return nil
	}
	var out []analyze.Finding
	for line, text := range analyze.Lines(t.Content) {
		if !containsAny(text, "debug", "DEBUG") {
			continue
		}
		text = stripSlashCommentKeepURL(analyze.StripPyComment(text))
		for _, re := range debugModePatterns {
			if re.MatchString(text) {
				out = append(out, analyze.Finding{
					RuleID:       "security/debug-mode-enabled",
					Severity:     analyze.SeverityMedium,
					Category:     analyze.CategorySecurity,
					Message:      "debug mode enabled — disable in production",
					Fix:          "Set debug off in production (drive from an env var).",
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

func init() { analyze.Register(DebugModeEnabled{}) }
