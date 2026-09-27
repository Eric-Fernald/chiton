package security

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// CommandInjection flags shell-out calls (shell=True, os.system, child_process.exec, sh -c): they
// run their argument through a shell, so any interpolated variable is an injection vector.
type CommandInjection struct{}

func (CommandInjection) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/command-injection",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.FileScope,
		Languages:  []string{"py", "js", "ts", "jsx", "tsx", "go"},
		Summary:    "Shell execution of a command string (command-injection risk).",
		Fix:        "Pass an argv list without a shell, and never interpolate untrusted input.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var commandInjectionPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)shell\s*=\s*True`),                                                 // Python subprocess shell=True
	regexp.MustCompile(`\bos\.system\s*\(`),                                                    // Python os.system
	regexp.MustCompile(`\bos\.popen\s*\(`),                                                     // Python os.popen
	regexp.MustCompile(`(?:child_process|cp)\s*\.\s*exec\s*\(`),                                // Node exec (not execFile)
	regexp.MustCompile(`(?:child_process|cp)\s*\.\s*execSync\s*\(`),                            // Node execSync (child_process only)
	regexp.MustCompile(`exec\.Command\s*\(\s*["'](?:sh|bash|cmd)["']\s*,\s*["']?(?:-c|/c|/C)`), // Go shell -c
}

func (CommandInjection) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	for line, text := range analyze.FileLines(t) {
		if !containsAny(text, "=", "os.", "exec") {
			continue
		}
		text = stripSlashCommentKeepURL(analyze.StripPyComment(text))
		for _, re := range commandInjectionPatterns {
			if re.MatchString(text) {
				out = append(out, analyze.Finding{
					RuleID:       "security/command-injection",
					Severity:     analyze.SeverityHigh,
					Category:     analyze.CategorySecurity,
					Message:      "shell execution of a command string — command-injection risk",
					Fix:          "Use an argv list without a shell; never interpolate untrusted input.",
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

func init() { analyze.Register(CommandInjection{}) }
