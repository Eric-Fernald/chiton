package bugs

import (
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// ErrorNotWrapped flags `fmt.Errorf` calls that format an `err`-named argument with `%v` or `%s`, not
// `%w`. That flattens the error to a string, so errors.Is and errors.As can no longer unwrap it.
type ErrorNotWrapped struct{}

func (ErrorNotWrapped) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "bugs/error-not-wrapped",
		Category:   analyze.CategoryBugs,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go"},
		Summary:    "fmt.Errorf formats an error with %v/%s instead of %w; the error chain is lost.",
		Fix:        "Use %w to wrap the error so errors.Is / errors.As can unwrap it.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var (
	errorfCallRe = regexp.MustCompile(`\bfmt\.Errorf\s*\(`)
	errVerbRe    = regexp.MustCompile(`%[+#]?[vs]`)
	// Requiring an err-named argument limits the rule to actual error interpolation.
	errArgRe   = regexp.MustCompile(`,\s*[\w.]*[eE]rr\b`)
	wrapVerbRe = regexp.MustCompile(`%[+#]?w`)
)

func (ErrorNotWrapped) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	if !bytes.Contains(t.Content, []byte("fmt.Errorf")) {
		return nil
	}
	var out []analyze.Finding
	for line, raw := range analyze.FileLines(t) {
		// StripLineComment keeps string contents, because the format verbs live inside the format string.
		text := analyze.StripLineComment(raw, false)
		if !errorfCallRe.MatchString(text) {
			continue
		}
		// Must format an error (err arg present) using %v/%s but no %w.
		if errVerbRe.MatchString(text) && errArgRe.MatchString(text) && !wrapVerbRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "bugs/error-not-wrapped",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryBugs,
				Message:      "fmt.Errorf uses %v/%s for an error; use %w to preserve the chain",
				Fix:          "Replace %v/%s with %w to wrap the error.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(ErrorNotWrapped{}) }
