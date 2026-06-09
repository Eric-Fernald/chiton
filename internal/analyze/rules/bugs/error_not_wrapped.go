package bugs

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// ErrorNotWrapped flags `fmt.Errorf(...)` calls that interpolate an error with
// `%v` or `%s` and pass an `err`-named argument, instead of using `%w`. `%w`
// wraps the error so errors.Is / errors.As can unwrap it; `%v`/`%s` flatten it
// to a string and break the error chain.
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
	// fmt.Errorf(...) whose format verbs contain a %v or %s (not %w / %+v).
	errorfCallRe = regexp.MustCompile(`\bfmt\.Errorf\s*\(`)
	// A %v or %s verb that is NOT %w. The trailing arg list must reference an
	// err-named identifier so we only flag actual error interpolation.
	errVerbRe = regexp.MustCompile(`%[+#]?[vs]`)
	errArgRe  = regexp.MustCompile(`,\s*[\w.]*[eE]rr\b`)
)

func (ErrorNotWrapped) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripGoComment(sc.Text())
		if !errorfCallRe.MatchString(text) {
			continue
		}
		// Must format an error (err arg present) using %v/%s but no %w.
		if errVerbRe.MatchString(text) && errArgRe.MatchString(text) && !containsWrapVerb(text) {
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

var wrapVerbRe = regexp.MustCompile(`%[+#]?w`)

func containsWrapVerb(s string) bool { return wrapVerbRe.MatchString(s) }

func init() { analyze.Register(ErrorNotWrapped{}) }
