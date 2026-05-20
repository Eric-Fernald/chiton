package bugs

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// UncheckedError flags Go lines that assign an error to the blank identifier:
// `_, err := ...` is fine, but `_ = thingThatReturnsError()` swallows the
// error and is the most common silent-failure pattern in Go.
type UncheckedError struct{}

func (UncheckedError) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "bugs/unchecked-error",
		Category:   analyze.CategoryBugs,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go"},
		Summary:    "Error from a call assigned to blank identifier (`_ = foo()`).",
		Fix:        "Capture the error, handle it, or document why it can be ignored.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var uncheckedErrorRe = regexp.MustCompile(`(^|\s)_\s*[:=]=\s*\w[\w.]*\s*\(`)

func (UncheckedError) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripGoComment(sc.Text())
		if uncheckedErrorRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "bugs/unchecked-error",
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryBugs,
				Message:      "error possibly swallowed via blank identifier",
				Fix:          "Replace `_ =` with proper error handling.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(UncheckedError{}) }
