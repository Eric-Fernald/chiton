package reliability

import (
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// ContextTODO flags `context.TODO()` / `context.Background()` in non-test, non-main Go code: deadline-less
// contexts that suppress cancellation and timeout propagation.
type ContextTODO struct{}

func (ContextTODO) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "reliability/context-without-deadline",
		Category:   analyze.CategoryReliability,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go"},
		Summary:    "context.TODO() / context.Background() used outside top-level setup.",
		Fix:        "Accept ctx from the caller, or wrap with context.WithTimeout / WithCancel.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var contextTodoRe = regexp.MustCompile(`\bcontext\.(TODO|Background)\s*\(\s*\)`)

func (ContextTODO) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	if isGoTestFile(t.Path) || isGoMainFile(t.Path) {
		return nil
	}
	if !bytes.Contains(t.Content, []byte("context.")) {
		return nil
	}
	var out []analyze.Finding
	for line, text := range analyze.FileLines(t) {
		if contextTodoRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "reliability/context-without-deadline",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryReliability,
				Message:      "context.TODO/Background used outside top-level setup",
				Fix:          "Accept ctx from the caller or wrap with WithTimeout/WithCancel.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func isGoTestFile(path string) bool {
	return len(path) > 8 && path[len(path)-8:] == "_test.go"
}

// isGoMainFile approximates the package clause by basename alone; "main.go" is a near-perfect signal.
func isGoMainFile(path string) bool {
	return analyze.PathBase(path) == "main.go"
}

func init() { analyze.Register(ContextTODO{}) }
