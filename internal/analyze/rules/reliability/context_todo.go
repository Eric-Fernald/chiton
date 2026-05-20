package reliability

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// ContextTODO flags non-test Go code that calls `context.TODO()` or
// `context.Background()` outside of `main` setup. These are deadline-less
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
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		if contextTodoRe.MatchString(sc.Text()) {
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

func isGoMainFile(path string) bool {
	// We only know this file's path, not its package clause. Approximate by
	// path basename — "main.go" is a near-perfect signal in real codebases.
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' || path[i] == '\\' {
			return path[i+1:] == "main.go"
		}
	}
	return path == "main.go"
}

func init() { analyze.Register(ContextTODO{}) }
