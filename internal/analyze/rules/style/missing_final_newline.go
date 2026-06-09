package style

import (
	"bytes"
	"context"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// MissingFinalNewline flags files whose last line is not terminated by a
// newline. POSIX text files should end with one; the absence breaks `cat`,
// produces "\ No newline at end of file" diffs, and trips many linters.
type MissingFinalNewline struct{}

func (MissingFinalNewline) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "style/missing-final-newline",
		Category:   analyze.CategoryStyle,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "rs", "java"},
		Summary:    "File does not end with a newline.",
		Fix:        "Add a trailing newline at the end of the file.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (MissingFinalNewline) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	// Empty files have no content to terminate; don't flag them.
	if len(t.Content) == 0 {
		return nil
	}
	last := t.Content[len(t.Content)-1]
	if last == '\n' || last == '\r' {
		return nil
	}
	// Report the span on the final (unterminated) line.
	lineCount := bytes.Count(t.Content, []byte{'\n'}) + 1
	return []analyze.Finding{{
		RuleID:       "style/missing-final-newline",
		Severity:     analyze.SeverityLow,
		Category:     analyze.CategoryStyle,
		Message:      "file does not end with a newline",
		Fix:          "Add a trailing newline at the end of the file.",
		TargetURI:    t.URI,
		TargetNodeID: t.FileNode.ID,
		Span:         &model.SourceSpan{StartLine: lineCount, EndLine: lineCount},
	}}
}

func init() { analyze.Register(MissingFinalNewline{}) }
