package architecture

import (
	"bytes"
	"context"
	"fmt"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// LargeFileLineThreshold is the line count above which a source file is flagged.
// Set high so only files that clearly warrant a split are reported.
const LargeFileLineThreshold = 800

// LargeFile flags any source file longer than LargeFileLineThreshold lines —
// large files concentrate too many responsibilities and resist review.
type LargeFile struct{}

func (LargeFile) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "architecture/large-file",
		Category:   analyze.CategoryArchitecture,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "rs", "java"},
		Summary:    fmt.Sprintf("Source file is longer than %d lines.", LargeFileLineThreshold),
		Fix:        "Split the file into smaller, cohesive units.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (LargeFile) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	lines := countLines(t.Content)
	if lines <= LargeFileLineThreshold {
		return nil
	}
	return []analyze.Finding{{
		RuleID:       "architecture/large-file",
		Severity:     analyze.SeverityLow,
		Category:     analyze.CategoryArchitecture,
		Message:      fmt.Sprintf("file is %d lines (limit %d)", lines, LargeFileLineThreshold),
		Fix:          "Split the file into smaller, cohesive units.",
		TargetURI:    t.URI,
		TargetNodeID: t.FileNode.ID,
		Span:         &model.SourceSpan{StartLine: 1, EndLine: 1},
	}}
}

// countLines counts newline-delimited lines, counting a final non-empty line
// without a trailing newline.
func countLines(content []byte) int {
	if len(content) == 0 {
		return 0
	}
	n := bytes.Count(content, []byte{'\n'})
	if content[len(content)-1] != '\n' {
		n++
	}
	return n
}

func init() { analyze.Register(LargeFile{}) }
