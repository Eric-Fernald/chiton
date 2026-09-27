package reliability

import (
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// EmptyCatch flags an empty `catch (e) {}` or `catch {}` in JS/TS and Java, which swallows the exception.
// It scans the whole file, so the `{}` may sit on a line after the `catch (...)` clause.
type EmptyCatch struct{}

func (EmptyCatch) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "reliability/empty-catch",
		Category:   analyze.CategoryReliability,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"js", "ts", "jsx", "tsx", "java"},
		Summary:    "Empty catch block silently swallows the exception.",
		Fix:        "Handle, log, or re-throw the caught error instead of discarding it.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// The body must be pure whitespace, so `catch (e) { doX(); }` and `catch (e) { /* why */ }` (masked to x's)
// are not flagged.
var emptyCatchRe = regexp.MustCompile(`\bcatch\b\s*(\([^)]*\)\s*)?\{\s*\}`)

func (EmptyCatch) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	if !bytes.Contains(t.Content, []byte("catch")) {
		return nil
	}
	// The non-blank 'x' filler keeps a commented-out catch body from looking empty.
	content := analyze.MaskedCLiterals(t, false)

	var out []analyze.Finding
	line, counted := 1, 0
	for _, loc := range emptyCatchRe.FindAllIndex(content, -1) {
		line += bytes.Count(content[counted:loc[0]], []byte{'\n'})
		counted = loc[0]
		out = append(out, analyze.Finding{
			RuleID:       "reliability/empty-catch",
			Severity:     analyze.SeverityMedium,
			Category:     analyze.CategoryReliability,
			Message:      "empty catch block swallows the exception",
			Fix:          "Handle, log, or re-throw the caught error.",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		})
	}
	return out
}

func init() { analyze.Register(EmptyCatch{}) }
