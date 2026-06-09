package reliability

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// EmptyCatch flags an empty catch block in C-family languages (JS/TS/Java):
// `catch (e) {}` or `catch {}`. An empty catch silently swallows the exception,
// hiding failures from logs, metrics, and callers. It scans the whole file
// content (not line-by-line) so the `{}` may sit on the next line after the
// `catch (...)` clause.
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

// Matches `catch (e) { }` or `catch { }` where the body is only whitespace.
// `[\s\S]*?` is avoided: we require the braces to enclose only whitespace so
// `catch (e) { doX(); }` and `catch (e) { /* comment */ }` are NOT flagged.
var emptyCatchRe = regexp.MustCompile(`\bcatch\b\s*(\([^)]*\)\s*)?\{\s*\}`)

func (EmptyCatch) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	// Mask string/char/template literals AND comment bodies with a filler byte
	// (preserving newlines). Masking string contents stops `"catch (e) {}"` in a
	// string from firing; masking comment bodies with a NON-whitespace filler
	// means a `catch (e) { /* intentional */ }` block is no longer "empty", so an
	// explicitly-documented ignore is not flagged — only truly empty braces are.
	content := maskLiteralsAndComments(t.Content)

	var out []analyze.Finding
	for _, loc := range emptyCatchRe.FindAllIndex(content, -1) {
		line := 1
		for _, b := range content[:loc[0]] {
			if b == '\n' {
				line++
			}
		}
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

// maskLiteralsAndComments returns a copy of src with the bodies of string,
// char, and template literals plus // and /* */ comments replaced by the filler
// byte 'x' (newlines preserved so line numbers stay accurate). The non-blank
// filler is deliberate: a comment inside catch-braces should make the body
// look NON-empty, so documented intentional ignores are not flagged. The
// delimiters themselves (quotes, comment markers) are also overwritten so they
// can never be re-interpreted by the catch regex.
func maskLiteralsAndComments(src []byte) []byte {
	out := make([]byte, len(src))
	copy(out, src)
	const filler = 'x'
	inLine := false
	inBlock := false
	inStr := byte(0) // 0, '"', '\'', or '`'
	for i := 0; i < len(out); i++ {
		c := out[i]
		switch {
		case inLine:
			if c == '\n' {
				inLine = false
			} else {
				out[i] = filler
			}
		case inBlock:
			if c == '*' && i+1 < len(out) && out[i+1] == '/' {
				out[i] = filler
				out[i+1] = filler
				i++
				inBlock = false
			} else if c != '\n' {
				out[i] = filler
			}
		case inStr != 0:
			if c == '\\' {
				if i+1 < len(out) && out[i+1] != '\n' {
					out[i] = filler
					out[i+1] = filler
				}
				i++ // skip escaped char
				continue
			}
			if c == inStr {
				inStr = 0
				out[i] = filler
			} else if c != '\n' {
				out[i] = filler
			}
		default:
			switch {
			case c == '/' && i+1 < len(out) && out[i+1] == '/':
				inLine = true
				out[i] = filler
			case c == '/' && i+1 < len(out) && out[i+1] == '*':
				inBlock = true
				out[i] = filler
			case c == '"' || c == '\'' || c == '`':
				inStr = c
				out[i] = filler
			}
		}
	}
	return out
}

func init() { analyze.Register(EmptyCatch{}) }
