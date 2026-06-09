package reliability

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// MissingHTTPTimeout flags HTTP calls that use a client with no timeout, so a
// hung server can block the caller indefinitely:
//   - Go:     http.Get / http.Post / http.PostForm / http.Head use the default
//     http.DefaultClient, which has no Timeout set.
//   - Python: requests.get/post/put/delete/patch/head(...) with no timeout=
//     keyword argument (requests defaults to waiting forever).
type MissingHTTPTimeout struct{}

func (MissingHTTPTimeout) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "reliability/missing-http-timeout",
		Category:   analyze.CategoryReliability,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py"},
		Summary:    "HTTP call with no timeout can hang forever on an unresponsive peer.",
		Fix:        "Use a client with a Timeout (Go) or pass timeout= (Python requests).",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// Go: package-level http helper functions that use the default (no-timeout) client.
var goHTTPDefaultRe = regexp.MustCompile(`\bhttp\.(Get|Post|PostForm|Head)\s*\(`)

// Python: requests module verb calls. We capture the call so we can inspect its
// argument list for a timeout= keyword.
var pyRequestsCallRe = regexp.MustCompile(`\brequests\.(get|post|put|delete|patch|head|options)\s*\(`)

// timeout= keyword anywhere in a python call's argument span.
var pyTimeoutKwRe = regexp.MustCompile(`\btimeout\s*=`)

func (MissingHTTPTimeout) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	switch langExt(t.Path) {
	case "go":
		return goHTTPTimeout(t)
	case "py":
		return pyHTTPTimeout(t)
	}
	return nil
}

func goHTTPTimeout(t analyze.FileTarget) []analyze.Finding {
	// Per-file masking removes http.Get/Post tokens inside string literals and
	// //, /* */ (incl. multi-line) comments, so commented-out code is not flagged.
	masked := maskGoLiteralsAndComments(t.Content)
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(masked))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
		if goHTTPDefaultRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "reliability/missing-http-timeout",
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryReliability,
				Message:      "http.Get/Post/Head uses the default client with no timeout",
				Fix:          "Use an http.Client with a Timeout (or a request context deadline).",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func pyHTTPTimeout(t analyze.FileTarget) []analyze.Finding {
	// Strip # comments line-by-line, then join so a call's args can span lines.
	stripped := stripPyComments(t.Content)
	var out []analyze.Finding
	for _, loc := range pyRequestsCallRe.FindAllIndex(stripped, -1) {
		// loc[1] is just past the opening '('. Find the matching close paren so
		// we only inspect THIS call's argument list (nested parens balanced).
		end := matchParen(stripped, loc[1]-1)
		if end < 0 {
			end = len(stripped) // unbalanced — inspect to EOF, conservatively
		}
		args := stripped[loc[1]:end]
		if pyTimeoutKwRe.Match(args) {
			continue // has an explicit timeout
		}
		line := 1
		for _, b := range stripped[:loc[0]] {
			if b == '\n' {
				line++
			}
		}
		out = append(out, analyze.Finding{
			RuleID:       "reliability/missing-http-timeout",
			Severity:     analyze.SeverityMedium,
			Category:     analyze.CategoryReliability,
			Message:      "requests call has no timeout= and can block forever",
			Fix:          "Pass timeout= to the requests call.",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		})
	}
	return out
}

// matchParen returns the index of the ')' that closes the '(' at openIdx, or -1
// if unbalanced. Honours single/double/triple python string literals so parens
// inside strings don't throw off the count.
func matchParen(src []byte, openIdx int) int {
	depth := 0
	for i := openIdx; i < len(src); i++ {
		c := src[i]
		switch c {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return i
			}
		case '\'', '"':
			i = skipPyString(src, i)
		}
	}
	return -1
}

// skipPyString returns the index of the closing quote of a python string that
// starts at i (handling triple-quoted strings and backslash escapes). On a
// malformed/unterminated string it returns the last index so the caller stops.
func skipPyString(src []byte, i int) int {
	q := src[i]
	if i+2 < len(src) && src[i+1] == q && src[i+2] == q {
		// Triple-quoted string: scan for the closing triple quote.
		for j := i + 3; j+2 < len(src); j++ {
			if src[j] == '\\' {
				j++
				continue
			}
			if src[j] == q && src[j+1] == q && src[j+2] == q {
				return j + 2
			}
		}
		return len(src) - 1
	}
	// Single- or double-quoted string: ends at the matching quote or newline.
	for j := i + 1; j < len(src); j++ {
		if src[j] == '\\' {
			j++
			continue
		}
		if src[j] == q || src[j] == '\n' {
			return j
		}
	}
	return len(src) - 1
}

// stripPyComments blanks out `#` line comments (outside string literals),
// preserving newlines, so the multi-line call scanner sees only code.
func stripPyComments(src []byte) []byte {
	out := make([]byte, len(src))
	copy(out, src)
	inStr := byte(0)
	for i := 0; i < len(out); i++ {
		c := out[i]
		switch {
		case inStr != 0:
			if c == '\\' {
				i++
				continue
			}
			if c == inStr {
				inStr = 0
			}
		default:
			switch c {
			case '#':
				for i < len(out) && out[i] != '\n' {
					out[i] = ' '
					i++
				}
			case '\'', '"':
				inStr = c
			}
		}
	}
	return out
}

func init() { analyze.Register(MissingHTTPTimeout{}) }
