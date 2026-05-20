package security

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// UnsafeDeserialize flags use of Python pickle.loads / cPickle.loads / yaml.load
// (without SafeLoader), and JavaScript's `eval`-style deserialization via
// `Function(...)`/`new Function(...)`. These are common RCE vectors.
type UnsafeDeserialize struct{}

func (UnsafeDeserialize) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/unsafe-deserialize",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityCritical,
		Scope:      analyze.FileScope,
		Languages:  []string{"py", "js", "ts", "jsx", "tsx"},
		Summary:    "Unsafe deserialization API used (pickle.loads, yaml.load without safe loader, Function(...)).",
		Fix:        "Use a safe loader (yaml.safe_load), JSON, or a vetted deserialisation library.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var unsafeDeserPatterns = []*regexp.Regexp{
	regexp.MustCompile(`\b(c)?[Pp]ickle\.loads?\s*\(`),
	regexp.MustCompile(`\byaml\.load\s*\(\s*[^,)]+\)`), // yaml.load without explicit Loader=
	regexp.MustCompile(`\bnew\s+Function\s*\(`),
	regexp.MustCompile(`(^|[^.\w])Function\s*\(\s*["'\x60]`),
}

func (UnsafeDeserialize) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
		for _, re := range unsafeDeserPatterns {
			if re.MatchString(text) {
				out = append(out, analyze.Finding{
					RuleID:       "security/unsafe-deserialize",
					Severity:     analyze.SeverityCritical,
					Category:     analyze.CategorySecurity,
					Message:      "unsafe deserialization API",
					Fix:          "Use safe_load / JSON / explicit Loader= instead.",
					TargetURI:    t.URI,
					TargetNodeID: t.FileNode.ID,
					Span:         &model.SourceSpan{StartLine: line, EndLine: line},
				})
				break
			}
		}
	}
	return out
}

func init() { analyze.Register(UnsafeDeserialize{}) }
