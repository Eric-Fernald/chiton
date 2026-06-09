package security

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// BindAllInterfaces flags a service bound to 0.0.0.0, which exposes it on every
// network interface. Often intentional in containers, hence low severity — but
// worth surfacing because it is also a common accidental exposure. The address
// must appear inside a string literal to count, so comments and arithmetic do
// not trip it.
type BindAllInterfaces struct{}

func (BindAllInterfaces) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "security/bind-all-interfaces",
		Category:   analyze.CategorySecurity,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "jsx", "tsx", "java"},
		Summary:    "Service bound to 0.0.0.0 (all network interfaces).",
		Fix:        "Bind to a specific interface (e.g. 127.0.0.1) unless external exposure is intended.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// Match 0.0.0.0 only inside a quoted string (optionally followed by :port),
// e.g. "0.0.0.0", "0.0.0.0:8080", '0.0.0.0'. Avoids matching version strings
// or comments that merely mention the address.
var bindAllRe = regexp.MustCompile(`["']0\.0\.0\.0(:\d+)?["']`)

func (BindAllInterfaces) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	if isTestFile(t.Path) {
		return nil
	}
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripSlashComment(stripPyComment(sc.Text()))
		if bindAllRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "security/bind-all-interfaces",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategorySecurity,
				Message:      "service bound to 0.0.0.0 (all interfaces)",
				Fix:          "Bind to a specific interface unless external exposure is intended.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(BindAllInterfaces{}) }
