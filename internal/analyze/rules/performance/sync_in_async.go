package performance

import (
	"bufio"
	"bytes"
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// SyncInAsync flags blocking I/O calls inside Python `async def` functions —
// `requests.get`, `time.sleep`, `open(...).read()`, etc. — that defeat the
// whole point of asyncio.
type SyncInAsync struct{}

func (SyncInAsync) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "performance/sync-in-async",
		Category:   analyze.CategoryPerformance,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"py"},
		Summary:    "Synchronous I/O call inside an async function blocks the event loop.",
		Fix:        "Use the awaitable equivalent (aiohttp, asyncio.sleep, aiofiles).",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

var (
	asyncDefRe = regexp.MustCompile(`^\s*async\s+def\s+\w+`)
	syncCallRe = regexp.MustCompile(
		`(?:^|\s)(?:requests\.(?:get|post|put|delete)|time\.sleep|urllib\.request\.urlopen)\s*\(`,
	)
)

func (SyncInAsync) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(t.Content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	asyncIndent := -1
	for sc.Scan() {
		line++
		text := sc.Text()
		trimmed := strings.TrimSpace(text)
		if trimmed == "" {
			continue
		}
		indent := leadingSpaces(text)
		if asyncDefRe.MatchString(text) {
			asyncIndent = indent
			continue
		}
		if asyncIndent < 0 {
			continue
		}
		if indent <= asyncIndent {
			asyncIndent = -1
			if asyncDefRe.MatchString(text) {
				asyncIndent = indent
			}
			continue
		}
		if syncCallRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "performance/sync-in-async",
				Severity:     analyze.SeverityMedium,
				Category:     analyze.CategoryPerformance,
				Message:      "blocking I/O call inside async function",
				Fix:          "Use the asyncio-aware equivalent.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func leadingSpaces(s string) int {
	for i, r := range s {
		if r != ' ' && r != '\t' {
			return i
		}
	}
	return len(s)
}

func init() { analyze.Register(SyncInAsync{}) }
