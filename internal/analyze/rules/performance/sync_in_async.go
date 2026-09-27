package performance

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// SyncInAsync flags blocking calls (requests.get/post/put/delete, time.sleep, urlopen) inside a
// Python `async def`, where they stall the event loop.
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
	asyncIndent := -1
	for line, text := range analyze.FileLines(t) {
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
