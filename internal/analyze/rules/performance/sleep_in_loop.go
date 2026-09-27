package performance

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// SleepInLoop flags time.Sleep / time.sleep / Thread.sleep inside a loop body, at low severity
// since polling and backoff loops sometimes sleep on purpose.
type SleepInLoop struct{}

func (SleepInLoop) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "performance/sleep-in-loop",
		Category:   analyze.CategoryPerformance,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "java"},
		Summary:    "Sleep call inside a loop body serializes iterations.",
		Fix:        "Prefer event-driven waits, a ticker/scheduler, or explicit backoff with a cap.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// `\btime\.` requires a word boundary before the `time` module so identifiers
// ending in `time` (uptime.sleep) do not match.
var sleepCallRe = regexp.MustCompile(`(?:\btime\.[Ss]leep|Thread\.sleep)\s*\(`)

func (SleepInLoop) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	const ruleID = "performance/sleep-in-loop"
	mk := func(line int) analyze.Finding {
		return analyze.Finding{
			RuleID:       ruleID,
			Severity:     analyze.SeverityLow,
			Category:     analyze.CategoryPerformance,
			Message:      "sleep call inside loop body serializes iterations",
			Fix:          "Use a ticker/scheduler or event-driven wait instead of sleeping in the loop.",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		}
	}
	return scanLoopBody(t.Content, analyze.FileExt(t.Path) == "py", sleepCallRe, mk)
}

func init() { analyze.Register(SleepInLoop{}) }
