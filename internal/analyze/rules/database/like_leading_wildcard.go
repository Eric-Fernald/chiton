package database

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// LikeLeadingWildcard flags a literal `LIKE '%...'` pattern, which defeats B-tree indexes;
// prefix patterns (`LIKE 'foo%'`) and parameterized `LIKE ?` are left alone.
type LikeLeadingWildcard struct{}

func (LikeLeadingWildcard) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "database/like-leading-wildcard",
		Category:   analyze.CategoryDatabase,
		Severity:   analyze.SeverityLow,
		Scope:      analyze.FileScope,
		Languages:  []string{"go", "py", "js", "ts", "java"},
		Summary:    "LIKE pattern with a leading wildcard ('%...') prevents index use.",
		Fix:        "Anchor the pattern ('foo%'), or use a full-text/trigram index for substring search.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// LIKE '%...'  or  LIKE "%..."  — the % immediately follows the opening quote.
var likeLeadingWildcardRe = regexp.MustCompile(`(?i)\bLIKE\s+(['"])%`)

func (LikeLeadingWildcard) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	var out []analyze.Finding
	py := analyze.FileExt(t.URI) == "py"
	for line, raw := range analyze.LinesOf(maskedDBText(t, py)) {
		text := stripDBComment(raw, py)
		if likeLeadingWildcardRe.MatchString(text) {
			out = append(out, analyze.Finding{
				RuleID:       "database/like-leading-wildcard",
				Severity:     analyze.SeverityLow,
				Category:     analyze.CategoryDatabase,
				Message:      "LIKE with leading wildcard cannot use an index",
				Fix:          "Anchor the pattern, or use a full-text/trigram index.",
				TargetURI:    t.URI,
				TargetNodeID: t.FileNode.ID,
				Span:         &model.SourceSpan{StartLine: line, EndLine: line},
			})
		}
	}
	return out
}

func init() { analyze.Register(LikeLeadingWildcard{}) }
