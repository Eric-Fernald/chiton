package architecture

import (
	"context"
	"fmt"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// GodModuleSymbolThreshold is the CONTAINS-out count above which a file is flagged as a god
// module, calibrated for typical 200-500-line files.
const GodModuleSymbolThreshold = 25

// GodModule flags files containing more than GodModuleSymbolThreshold
// top-level symbols.
type GodModule struct{}

func (GodModule) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "architecture/god-module",
		Category:   analyze.CategoryArchitecture,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.GraphScope,
		Summary:    fmt.Sprintf("File contains more than %d top-level symbols.", GodModuleSymbolThreshold),
		Fix:        "Split responsibilities across smaller, focused files.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (GodModule) MatchGraph(_ context.Context, t analyze.GraphTarget) []analyze.Finding {
	nodes := indexNodesByID(t.Nodes)
	containsOut := map[string]int{}
	for _, e := range t.Edges {
		if e.Kind != model.EdgeContains {
			continue
		}
		src := nodes[e.Src]
		if src == nil || src.Kind != model.KindFile {
			continue
		}
		containsOut[e.Src]++
	}

	var out []analyze.Finding
	for id, count := range containsOut {
		if count <= GodModuleSymbolThreshold {
			continue
		}
		n := nodes[id]
		if n == nil {
			continue
		}
		out = append(out, analyze.Finding{
			RuleID:       "architecture/god-module",
			Severity:     analyze.SeverityMedium,
			Category:     analyze.CategoryArchitecture,
			Message:      fmt.Sprintf("file holds %d top-level symbols (limit %d)", count, GodModuleSymbolThreshold),
			Fix:          "Split into smaller focused files.",
			TargetURI:    n.SourceURI,
			TargetNodeID: n.ID,
		})
	}
	return out
}

func init() { analyze.Register(GodModule{}) }
