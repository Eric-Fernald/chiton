// Package architecture contains graph-scope rules over the import topology.
package architecture

import (
	"context"
	"sort"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// CircularImport detects cycles in the IMPORTS graph; edges into external package nodes
// (external=true) are ignored since they cannot close a cycle.
type CircularImport struct{}

func (CircularImport) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "architecture/circular-import",
		Category:   analyze.CategoryArchitecture,
		Severity:   analyze.SeverityHigh,
		Scope:      analyze.GraphScope,
		Summary:    "Two or more files form an import cycle.",
		Fix:        "Break the cycle by extracting a shared interface to a third module.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

func (CircularImport) MatchGraph(_ context.Context, t analyze.GraphTarget) []analyze.Finding {
	nodes := indexNodesByID(t.Nodes)
	adj := map[string][]string{}
	for _, e := range t.Edges {
		if e.Kind != model.EdgeImports {
			continue
		}
		src, dst := nodes[e.Src], nodes[e.Dst]
		// Both endpoints must be ingested nodes, and external packages cannot close a cycle.
		if src == nil || dst == nil {
			continue
		}
		if isExternal(*dst) {
			continue
		}
		adj[e.Src] = append(adj[e.Src], e.Dst)
	}

	var out []analyze.Finding
	for _, cyc := range findCycles(adj) {
		labels := make([]string, 0, len(cyc))
		var firstURI string
		var firstNode string
		for _, id := range cyc {
			n := nodes[id] // never nil: adj only holds edges whose endpoints are indexed
			labels = append(labels, n.Label)
			if firstURI == "" {
				firstURI = n.SourceURI
				firstNode = n.ID
			}
		}
		out = append(out, analyze.Finding{
			RuleID:       "architecture/circular-import",
			Severity:     analyze.SeverityHigh,
			Category:     analyze.CategoryArchitecture,
			Message:      "import cycle: " + joinArrows(labels),
			Fix:          "Break the cycle by extracting a shared interface.",
			TargetURI:    firstURI,
			TargetNodeID: firstNode,
		})
	}
	return out
}

func isExternal(n model.Node) bool {
	if v, ok := n.Attrs["external"].(bool); ok {
		return v
	}
	return false
}

func indexNodesByID(ns []model.Node) map[string]*model.Node {
	m := make(map[string]*model.Node, len(ns))
	for i := range ns {
		m[ns[i].ID] = &ns[i]
	}
	return m
}

// findCycles returns the members of each strongly-connected component of size > 1 in adj (Tarjan),
// sorted by ID so the reported cycle is stable across runs.
func findCycles(adj map[string][]string) [][]string {
	index := 0
	indices := map[string]int{}
	lowlink := map[string]int{}
	onStack := map[string]bool{}
	var stack []string
	var sccs [][]string

	var strongconnect func(v string)
	strongconnect = func(v string) {
		indices[v] = index
		lowlink[v] = index
		index++
		stack = append(stack, v)
		onStack[v] = true

		for _, w := range adj[v] {
			if _, ok := indices[w]; !ok {
				strongconnect(w)
				if lowlink[w] < lowlink[v] {
					lowlink[v] = lowlink[w]
				}
			} else if onStack[w] {
				if indices[w] < lowlink[v] {
					lowlink[v] = indices[w]
				}
			}
		}
		if lowlink[v] == indices[v] {
			var comp []string
			for {
				w := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				onStack[w] = false
				comp = append(comp, w)
				if w == v {
					break
				}
			}
			if len(comp) > 1 {
				sort.Strings(comp)
				sccs = append(sccs, comp)
			}
		}
	}

	for v := range adj {
		if _, ok := indices[v]; !ok {
			strongconnect(v)
		}
	}
	return sccs
}

func joinArrows(labels []string) string {
	if len(labels) == 0 {
		return ""
	}
	return strings.Join(labels, " → ") + " → " + labels[0]
}

func init() { analyze.Register(CircularImport{}) }
