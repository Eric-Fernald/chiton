// Package architecture contains graph-scope rules over the import topology.
package architecture

import (
	"context"
	"maps"
	"path"
	"slices"
	"sort"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// CircularImport detects cycles in the IMPORTS graph. Extractors point every import at an external package
// node, so the rule resolves relative imports back to the scanned file they name; other external targets
// cannot close a cycle.
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
	files := indexFilesByPath(t.Nodes)
	adj := map[string][]string{}
	for _, e := range t.Edges {
		if e.Kind != model.EdgeImports {
			continue
		}
		src, dst := nodes[e.Src], nodes[e.Dst]
		if src == nil || dst == nil {
			continue // both endpoints must be ingested nodes
		}
		target := e.Dst
		if model.IsExternal(*dst) {
			id, ok := files.resolve(src.SourceURI, dst.Label)
			if !ok {
				continue
			}
			target = id
		}
		if target != e.Src {
			adj[e.Src] = append(adj[e.Src], target)
		}
	}

	var out []analyze.Finding
	for _, cyc := range findCycles(adj) {
		labels := make([]string, 0, len(cyc))
		var firstURI string
		var firstNode string
		for _, id := range cyc {
			n := nodes[id] // never nil, because adj only holds edges whose endpoints are indexed
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

// fileIndex maps each scanned file's slash path (its file:// URI minus the scheme) to its node ID.
type fileIndex map[string]string

func indexFilesByPath(ns []model.Node) fileIndex {
	idx := fileIndex{}
	for _, n := range ns {
		if p, ok := strings.CutPrefix(n.SourceURI, "file://"); ok && n.Kind == model.KindFile {
			idx[p] = n.ID
		}
	}
	return idx
}

// jsImportExts are the extensions, in resolution order, a JS/TS relative import may omit.
var jsImportExts = []string{".ts", ".tsx", ".js", ".jsx"}

// resolve maps an import spec written in the file fromURI to the scanned file it names. It handles relative
// imports only: JS/TS "./x" and "../x" (with an omitted extension or index file) and Python ".x" / "..x.y".
// Go has none, and the Go toolchain already rejects import cycles.
func (idx fileIndex) resolve(fromURI, spec string) (string, bool) {
	from, ok := strings.CutPrefix(fromURI, "file://")
	if !ok {
		return "", false
	}
	dir := path.Dir(from)
	var candidates []string
	switch ext := analyze.FileExt(from); {
	case slices.Contains([]string{"js", "jsx", "ts", "tsx"}, ext) && (strings.HasPrefix(spec, "./") || strings.HasPrefix(spec, "../")):
		base := path.Join(dir, spec)
		candidates = append(candidates, base)
		for _, e := range jsImportExts {
			candidates = append(candidates, base+e, base+"/index"+e)
		}
	case ext == "py" && strings.HasPrefix(spec, "."):
		rest := strings.TrimLeft(spec, ".")
		for range len(spec) - len(rest) - 1 { // one dot is this package; each further dot goes up one
			dir = path.Dir(dir)
		}
		if rest == "" {
			candidates = append(candidates, dir+"/__init__.py")
		} else {
			base := path.Join(dir, strings.ReplaceAll(rest, ".", "/"))
			candidates = append(candidates, base+".py", base+"/__init__.py")
		}
	}
	for _, c := range candidates {
		if id, ok := idx[c]; ok {
			return id, true
		}
	}
	return "", false
}

func indexNodesByID(ns []model.Node) map[string]*model.Node {
	m := make(map[string]*model.Node, len(ns))
	for i := range ns {
		m[ns[i].ID] = &ns[i]
	}
	return m
}

// findCycles uses Tarjan's algorithm to return the members of each strongly-connected component in adj
// with more than one node. Members and components are sorted by ID, so the output is stable across runs.
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

	for _, v := range slices.Sorted(maps.Keys(adj)) {
		if _, ok := indices[v]; !ok {
			strongconnect(v)
		}
	}
	// Components are disjoint, so their smallest (first) IDs differ.
	slices.SortFunc(sccs, func(a, b []string) int { return strings.Compare(a[0], b[0]) })
	return sccs
}

func joinArrows(labels []string) string {
	if len(labels) == 0 {
		return ""
	}
	return strings.Join(labels, " → ") + " → " + labels[0]
}

func init() { analyze.Register(CircularImport{}) }
