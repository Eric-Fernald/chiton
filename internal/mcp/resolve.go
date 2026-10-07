package mcpsrv

import (
	"context"
	"slices"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// resolver joins Go package-level functions to the pkg:// references their callers point at. The extractor
// sees one file at a time, so a call into another file (another package, or a sibling file of the same one)
// targets model.PkgFuncRefID(importPath, name) rather than the definition. The resolver treats the reference
// and the definition as one node, so "who calls X" sees every caller and a walk continues past package
// boundaries. It caches lookups for one request.
type resolver struct {
	gs      store.GraphStore
	files   map[string][]model.Node // import path → its Go file nodes
	fileOf  map[string]*model.Node  // source URI → file node (nil when the URI has none)
	defs    map[string][]model.Node // reference ID → the definitions it names
	refOf   map[string]string       // definition ID → its reference ID ("" when it has none)
	symbols map[string][]model.Node // source URI → the symbols defined there
}

func newResolver(gs store.GraphStore) *resolver {
	return &resolver{
		gs:      gs,
		files:   map[string][]model.Node{},
		fileOf:  map[string]*model.Node{},
		defs:    map[string][]model.Node{},
		refOf:   map[string]string{},
		symbols: map[string][]model.Node{},
	}
}

// filesOf returns the Go file nodes of the package at importPath.
func (r *resolver) filesOf(ctx context.Context, importPath string) ([]model.Node, error) {
	if fs, ok := r.files[importPath]; ok {
		return fs, nil
	}
	fs, err := store.NodesMatching(ctx, r.gs, store.NodeFilter{
		Kinds: []model.NodeKind{model.KindFile},
		Attrs: map[string]string{model.AttrImportPath: importPath},
	})
	if err != nil {
		return nil, err
	}
	r.files[importPath] = fs
	return fs, nil
}

// symbolsIn returns the symbols defined in the file at uri.
func (r *resolver) symbolsIn(ctx context.Context, uri string) ([]model.Node, error) {
	if ss, ok := r.symbols[uri]; ok {
		return ss, nil
	}
	ss, err := store.NodesMatching(ctx, r.gs, store.NodeFilter{Kinds: []model.NodeKind{model.KindSymbol}, SourceURI: uri})
	if err != nil {
		return nil, err
	}
	r.symbols[uri] = ss
	return ss, nil
}

// fileNode returns the file node for uri, or nil.
func (r *resolver) fileNode(ctx context.Context, uri string) (*model.Node, error) {
	if f, ok := r.fileOf[uri]; ok {
		return f, nil
	}
	fs, err := store.NodesMatching(ctx, r.gs, store.NodeFilter{Kinds: []model.NodeKind{model.KindFile}, SourceURI: uri})
	if err != nil {
		return nil, err
	}
	var f *model.Node
	if len(fs) > 0 {
		f = &fs[0]
	}
	r.fileOf[uri] = f
	return f, nil
}

// definitions returns the package-level functions a pkg:// function reference names, or nil when n is not
// one or names something outside the indexed module.
func (r *resolver) definitions(ctx context.Context, n model.Node) ([]model.Node, error) {
	importPath, ok := model.PkgRefImportPath(n)
	if !ok || n.Label == importPath { // a package node, not a function reference
		return nil, nil
	}
	if ds, ok := r.defs[n.ID]; ok {
		return ds, nil
	}
	name := n.Label[strings.LastIndexByte(n.Label, '.')+1:]
	if model.PkgFuncRefID(importPath, name) != n.ID {
		r.defs[n.ID] = nil
		return nil, nil
	}
	files, err := r.filesOf(ctx, importPath)
	if err != nil {
		return nil, err
	}
	var ds []model.Node
	for _, f := range files {
		ss, err := r.symbolsIn(ctx, f.SourceURI)
		if err != nil {
			return nil, err
		}
		for _, s := range ss {
			if s.Label == name {
				ds = append(ds, s)
			}
		}
	}
	r.defs[n.ID] = ds
	return ds, nil
}

// reference returns the pkg:// reference ID for a package-level Go function definition, or "".
func (r *resolver) reference(ctx context.Context, n model.Node) (string, error) {
	if ref, ok := r.refOf[n.ID]; ok {
		return ref, nil
	}
	ref := ""
	if n.Kind == model.KindSymbol && !model.IsExternal(n) && !strings.Contains(n.Label, ".") &&
		model.AttrString(n.Attrs, "lang_kind") == "func" {
		f, err := r.fileNode(ctx, n.SourceURI)
		if err != nil {
			return "", err
		}
		if f != nil {
			if ip := model.AttrString(f.Attrs, model.AttrImportPath); ip != "" {
				ref = model.PkgFuncRefID(ip, n.Label)
			}
		}
	}
	r.refOf[n.ID] = ref
	return ref, nil
}

// canonical returns the definition a function reference names when there is exactly one, else n itself.
func (r *resolver) canonical(ctx context.Context, n model.Node) (model.Node, error) {
	ds, err := r.definitions(ctx, n)
	if err != nil || len(ds) != 1 {
		return n, err
	}
	return ds[0], nil
}

// Walk directions for get_node_neighborhood.
const (
	dirBoth = "both"
	dirIn   = "in"  // follow edges into a node: its callers, importers, containers
	dirOut  = "out" // follow edges out of a node: what it calls, imports, contains
)

// walk does a breadth-first walk from seed up to depth hops along edges of kinds (all kinds when empty) in
// direction, returning the nodes in visiting order (seed first) and the edges it followed. A function and its
// pkg:// reference count as one node, reported as the definition. excludeTests skips nodes from _test.go files.
func (r *resolver) walk(ctx context.Context, seed model.Node, depth int, kinds []model.EdgeKind, direction string, excludeTests bool) ([]model.Node, []model.Edge, error) {
	seed, err := r.canonical(ctx, seed)
	if err != nil {
		return nil, nil, err
	}
	nodes := []model.Node{seed}
	visited := map[string]bool{seed.ID: true}
	seenEdge := map[string]bool{}
	var edges []model.Edge
	frontier := []model.Node{seed}

	for d := 0; d < depth && len(frontier) > 0; d++ {
		var next []model.Node
		for _, x := range frontier {
			// Walk from both x and its reference, so callers of either count as callers of x.
			members := []string{x.ID}
			if ref, err := r.reference(ctx, x); err != nil {
				return nil, nil, err
			} else if ref != "" {
				members = append(members, ref)
			}
			for _, m := range members {
				sub, err := r.gs.Neighbors(ctx, m, 1, kinds)
				if err != nil {
					return nil, nil, err
				}
				byID := make(map[string]model.Node, len(sub.Nodes))
				for _, n := range sub.Nodes {
					byID[n.ID] = n
				}
				for _, e := range sub.Edges {
					if e.Src != m && e.Dst != m {
						continue
					}
					if (direction == dirIn && e.Dst != m) || (direction == dirOut && e.Src != m) {
						continue
					}
					otherID := e.Src
					if e.Src == m {
						otherID = e.Dst
					}
					other, ok := byID[otherID]
					if !ok {
						continue
					}
					other, err := r.canonical(ctx, other)
					if err != nil {
						return nil, nil, err
					}
					if excludeTests && isGoTestFile(other.SourceURI) {
						continue
					}
					// Report the edge between canonical nodes, so a call through a reference reads as a call to x.
					ce := e
					if ce.Src == m {
						ce.Src, ce.Dst = x.ID, other.ID
					} else {
						ce.Src, ce.Dst = other.ID, x.ID
					}
					key := ce.Src + "\x00" + ce.Dst + "\x00" + string(ce.Kind)
					if ce.Src != ce.Dst && !seenEdge[key] {
						seenEdge[key] = true
						edges = append(edges, ce)
					}
					if !visited[other.ID] {
						visited[other.ID] = true
						nodes = append(nodes, other)
						next = append(next, other)
					}
				}
			}
		}
		frontier = next
	}
	return nodes, edges, nil
}

// lookup resolves a node label for the graph tools. Beyond a plain label it accepts "<dir>:<Name>" (for
// example "internal/analyze:New" or "internal/mcp:Server.Close"), naming a symbol by its package directory.
// It returns the node and how many other symbols share its name, so a caller can be told the lookup was
// ambiguous.
func (r *resolver) lookup(ctx context.Context, label, file string) (node model.Node, others int, found bool, err error) {
	// A label with a colon and no spaces is a "<dir>:<Name>" name; a heading such as "Step 1: install" has spaces.
	if dir, name, ok := strings.Cut(label, ":"); ok && dir != "" && name != "" && !strings.ContainsAny(label, " \t") {
		n, found, err := r.lookupQualified(ctx, dir, name)
		if err != nil || found {
			return n, 0, found, err
		}
	}
	nodes, err := store.FindNodes(ctx, r.gs, label, file)
	if err != nil || len(nodes) == 0 {
		return model.Node{}, 0, false, err
	}
	best, err := r.canonical(ctx, nodes[0])
	if err != nil {
		return model.Node{}, 0, false, err
	}
	for _, n := range nodes[1:] {
		if n.Label == best.Label && n.Kind == model.KindSymbol && !model.IsExternal(n) && n.ID != best.ID {
			others++
		}
	}
	return best, others, true, nil
}

// lookupQualified finds the symbol called name in the Go package whose import path is dir or ends in "/"+dir.
func (r *resolver) lookupQualified(ctx context.Context, dir, name string) (model.Node, bool, error) {
	dir = strings.Trim(strings.ReplaceAll(dir, `\`, "/"), "/")
	all, err := store.NodesMatching(ctx, r.gs, store.NodeFilter{Kinds: []model.NodeKind{model.KindFile}})
	if err != nil {
		return model.Node{}, false, err
	}
	for _, f := range all {
		ip := model.AttrString(f.Attrs, model.AttrImportPath)
		mod := model.AttrString(f.Attrs, model.AttrModule)
		if ip == "" || !(ip == dir || strings.HasSuffix(ip, "/"+dir) || (dir == "." && ip == mod)) {
			continue
		}
		ss, err := r.symbolsIn(ctx, f.SourceURI)
		if err != nil {
			return model.Node{}, false, err
		}
		if i := slices.IndexFunc(ss, func(s model.Node) bool { return s.Label == name }); i >= 0 {
			return ss[i], true, nil
		}
	}
	return model.Node{}, false, nil
}
