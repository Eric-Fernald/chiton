package ast

import (
	"context"
	"fmt"
	goast "go/ast"
	"go/parser"
	"go/token"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// GoExtractor parses Go source files and emits typed graph nodes and edges.
type GoExtractor struct{}

func (g *GoExtractor) ExtractFile(ctx context.Context, path string) ([]model.Node, []model.Edge, error) {
	return extractFile(ctx, g, path)
}

// ExtractSource parses a single .go file into file, top-level func and type, and imported-package nodes.
// CALLS covers direct intra-file calls and pkg.Func selector calls. Receiver method calls need go/types.
func (g *GoExtractor) ExtractSource(_ context.Context, path string, src []byte) ([]model.Node, []model.Edge, error) {
	fset := token.NewFileSet()
	// Skip object resolution, since the extractor reads only identifier names, never ast.Object links.
	f, err := parser.ParseFile(fset, path, src, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		return nil, nil, fmt.Errorf("parse %s: %w", path, err)
	}

	fileNode := newFileNode(path, src, map[string]any{"language": "go", "package": f.Name.Name})
	uri := fileNode.SourceURI

	nodes := []model.Node{fileNode}
	var edges []model.Edge

	// symbols maps plain func and type names to node IDs for intra-file CALLS resolution. It omits methods
	// because a bare call can't name one and same-named methods on different types would overwrite each other.
	symbols := map[string]string{}
	type funcBody struct {
		id   string
		body *goast.BlockStmt
	}
	var bodies []funcBody

	// Imports become package nodes and IMPORTS edges. Aliases map to package paths for selector calls.
	importAliases := make(map[string]string, len(f.Imports))
	for _, imp := range f.Imports {
		pkgPath := strings.Trim(imp.Path.Value, `"`)
		pkgNode := importNode(pkgPath)
		nodes = append(nodes, pkgNode)
		edges = append(edges, importEdge(fileNode.ID, pkgNode.ID))

		alias := pkgPath[strings.LastIndexByte(pkgPath, '/')+1:]
		if imp.Name != nil && imp.Name.Name != "_" && imp.Name.Name != "." {
			alias = imp.Name.Name
		}
		importAliases[alias] = pkgPath
	}

	// Top-level declarations → symbol nodes + CONTAINS edges.
	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *goast.FuncDecl:
			label := funcLabel(d)
			pos := fset.Position(d.Pos())
			end := fset.Position(d.End())

			attrs := map[string]any{"lang_kind": "func"}
			if d.Doc != nil {
				if text := strings.TrimSpace(d.Doc.Text()); text != "" {
					attrs["docstring"] = text
				}
			}

			sym := model.Node{
				ID:         model.NewNodeID(uri, label),
				Kind:       model.KindSymbol,
				Label:      label,
				SourceURI:  uri,
				Span:       &model.SourceSpan{StartLine: pos.Line, EndLine: end.Line},
				Provenance: model.ProvExtracted,
				Confidence: 1.0,
				Attrs:      attrs,
			}
			nodes = append(nodes, sym)
			if d.Recv == nil {
				symbols[d.Name.Name] = sym.ID
			}
			if d.Body != nil {
				bodies = append(bodies, funcBody{id: sym.ID, body: d.Body})
			}
			edges = append(edges, containsEdge(fileNode.ID, sym.ID))

		case *goast.GenDecl:
			for _, spec := range d.Specs {
				ts, ok := spec.(*goast.TypeSpec)
				if !ok {
					continue
				}
				pos := fset.Position(ts.Pos())
				attrs := map[string]any{"lang_kind": "type"}
				if d.Doc != nil {
					if text := strings.TrimSpace(d.Doc.Text()); text != "" {
						attrs["docstring"] = text
					}
				}
				sym := model.Node{
					ID:         model.NewNodeID(uri, ts.Name.Name),
					Kind:       model.KindSymbol,
					Label:      ts.Name.Name,
					SourceURI:  uri,
					Span:       &model.SourceSpan{StartLine: pos.Line},
					Provenance: model.ProvExtracted,
					Confidence: 1.0,
					Attrs:      attrs,
				}
				nodes = append(nodes, sym)
				symbols[ts.Name.Name] = sym.ID
				edges = append(edges, containsEdge(fileNode.ID, sym.ID))
			}
		}
	}

	// Add CALLS edges for direct calls to this file's symbols and pkg.Func calls where pkg is an import alias.
	for _, fb := range bodies {
		callerID := fb.id
		goast.Inspect(fb.body, func(n goast.Node) bool {
			call, ok := n.(*goast.CallExpr)
			if !ok {
				return true
			}
			switch fun := call.Fun.(type) {
			case *goast.Ident:
				calleeID, ok := symbols[fun.Name]
				if !ok || calleeID == callerID {
					return true
				}
				edges = append(edges, callsEdge(callerID, calleeID))
			case *goast.SelectorExpr:
				xIdent, ok := fun.X.(*goast.Ident)
				if !ok {
					return true
				}
				pkgPath, ok := importAliases[xIdent.Name]
				if !ok {
					return true
				}
				pkgURI := "pkg://" + pkgPath
				calleeNode := model.Node{
					ID:         model.NewNodeID(pkgURI, fun.Sel.Name),
					Kind:       model.KindSymbol,
					Label:      xIdent.Name + "." + fun.Sel.Name,
					SourceURI:  pkgURI,
					Provenance: model.ProvExtracted,
					Confidence: 1.0,
					Attrs:      map[string]any{"external": true},
				}
				nodes = append(nodes, calleeNode)
				edges = append(edges, callsEdge(callerID, calleeNode.ID))
			}
			return true
		})
	}

	return nodes, edges, nil
}

// --- helpers ---

// funcLabel returns "TypeName.MethodName" for methods, "FuncName" for functions.
func funcLabel(d *goast.FuncDecl) string {
	if d.Recv != nil && len(d.Recv.List) > 0 {
		if t, ok := receiverType(d.Recv.List[0].Type); ok {
			return t + "." + d.Name.Name
		}
	}
	return d.Name.Name
}

func receiverType(expr goast.Expr) (string, bool) {
	switch t := expr.(type) {
	case *goast.Ident:
		return t.Name, true
	case *goast.StarExpr:
		if id, ok := t.X.(*goast.Ident); ok {
			return id.Name, true
		}
	}
	return "", false
}
