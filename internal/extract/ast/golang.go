// Package ast extracts graph nodes and edges from source files using
// the Go standard library parser. No CGo required.
package ast

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io/fs"
	goast "go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// GoExtractor parses Go source files and emits typed graph nodes and edges.
type GoExtractor struct{}

// ExtractFile parses a single .go file and returns its nodes and edges.
//
// Emits:
//   - 1 file node per source file
//   - 1 symbol node per top-level func/type declaration
//   - CONTAINS edges from file → each symbol
//   - IMPORTS edges from file → each imported package (package node is external)
//   - CALLS edges: direct intra-file calls and package-level selector calls
//     (pkg.Method where pkg is an import alias). Cross-file direct calls and
//     receiver method calls require go/types; skipped.
func (g *GoExtractor) ExtractFile(_ context.Context, path string) ([]model.Node, []model.Edge, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("read %s: %w", path, err)
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, src, parser.ParseComments)
	if err != nil {
		return nil, nil, fmt.Errorf("parse %s: %w", path, err)
	}

	absPath, _ := filepath.Abs(path)
	uri := "file://" + filepath.ToSlash(absPath)
	sum := sha256.Sum256(src)

	fileNode := model.Node{
		ID:            model.NewNodeID(uri, uri),
		Kind:          model.KindFile,
		Label:         filepath.Base(path),
		SourceURI:     uri,
		Provenance:    model.ProvExtracted,
		Confidence:    1.0,
		ContentSHA256: fmt.Sprintf("%x", sum[:]),
		Attrs:         map[string]any{"language": "go", "package": f.Name.Name},
	}

	nodes := []model.Node{fileNode}
	var edges []model.Edge

	// symbols maps unqualified func name → node ID for intra-file CALLS resolution.
	symbols := map[string]string{}

	// Imports → package nodes + IMPORTS edges.
	for _, imp := range f.Imports {
		pkgPath := strings.Trim(imp.Path.Value, `"`)
		pkgURI := "pkg://" + pkgPath
		pkgNode := model.Node{
			ID:         model.NewNodeID(pkgURI, pkgPath),
			Kind:       model.KindSymbol,
			Label:      pkgPath,
			SourceURI:  pkgURI,
			Provenance: model.ProvExtracted,
			Confidence: 1.0,
			Attrs:      map[string]any{"external": true},
		}
		nodes = append(nodes, pkgNode)
		edges = append(edges, model.Edge{
			ID:         model.NewEdgeID(fileNode.ID, pkgNode.ID, model.EdgeImports),
			Src:        fileNode.ID,
			Dst:        pkgNode.ID,
			Kind:       model.EdgeImports,
			Provenance: model.ProvExtracted,
			Confidence: 1.0,
		})
	}

	// Build alias → full package path map for resolving selector call expressions.
	importAliases := map[string]string{}
	for _, imp := range f.Imports {
		pkgPath := strings.Trim(imp.Path.Value, `"`)
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
			symbols[d.Name.Name] = sym.ID
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

	// CALLS edges: walk each func body for two cases:
	//   1. Direct calls to symbols defined in this file (e.g. fetch(url)).
	//   2. Selector calls where X is an import alias (e.g. http.Get(url)).
	//      Receiver method calls and chained calls require go/types; skipped.
	for _, decl := range f.Decls {
		fd, ok := decl.(*goast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}
		callerID, ok := symbols[fd.Name.Name]
		if !ok {
			continue
		}
		goast.Inspect(fd.Body, func(n goast.Node) bool {
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
				edges = append(edges, model.Edge{
					ID:         model.NewEdgeID(callerID, calleeID, model.EdgeCalls),
					Src:        callerID,
					Dst:        calleeID,
					Kind:       model.EdgeCalls,
					Provenance: model.ProvExtracted,
					Confidence: 1.0,
				})
			case *goast.SelectorExpr:
				xIdent, ok := fun.X.(*goast.Ident)
				if !ok {
					return true
				}
				pkgPath, ok := importAliases[xIdent.Name]
				if !ok {
					return true
				}
				calleeNode := model.Node{
					ID:         model.NewNodeID("pkg://"+pkgPath, fun.Sel.Name),
					Kind:       model.KindSymbol,
					Label:      xIdent.Name + "." + fun.Sel.Name,
					SourceURI:  "pkg://" + pkgPath,
					Provenance: model.ProvExtracted,
					Confidence: 1.0,
					Attrs:      map[string]any{"external": true},
				}
				nodes = append(nodes, calleeNode)
				edges = append(edges, model.Edge{
					ID:         model.NewEdgeID(callerID, calleeNode.ID, model.EdgeCalls),
					Src:        callerID,
					Dst:        calleeNode.ID,
					Kind:       model.EdgeCalls,
					Provenance: model.ProvExtracted,
					Confidence: 1.0,
				})
			}
			return true
		})
	}

	return nodes, edges, nil
}

// ExtractDir walks dir recursively, extracts all .go files, and returns
// the combined nodes and edges. Files that fail to parse are skipped with
// a warning printed to stderr (parse errors in one file should not abort the build).
func (g *GoExtractor) ExtractDir(ctx context.Context, dir string) ([]model.Node, []model.Edge, error) {
	var allNodes []model.Node
	var allEdges []model.Edge

	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}
		nodes, edges, err := g.ExtractFile(ctx, path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "chiton: skip %s: %v\n", path, err)
			return nil
		}
		allNodes = append(allNodes, nodes...)
		allEdges = append(allEdges, edges...)
		return nil
	})
	return allNodes, allEdges, err
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

func containsEdge(fileID, symID string) model.Edge {
	return model.Edge{
		ID:         model.NewEdgeID(fileID, symID, model.EdgeContains),
		Src:        fileID,
		Dst:        symID,
		Kind:       model.EdgeContains,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
	}
}
