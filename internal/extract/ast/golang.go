package ast

import (
	"context"
	"fmt"
	goast "go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// GoExtractor parses Go source files and emits typed graph nodes and edges.
type GoExtractor struct{}

func (g *GoExtractor) ExtractFile(ctx context.Context, path string) ([]model.Node, []model.Edge, error) {
	return extractFile(ctx, g, path)
}

// ExtractSource parses a single .go file into file, top-level func and type, and imported-package nodes.
// CALLS covers direct calls to this file's functions, and calls to package-level functions elsewhere (pkg.Func
// selector calls, and bare calls into sibling files of the package) as pkg:// references. Receiver method calls
// need go/types.
func (g *GoExtractor) ExtractSource(_ context.Context, path string, src []byte) ([]model.Node, []model.Edge, error) {
	fset := token.NewFileSet()
	// Skip object resolution, since the extractor reads only identifier names, never ast.Object links.
	f, err := parser.ParseFile(fset, path, src, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		return nil, nil, fmt.Errorf("parse %s: %w", path, err)
	}

	// The file's import path lets a call into a sibling file of the same package, which this file can't
	// resolve, still name its target: the pkg:// reference other packages use for the same function.
	fileAttrs := map[string]any{"language": "go", "package": f.Name.Name}
	importPath, module := goImportPath(path)
	if importPath != "" {
		fileAttrs[model.AttrImportPath] = importPath
		fileAttrs[model.AttrModule] = module
	}

	// fg drops repeat nodes and edges (a call site per call, a package imported twice, repeated init funcs).
	fg := newFileGraph(newFileNode(path, src, fileAttrs))

	// symbols maps plain func and type names to node IDs for intra-file CALLS resolution. It omits methods
	// because a bare call can't name one and same-named methods on different types would overwrite each other.
	symbols := map[string]string{}
	type funcBody struct {
		id     string
		body   *goast.BlockStmt
		locals map[string]bool
	}
	var bodies []funcBody

	// Imports become package nodes and IMPORTS edges. Aliases map to package paths for selector calls.
	importAliases := make(map[string]string, len(f.Imports))
	for _, imp := range f.Imports {
		pkgPath := strings.Trim(imp.Path.Value, `"`)
		fg.addExternal(fg.fileID, pkgPath, importEdge)

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
				ID:         model.NewNodeID(fg.uri, label),
				Kind:       model.KindSymbol,
				Label:      label,
				SourceURI:  fg.uri,
				Span:       &model.SourceSpan{StartLine: pos.Line, EndLine: end.Line},
				Provenance: model.ProvExtracted,
				Confidence: 1.0,
				Attrs:      attrs,
			}
			fg.addNode(sym)
			if d.Recv == nil {
				symbols[d.Name.Name] = sym.ID
			}
			if d.Body != nil {
				bodies = append(bodies, funcBody{id: sym.ID, body: d.Body, locals: goLocalNames(d)})
			}
			fg.addEdge(containsEdge(fg.fileID, sym.ID))

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
					ID:         model.NewNodeID(fg.uri, ts.Name.Name),
					Kind:       model.KindSymbol,
					Label:      ts.Name.Name,
					SourceURI:  fg.uri,
					Span:       &model.SourceSpan{StartLine: pos.Line},
					Provenance: model.ProvExtracted,
					Confidence: 1.0,
					Attrs:      attrs,
				}
				fg.addNode(sym)
				symbols[ts.Name.Name] = sym.ID
				fg.addEdge(containsEdge(fg.fileID, sym.ID))
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
				if ok {
					if calleeID != callerID {
						fg.addEdge(callsEdge(callerID, calleeID))
					}
					return true
				}
				// Not declared in this file: a function in a sibling file of the package, unless it is a
				// builtin or a local. It gets the same pkg:// reference a call from another package would.
				if importPath == "" || goBuiltins[fun.Name] || fb.locals[fun.Name] {
					return true
				}
				ref := pkgFuncRef(importPath, f.Name.Name, fun.Name)
				fg.addNode(ref)
				fg.addEdge(callsEdge(callerID, ref.ID))
			case *goast.SelectorExpr:
				xIdent, ok := fun.X.(*goast.Ident)
				if !ok {
					return true
				}
				pkgPath, ok := importAliases[xIdent.Name]
				if !ok || fb.locals[xIdent.Name] {
					return true
				}
				ref := pkgFuncRef(pkgPath, xIdent.Name, fun.Sel.Name)
				fg.addNode(ref)
				fg.addEdge(callsEdge(callerID, ref.ID))
			}
			return true
		})
	}

	return fg.nodes, fg.edges, nil
}

// pkgFuncRef is the node a call to function name in the package at pkgPath points to when the definition
// isn't in the calling file. Every caller, in any package, gets the same ID, and model.PkgFuncRefID gives the
// definition's way back to it, so the MCP server can join callers to the function they call.
func pkgFuncRef(pkgPath, pkgName, name string) model.Node {
	pkgURI := "pkg://" + pkgPath
	return model.Node{
		ID:         model.PkgFuncRefID(pkgPath, name),
		Kind:       model.KindSymbol,
		Label:      pkgName + "." + name,
		SourceURI:  pkgURI,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		Attrs:      map[string]any{"external": true},
	}
}

// goBuiltins are the predeclared functions and conversion types a bare call can name without declaring them.
var goBuiltins = map[string]bool{
	"append": true, "cap": true, "clear": true, "close": true, "complex": true, "copy": true, "delete": true,
	"imag": true, "len": true, "make": true, "max": true, "min": true, "new": true, "panic": true, "print": true,
	"println": true, "real": true, "recover": true,
	"any": true, "bool": true, "byte": true, "comparable": true, "complex64": true, "complex128": true,
	"error": true, "float32": true, "float64": true, "int": true, "int8": true, "int16": true, "int32": true,
	"int64": true, "rune": true, "string": true, "uint": true, "uint8": true, "uint16": true, "uint32": true,
	"uint64": true, "uintptr": true,
}

// goLocalNames collects every name fd declares (receiver, parameters, results, and anything bound in its body),
// so a call through a local func value, or a local that shadows an import, isn't taken for a package call. It
// ignores scoping: a name declared anywhere in the function counts everywhere in it, which can drop a real call
// but never invents one.
func goLocalNames(fd *goast.FuncDecl) map[string]bool {
	names := map[string]bool{}
	addFields := func(fl *goast.FieldList) {
		if fl == nil {
			return
		}
		for _, field := range fl.List {
			for _, n := range field.Names {
				names[n.Name] = true
			}
		}
	}
	addFields(fd.Recv)
	addFields(fd.Type.Params)
	addFields(fd.Type.Results)
	if fd.Body == nil {
		return names
	}
	goast.Inspect(fd.Body, func(n goast.Node) bool {
		switch x := n.(type) {
		case *goast.AssignStmt:
			if x.Tok == token.DEFINE {
				for _, l := range x.Lhs {
					if id, ok := l.(*goast.Ident); ok {
						names[id.Name] = true
					}
				}
			}
		case *goast.ValueSpec:
			for _, id := range x.Names {
				names[id.Name] = true
			}
		case *goast.RangeStmt:
			if x.Tok == token.DEFINE {
				for _, e := range []goast.Expr{x.Key, x.Value} {
					if id, ok := e.(*goast.Ident); ok {
						names[id.Name] = true
					}
				}
			}
		case *goast.FuncLit:
			addFields(x.Type.Params)
			addFields(x.Type.Results)
		}
		return true
	})
	return names
}

// goModCache maps a directory to the module that contains it, so each go.mod is read once per process.
var goModCache sync.Map // abs dir → goMod

type goMod struct{ root, path string }

// goImportPath returns the import path of the package in file's directory and its module path, from the
// nearest go.mod above it. Both are "" outside a module.
func goImportPath(file string) (importPath, module string) {
	dir, err := filepath.Abs(filepath.Dir(file))
	if err != nil {
		return "", ""
	}
	m := findGoMod(dir)
	if m.path == "" {
		return "", ""
	}
	rel, err := filepath.Rel(m.root, dir)
	if err != nil {
		return "", ""
	}
	if rel == "." {
		return m.path, m.path
	}
	return m.path + "/" + filepath.ToSlash(rel), m.path
}

func findGoMod(dir string) goMod {
	if v, ok := goModCache.Load(dir); ok {
		return v.(goMod)
	}
	var m goMod
	if path := readModulePath(filepath.Join(dir, "go.mod")); path != "" {
		m = goMod{root: dir, path: path}
	} else if parent := filepath.Dir(dir); parent != dir {
		m = findGoMod(parent)
	}
	goModCache.Store(dir, m)
	return m
}

// readModulePath returns the module path a go.mod declares, or "" when there is no readable go.mod.
func readModulePath(gomod string) string {
	b, err := os.ReadFile(gomod)
	if err != nil {
		return ""
	}
	for line := range strings.Lines(string(b)) {
		if rest, ok := strings.CutPrefix(strings.TrimSpace(line), "module "); ok {
			return strings.Trim(strings.TrimSpace(rest), `"`)
		}
	}
	return ""
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
