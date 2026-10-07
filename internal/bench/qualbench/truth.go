package main

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Facts is the call and import structure of a Go module, computed from its source with go/parser alone so it
// is independent of chiton's extractor. It covers every non-test .go file regardless of build tags or GOOS,
// which is also what a reader grepping the tree sees.
type Facts struct {
	Module string
	// Packages maps a package directory (relative, "/"-separated, "." for the root) to its package name.
	Packages map[string]string
	// Funcs maps a package directory to its package-level functions and the files defining each one.
	Funcs map[string]map[string][]string
	// Calls maps a caller ("dir:Func" or "dir:Type.Method") to the package-level functions it calls directly
	// ("dir:Func"), counting calls inside closures for the enclosing declaration.
	Calls map[string]map[string]bool
	// Imports maps a package directory to the module-internal package directories it imports.
	Imports map[string]map[string]bool
}

// Callers returns the declarations that call fn ("dir:Func") directly, sorted.
func (f *Facts) Callers(fn string) []string {
	var out []string
	for caller, callees := range f.Calls {
		if callees[fn] {
			out = append(out, caller)
		}
	}
	sort.Strings(out)
	return out
}

// Callees returns the package-level module functions fn calls directly, sorted.
func (f *Facts) Callees(fn string) []string { return sortedKeys(f.Calls[fn]) }

// ImportsOf returns the module packages dir imports, sorted.
func (f *Facts) ImportsOf(dir string) []string { return sortedKeys(f.Imports[dir]) }

// Importers returns the module packages that import dir, sorted.
func (f *Facts) Importers(dir string) []string {
	var out []string
	for pkg, imps := range f.Imports {
		if imps[dir] {
			out = append(out, pkg)
		}
	}
	sort.Strings(out)
	return out
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// skipDir reports directories that hold no module source worth asking about.
func skipDir(name string) bool {
	switch name {
	case "testdata", "vendor", "node_modules", "chiton-out", "dist":
		return true
	}
	return strings.HasPrefix(name, ".") && name != "."
}

// Analyze parses every non-test .go file under root, the directory holding go.mod.
func Analyze(root string) (*Facts, error) {
	module, err := modulePath(filepath.Join(root, "go.mod"))
	if err != nil {
		return nil, err
	}
	f := &Facts{
		Module:   module,
		Packages: map[string]string{},
		Funcs:    map[string]map[string][]string{},
		Calls:    map[string]map[string]bool{},
		Imports:  map[string]map[string]bool{},
	}

	type parsed struct {
		dir  string
		file *ast.File
	}
	var files []parsed
	fset := token.NewFileSet()
	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if p != root && skipDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		af, err := parser.ParseFile(fset, p, nil, parser.SkipObjectResolution)
		if err != nil {
			return fmt.Errorf("parse %s: %w", p, err)
		}
		rel, _ := filepath.Rel(root, filepath.Dir(p))
		dir := filepath.ToSlash(rel)
		f.Packages[dir] = af.Name.Name
		files = append(files, parsed{dir, af})
		for _, decl := range af.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || fd.Recv != nil {
				continue
			}
			if f.Funcs[dir] == nil {
				f.Funcs[dir] = map[string][]string{}
			}
			f.Funcs[dir][fd.Name.Name] = append(f.Funcs[dir][fd.Name.Name], filepath.ToSlash(p))
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	for _, pf := range files {
		// alias → module package directory, for this file's module-internal imports
		aliases := map[string]string{}
		for _, imp := range pf.file.Imports {
			ip, _ := strconv.Unquote(imp.Path.Value)
			dir, ok := f.moduleDir(ip)
			if !ok {
				continue
			}
			if f.Imports[pf.dir] == nil {
				f.Imports[pf.dir] = map[string]bool{}
			}
			f.Imports[pf.dir][dir] = true
			name := f.Packages[dir]
			if name == "" {
				name = path.Base(ip)
			}
			if imp.Name != nil {
				name = imp.Name.Name
			}
			if name != "_" && name != "." {
				aliases[name] = dir
			}
		}
		for _, decl := range pf.file.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			caller := pf.dir + ":" + declName(fd)
			locals := localNames(fd)
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				var callee string
				switch fun := call.Fun.(type) {
				case *ast.Ident:
					if !locals[fun.Name] && f.Funcs[pf.dir][fun.Name] != nil {
						callee = pf.dir + ":" + fun.Name
					}
				case *ast.SelectorExpr:
					if x, ok := fun.X.(*ast.Ident); ok && !locals[x.Name] {
						if dir, ok := aliases[x.Name]; ok && f.Funcs[dir][fun.Sel.Name] != nil {
							callee = dir + ":" + fun.Sel.Name
						}
					}
				}
				if callee != "" {
					if f.Calls[caller] == nil {
						f.Calls[caller] = map[string]bool{}
					}
					f.Calls[caller][callee] = true
				}
				return true
			})
		}
	}
	return f, nil
}

// moduleDir maps an import path inside the module to its package directory.
func (f *Facts) moduleDir(importPath string) (string, bool) {
	if importPath == f.Module {
		return ".", true
	}
	rest, ok := strings.CutPrefix(importPath, f.Module+"/")
	return rest, ok
}

// declName is "Func" for a function and "Type.Method" for a method (pointer or value receiver).
func declName(fd *ast.FuncDecl) string {
	if fd.Recv == nil || len(fd.Recv.List) == 0 {
		return fd.Name.Name
	}
	t := fd.Recv.List[0].Type
	if st, ok := t.(*ast.StarExpr); ok {
		t = st.X
	}
	switch tt := t.(type) {
	case *ast.IndexExpr: // generic receiver T[P]
		t = tt.X
	case *ast.IndexListExpr:
		t = tt.X
	}
	if id, ok := t.(*ast.Ident); ok {
		return id.Name + "." + fd.Name.Name
	}
	return fd.Name.Name
}

// localNames collects every name a function declares (receiver, parameters, results, and anything bound in its
// body), so a local that shadows a package-level function or an import alias is not mistaken for a call to it.
// It ignores scoping, so a name declared anywhere in the function shadows everywhere in it; targets whose
// callers trip this are rare, and it errs toward leaving a call out rather than inventing one.
func localNames(fd *ast.FuncDecl) map[string]bool {
	names := map[string]bool{}
	addFields := func(fl *ast.FieldList) {
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
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.AssignStmt:
			if x.Tok == token.DEFINE {
				for _, l := range x.Lhs {
					if id, ok := l.(*ast.Ident); ok {
						names[id.Name] = true
					}
				}
			}
		case *ast.ValueSpec:
			for _, id := range x.Names {
				names[id.Name] = true
			}
		case *ast.RangeStmt:
			if x.Tok == token.DEFINE {
				for _, e := range []ast.Expr{x.Key, x.Value} {
					if id, ok := e.(*ast.Ident); ok {
						names[id.Name] = true
					}
				}
			}
		case *ast.FuncLit:
			addFields(x.Type.Params)
			addFields(x.Type.Results)
		}
		return true
	})
	return names
}

// modulePath reads the module line of a go.mod file.
func modulePath(gomod string) (string, error) {
	fh, err := os.Open(gomod)
	if err != nil {
		return "", err
	}
	defer fh.Close()
	sc := bufio.NewScanner(fh)
	for sc.Scan() {
		if rest, ok := strings.CutPrefix(strings.TrimSpace(sc.Text()), "module "); ok {
			return strings.Trim(strings.TrimSpace(rest), `"`), nil
		}
	}
	if err := sc.Err(); err != nil {
		return "", err
	}
	return "", fmt.Errorf("%s has no module line", gomod)
}
