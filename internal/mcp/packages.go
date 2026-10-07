package mcpsrv

import (
	"context"
	"fmt"
	"sort"
	"strings"

	mcplib "github.com/mark3labs/mcp-go/mcp"
	mcpsdk "github.com/mark3labs/mcp-go/server"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

func packageDepsTool() mcplib.Tool {
	return mcplib.NewTool("package_deps",
		mcplib.WithDescription("Package-level dependencies of a Go package, in one call: the packages it imports, or the packages that import it, "+
			"optionally followed transitively. It joins the imports of every file in the package, which per-file IMPORTS edges and grep "+
			"leave you to piece together. Use it before grep for \"what does package X depend on\" or \"what depends on package X\"."),
		mcplib.WithString("package",
			mcplib.Required(),
			mcplib.Description("The package: its directory relative to the module root (e.g. \"internal/graph/store\", \".\" for the root), its import path, or a unique trailing part of either"),
		),
		mcplib.WithString("direction",
			mcplib.DefaultString("imports"),
			mcplib.Description("\"imports\" (default): packages this one imports. \"importers\": packages that import this one."),
			mcplib.Enum("imports", "importers"),
		),
		mcplib.WithBoolean("transitive",
			mcplib.DefaultBool(false),
			mcplib.Description("Follow the dependency chain to the end instead of stopping at direct dependencies. Each result carries its depth."),
		),
		mcplib.WithBoolean("include_external",
			mcplib.DefaultBool(false),
			mcplib.Description("For \"imports\", also list the standard library and third-party packages; by default only packages in the indexed module are returned"),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

// goPackage is one Go package in the graph: its import path, module, and file node IDs.
type goPackage struct {
	importPath, module string
	files              []string
}

func (p goPackage) dir() string {
	if p.importPath == p.module {
		return "."
	}
	return strings.TrimPrefix(p.importPath, p.module+"/")
}

// goPackages groups the graph's Go file nodes into packages by their import_path attribute.
func goPackages(ctx context.Context, gs store.GraphStore) (map[string]*goPackage, error) {
	files, err := store.NodesMatching(ctx, gs, store.NodeFilter{Kinds: []model.NodeKind{model.KindFile}})
	if err != nil {
		return nil, err
	}
	pkgs := map[string]*goPackage{}
	for _, f := range files {
		ip := model.AttrString(f.Attrs, model.AttrImportPath)
		if ip == "" {
			continue
		}
		p := pkgs[ip]
		if p == nil {
			p = &goPackage{importPath: ip, module: model.AttrString(f.Attrs, model.AttrModule)}
			pkgs[ip] = p
		}
		p.files = append(p.files, f.ID)
	}
	return pkgs, nil
}

// matchPackage resolves the package argument against the indexed packages.
func matchPackage(pkgs map[string]*goPackage, arg string) ([]*goPackage, error) {
	arg = strings.Trim(strings.ReplaceAll(strings.TrimSpace(arg), `\`, "/"), "/")
	arg = strings.TrimPrefix(arg, "./")
	var exact, suffix []*goPackage
	for _, p := range pkgs {
		switch {
		case p.importPath == arg || p.dir() == arg || (arg == "" && p.dir() == "."):
			exact = append(exact, p)
		case strings.HasSuffix(p.importPath, "/"+arg):
			suffix = append(suffix, p)
		}
	}
	if len(exact) > 0 {
		return exact, nil
	}
	return suffix, nil
}

// directImports returns the import paths the package's files import.
func directImports(ctx context.Context, gs store.GraphStore, p *goPackage) ([]string, error) {
	seen := map[string]bool{}
	for _, fileID := range p.files {
		sub, err := gs.Neighbors(ctx, fileID, 1, []model.EdgeKind{model.EdgeImports})
		if err != nil {
			return nil, err
		}
		byID := map[string]model.Node{}
		for _, n := range sub.Nodes {
			byID[n.ID] = n
		}
		for _, e := range sub.Edges {
			if e.Src != fileID {
				continue
			}
			if ip, ok := model.PkgRefImportPath(byID[e.Dst]); ok {
				seen[ip] = true
			}
		}
	}
	return sortedKeys(seen), nil
}

// directImporters returns the import paths of the indexed packages with a file that imports p.
func directImporters(ctx context.Context, gs store.GraphStore, p *goPackage) ([]string, error) {
	pkgNode := model.NewNodeID("pkg://"+p.importPath, p.importPath)
	sub, err := gs.Neighbors(ctx, pkgNode, 1, []model.EdgeKind{model.EdgeImports})
	if err != nil {
		return nil, err
	}
	byID := map[string]model.Node{}
	for _, n := range sub.Nodes {
		byID[n.ID] = n
	}
	seen := map[string]bool{}
	for _, e := range sub.Edges {
		if e.Dst != pkgNode {
			continue
		}
		if ip := model.AttrString(byID[e.Src].Attrs, model.AttrImportPath); ip != "" && ip != p.importPath {
			seen[ip] = true
		}
	}
	return sortedKeys(seen), nil
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func packageDepsHandler(gs store.GraphStore) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		arg, err := req.RequireString("package")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		direction := req.GetString("direction", "imports")
		if direction != "importers" {
			direction = "imports"
		}
		transitive := req.GetBool("transitive", false)
		external := req.GetBool("include_external", false) && direction == "imports"

		pkgs, err := goPackages(ctx, gs)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("listing packages failed", err), nil
		}
		matches, err := matchPackage(pkgs, arg)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("matching package failed", err), nil
		}
		switch len(matches) {
		case 0:
			return mcplib.NewToolResultText(fmt.Sprintf("no indexed Go package matches %q; package_deps needs a graph built by this chiton version (rebuild with chiton build)", arg)), nil
		case 1:
		default:
			var names []string
			for _, m := range matches {
				names = append(names, m.dir())
			}
			sort.Strings(names)
			return mcplib.NewToolResultText(fmt.Sprintf("%q matches %d packages; name one: %s", arg, len(matches), strings.Join(names, ", "))), nil
		}
		root := matches[0]

		type dep struct {
			Dir        string `json:"dir,omitempty"`
			ImportPath string `json:"import_path"`
			Depth      int    `json:"depth"`
			External   bool   `json:"external,omitempty"`
		}
		var deps []dep
		visited := map[string]bool{root.importPath: true}
		frontier := []*goPackage{root}
		for depth := 1; len(frontier) > 0; depth++ {
			var next []*goPackage
			for _, p := range frontier {
				var ips []string
				if direction == "imports" {
					ips, err = directImports(ctx, gs, p)
				} else {
					ips, err = directImporters(ctx, gs, p)
				}
				if err != nil {
					return mcplib.NewToolResultErrorFromErr("reading imports failed", err), nil
				}
				for _, ip := range ips {
					if visited[ip] {
						continue
					}
					q, inModule := pkgs[ip]
					if !inModule && !external {
						continue
					}
					visited[ip] = true
					d := dep{ImportPath: ip, Depth: depth, External: !inModule}
					if inModule {
						d.Dir = q.dir()
						next = append(next, q)
					}
					deps = append(deps, d)
				}
			}
			if !transitive {
				break
			}
			frontier = next
		}
		sort.SliceStable(deps, func(i, j int) bool {
			if deps[i].Depth != deps[j].Depth {
				return deps[i].Depth < deps[j].Depth
			}
			return deps[i].ImportPath < deps[j].ImportPath
		})
		if deps == nil {
			deps = []dep{}
		}
		scope := "direct"
		if transitive {
			scope = "transitive"
		}
		return jsonResult(struct {
			Package    string `json:"package"`
			Direction  string `json:"direction"`
			Transitive bool   `json:"transitive"`
			Packages   []dep  `json:"packages"`
			Summary    string `json:"summary"`
		}{
			Package:    root.dir(),
			Direction:  direction,
			Transitive: transitive,
			Packages:   deps,
			Summary:    fmt.Sprintf("%d %s %s of %s", len(deps), scope, direction, root.dir()),
		})
	}
}
