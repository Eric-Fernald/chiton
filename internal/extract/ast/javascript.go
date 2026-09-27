package ast

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

var (
	reJSImportFrom = regexp.MustCompile(`\bfrom\s+['"]([^'"]+)['"]`)
	reJSRequire    = regexp.MustCompile(`\brequire\s*\(\s*['"]([^'"]+)['"]\s*\)`)

	reJSFuncDecl  = regexp.MustCompile(`^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s+(\w+)\s*\(`)
	reJSArrowFunc = regexp.MustCompile(`^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>`)
	reJSFuncExpr  = regexp.MustCompile(`^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function`)
	reJSClassDecl = regexp.MustCompile(`^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)`)
	reJSInterface = regexp.MustCompile(`^(?:export\s+)?interface\s+(\w+)`)
	reJSTypeAlias = regexp.MustCompile(`^(?:export\s+)?type\s+(\w+)\s*=`)
	reJSEnumDecl  = regexp.MustCompile(`^(?:export\s+)?(?:const\s+)?enum\s+(\w+)`)

	reJSExtends    = regexp.MustCompile(`\bextends\s+([\w.]+)`)
	reJSImplements = regexp.MustCompile(`\bimplements\s+([\w,\s]+?)(?:\{|$)`)

	// Method inside a class: line starts with whitespace, optional modifiers, then name + (
	reJSMethod = regexp.MustCompile(`^\s+(?:(?:static|async|override|get|set|public|private|protected|abstract|readonly)\s+)*([a-zA-Z_$][\w$]*)\s*[(<]`)

	reJSCall = regexp.MustCompile(`\b([a-zA-Z_$][\w$]*)\s*\(`)
)

// JavaScriptExtractor parses JavaScript and TypeScript source files.
type JavaScriptExtractor struct{}

func (j *JavaScriptExtractor) ExtractFile(ctx context.Context, path string) ([]model.Node, []model.Edge, error) {
	return extractFile(ctx, j, path)
}

func (j *JavaScriptExtractor) ExtractSource(_ context.Context, path string, src []byte) ([]model.Node, []model.Edge, error) {
	lang := "javascript"
	lower := strings.ToLower(path)
	if strings.HasSuffix(lower, ".ts") || strings.HasSuffix(lower, ".tsx") {
		lang = "typescript"
	}

	g := newFileGraph(newFileNode(path, src, map[string]any{"language": lang}))
	lines := splitLines(src)
	var scopes braceScopes // kinds: "class" (named for the class) or "func"
	var funcs []funcRecord

	// Pass 1: collect symbols. A line wholly inside a /* */ block contributes none.
	scopes.scan(lines, nil, func(i int, line string, pre, _ int) {
		if t := scopes.top(); t != nil && t.kind == "class" && pre > t.pushDepth {
			if m := reJSMethod.FindStringSubmatch(line); m != nil && !isJSKeyword(m[1]) {
				label := scopes.enclosing("class") + "." + m[1]
				funcs = append(funcs, funcRecord{lineStart: i, id: g.declare(label, "method", i+1, m[1])})
				scopes.push("func", label, pre)
				return
			}
		}

		if name, kind := jsTopLevelDecl(line); name != "" && !isJSKeyword(name) {
			id := g.declare(name, kind, i+1)
			switch kind {
			case "class":
				g.addHeritage(id, line, reJSExtends, reJSImplements)
				scopes.push("class", name, pre)
			case "func", "arrow":
				funcs = append(funcs, funcRecord{lineStart: i, id: id})
				scopes.push("func", name, pre)
			}
		}

		if m := findWith(reJSImportFrom, line, "from"); m != nil {
			g.addImport(m[1])
		}
		if m := findWith(reJSRequire, line, "require"); m != nil {
			g.addImport(m[1])
		}
	})

	g.addBraceCalls(lines, funcs, true, reJSCall, isJSKeyword)
	return g.nodes, g.edges, nil
}

// jsTopLevelDecl returns (name, kind) for a top-level declaration on this line.
func jsTopLevelDecl(line string) (string, string) {
	if line == "" || line[0] < 'a' || line[0] > 'z' {
		return "", "" // every pattern opens with a lowercase keyword at column 0
	}
	if m := reJSClassDecl.FindStringSubmatch(line); m != nil {
		return m[1], "class"
	}
	if m := reJSFuncDecl.FindStringSubmatch(line); m != nil {
		return m[1], "func"
	}
	if m := reJSFuncExpr.FindStringSubmatch(line); m != nil {
		return m[1], "func"
	}
	if m := reJSArrowFunc.FindStringSubmatch(line); m != nil {
		return m[1], "arrow"
	}
	if m := reJSInterface.FindStringSubmatch(line); m != nil {
		return m[1], "interface"
	}
	if m := reJSEnumDecl.FindStringSubmatch(line); m != nil {
		return m[1], "enum"
	}
	if m := reJSTypeAlias.FindStringSubmatch(line); m != nil {
		return m[1], "type"
	}
	return "", ""
}

var jsKeywords = map[string]bool{
	"if": true, "else": true, "for": true, "while": true, "do": true,
	"switch": true, "case": true, "break": true, "continue": true, "return": true,
	"new": true, "delete": true, "typeof": true, "instanceof": true, "void": true,
	"throw": true, "catch": true, "finally": true, "try": true, "class": true,
	"function": true, "import": true, "export": true, "from": true, "const": true,
	"let": true, "var": true, "async": true, "await": true, "yield": true,
	"super": true, "this": true, "true": true, "false": true, "null": true,
	"undefined": true, "constructor": true, "of": true, "in": true,
}

func isJSKeyword(s string) bool { return jsKeywords[s] }
