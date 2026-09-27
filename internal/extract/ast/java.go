package ast

import (
	"context"
	"regexp"
	"slices"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

var (
	reJavaImport     = regexp.MustCompile(`^\s*import\s+(?:static\s+)?([^;]+);`)
	reJavaClass      = regexp.MustCompile(`(?:^|\s)(?:public|protected|private|abstract|final|static|\s)*class\s+(\w+)`)
	reJavaInterface  = regexp.MustCompile(`(?:^|\s)(?:public|protected|private|abstract|\s)*interface\s+(\w+)`)
	reJavaEnum       = regexp.MustCompile(`(?:^|\s)(?:public|protected|private|\s)*enum\s+(\w+)`)
	reJavaExtends    = regexp.MustCompile(`\bextends\s+([\w.]+)`)
	reJavaImplements = regexp.MustCompile(`\bimplements\s+([\w.,\s]+?)(?:\{|$)`)

	// Method: modifiers* ReturnType methodName ( — at indented level
	reJavaMethod = regexp.MustCompile(`^\s+(?:(?:public|protected|private|static|final|abstract|synchronized|native|default|override)\s+)*(?:<[^>]*>\s+)?(?:\w[\w.<>\[\]]*\s+)+(\w+)\s*\(`)

	reJavaCall = regexp.MustCompile(`\b([a-zA-Z_]\w*)\s*\(`)
)

// JavaExtractor parses Java source files using regex-based line scanning.
type JavaExtractor struct{}

func (j *JavaExtractor) ExtractFile(ctx context.Context, path string) ([]model.Node, []model.Edge, error) {
	return extractFile(ctx, j, path)
}

// javaTypeKinds are the scope kinds whose body can declare methods.
var javaTypeKinds = []string{"class", "interface", "enum"}

func (j *JavaExtractor) ExtractSource(_ context.Context, path string, src []byte) ([]model.Node, []model.Edge, error) {
	g := newFileGraph(newFileNode(path, src, map[string]any{"language": "java"}))
	lines := splitLines(src)
	var scopes braceScopes
	var funcs []funcRecord

	// Pass 1: collect symbols, skipping comment lines (scan still tracks their braces and /* */ state).
	isComment := func(s string) bool { return strings.HasPrefix(s, "//") || strings.HasPrefix(s, "*") }
	scopes.scan(lines, isComment, func(i int, line string, pre, open int) {
		t := scopes.top()
		inTypeBody := t != nil && slices.Contains(javaTypeKinds, t.kind) && pre > t.pushDepth

		// Import (always at top level, depth == 0); the full dotted path is kept as the package label.
		if pre == 0 {
			if m := reJavaImport.FindStringSubmatch(line); m != nil {
				g.addImport(strings.TrimSpace(m[1]))
			}
		}

		// Type declarations (class/interface/enum) can nest; otherwise look for a method in a type body.
		if m := findWith(reJavaClass, line, "class"); m != nil {
			g.addHeritage(g.declare(m[1], "class", i+1), line, reJavaExtends, reJavaImplements)
			scopes.push("class", m[1], pre)
		} else if m := findWith(reJavaInterface, line, "interface"); m != nil {
			g.declare(m[1], "interface", i+1)
			scopes.push("interface", m[1], pre)
		} else if m := findWith(reJavaEnum, line, "enum"); m != nil {
			g.declare(m[1], "enum", i+1)
			scopes.push("enum", m[1], pre)
		} else if inTypeBody {
			if m := findWith(reJavaMethod, line, "("); m != nil && !isJavaKeyword(m[1]) {
				label := scopes.enclosing(javaTypeKinds...) + "." + m[1]
				id := g.declare(label, "method", i+1, m[1])
				// Only record as a function if it has a body (not abstract/interface).
				if open > 0 || !strings.HasSuffix(strings.TrimRight(line, " \t"), ";") {
					funcs = append(funcs, funcRecord{lineStart: i, id: id})
					scopes.push("method", label, pre)
				}
			}
		}
	})

	g.addBraceCalls(lines, funcs, false, reJavaCall, isJavaKeyword)
	return g.nodes, g.edges, nil
}

var javaKeywords = map[string]bool{
	"if": true, "else": true, "for": true, "while": true, "do": true,
	"switch": true, "case": true, "break": true, "continue": true, "return": true,
	"new": true, "this": true, "super": true, "throw": true, "throws": true,
	"try": true, "catch": true, "finally": true, "class": true, "interface": true,
	"enum": true, "extends": true, "implements": true, "import": true, "package": true,
	"public": true, "protected": true, "private": true, "static": true, "final": true,
	"abstract": true, "synchronized": true, "native": true, "void": true,
	"boolean": true, "byte": true, "char": true, "short": true, "int": true,
	"long": true, "float": true, "double": true, "null": true, "true": true, "false": true,
	"instanceof": true, "assert": true, "default": true, "volatile": true, "transient": true,
	"String": true, "System": true, "Object": true, "Integer": true, "List": true,
	"Map": true, "Set": true, "Arrays": true, "Collections": true,
}

func isJavaKeyword(s string) bool { return javaKeywords[s] }
