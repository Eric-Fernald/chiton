package ast

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

var (
	rePyImport     = regexp.MustCompile(`^import\s+(\S+)`)
	rePyFromImport = regexp.MustCompile(`^from\s+(\S+)\s+import\b`)
	rePyClassDef   = regexp.MustCompile(`^(\s*)class\s+(\w+)`)
	rePyClassBases = regexp.MustCompile(`^class\s+\w+\s*\(([^)]+)\):`)
	rePyFuncDef    = regexp.MustCompile(`^(\s*)(?:async\s+)?def\s+(\w+)\s*\(`)
	rePyCall       = regexp.MustCompile(`\b(\w+)\s*\(`)
)

// PythonExtractor parses Python source files using regex-based line scanning.
type PythonExtractor struct{}

func (p *PythonExtractor) ExtractFile(ctx context.Context, path string) ([]model.Node, []model.Edge, error) {
	return extractFile(ctx, p, path)
}

func (p *PythonExtractor) ExtractSource(_ context.Context, path string, src []byte) ([]model.Node, []model.Edge, error) {
	g := newFileGraph(newFileNode(path, src, map[string]any{"language": "python"}))
	lines := splitLines(src)

	type stackEntry struct {
		indent int
		kind   string // "class" or "func"
		name   string
	}

	var stack []stackEntry
	var funcs []funcRecord

	// Pass 1: collect symbols and imports.
	for lineNum, line := range lines {
		trimmed := strings.TrimLeft(line, " \t")
		if len(trimmed) == 0 || strings.HasPrefix(trimmed, "#") {
			continue
		}
		cur := indentLen(line)

		// Pop stack on dedent.
		for len(stack) > 0 && cur <= stack[len(stack)-1].indent {
			stack = stack[:len(stack)-1]
		}

		if m := rePyClassDef.FindStringSubmatch(line); m != nil {
			id := g.declare(m[2], "class", lineNum+1)
			// EXTENDS edge: class Foo(Bar):
			if reBase := rePyClassBases.FindStringSubmatch(line); reBase != nil {
				for base := range strings.SplitSeq(reBase[1], ",") {
					if base = strings.TrimSpace(base); base != "" && base != "object" {
						g.addExternal(id, base, extendsEdge)
					}
				}
			}
			stack = append(stack, stackEntry{indent: len(m[1]), kind: "class", name: m[2]})
			continue
		}

		if m := rePyFuncDef.FindStringSubmatch(line); m != nil {
			indent := len(m[1])
			funcName := m[2]

			// Find enclosing class (nearest class with smaller indent).
			var classCtx string
			for j := len(stack) - 1; j >= 0; j-- {
				if stack[j].kind == "class" && stack[j].indent < indent {
					classCtx = stack[j].name
					break
				}
			}

			label := funcName
			if classCtx != "" {
				label = classCtx + "." + funcName
			}

			id := g.declare(label, "func", lineNum+1, funcName)
			funcs = append(funcs, funcRecord{lineStart: lineNum, indent: indent, id: id})
			stack = append(stack, stackEntry{indent: indent, kind: "func", name: funcName})
			continue
		}

		// Imports at module level (indent == 0).
		if cur == 0 {
			if m := rePyImport.FindStringSubmatch(line); m != nil {
				g.addImport(m[1])
				continue
			}
			if m := rePyFromImport.FindStringSubmatch(line); m != nil {
				g.addImport(m[1])
				continue
			}
		}
	}

	ambiguous := ambiguousBareNames(g.symbols)

	// Pass 2: CALLS edges — scan each function's body lines for calls.
	for _, fr := range funcs {
		for i := fr.lineStart + 1; i < len(lines); i++ {
			bodyLine := lines[i]
			bt := strings.TrimLeft(bodyLine, " \t")
			if len(bt) == 0 || strings.HasPrefix(bt, "#") {
				continue
			}
			if indentLen(bodyLine) <= fr.indent {
				break // exited function body
			}
			g.addLineCalls(bodyLine, fr.id, rePyCall, nil, ambiguous)
		}
	}

	return g.nodes, g.edges, nil
}
