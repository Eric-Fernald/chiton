package ast

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

var (
	rePyImport     = regexp.MustCompile(`^import\s+(\S+)`)
	rePyFromImport = regexp.MustCompile(`^from\s+(\S+)\s+import\b`)
	rePyClassDef   = regexp.MustCompile(`^(\s*)class\s+(\w+)`)
	rePyFuncDef    = regexp.MustCompile(`^(\s*)(?:async\s+)?def\s+(\w+)\s*\(`)
	rePyCall       = regexp.MustCompile(`\b(\w+)\s*\(`)
)

// PythonExtractor parses Python source files using regex-based line scanning.
type PythonExtractor struct{}

func (p *PythonExtractor) ExtractFile(_ context.Context, path string) ([]model.Node, []model.Edge, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("read %s: %w", path, err)
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
		Attrs:         map[string]any{"language": "python"},
	}

	nodes := []model.Node{fileNode}
	var edges []model.Edge
	seenEdge := map[string]bool{}

	addEdge := func(e model.Edge) {
		if !seenEdge[e.ID] {
			seenEdge[e.ID] = true
			edges = append(edges, e)
		}
	}
	seenNode := map[string]bool{fileNode.ID: true}
	addNode := func(n model.Node) {
		if !seenNode[n.ID] {
			seenNode[n.ID] = true
			nodes = append(nodes, n)
		}
	}

	// symbols maps bare name and qualified name → nodeID for CALLS resolution.
	symbols := map[string]string{}

	lines := strings.Split(strings.ReplaceAll(string(src), "\r\n", "\n"), "\n")

	type stackEntry struct {
		indent int
		kind   string // "class" or "func"
		name   string
	}

	// funcRecord tracks where each function was defined for body scanning.
	type funcRecord struct {
		lineStart int
		indent    int
		id        string
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
			indent := len(m[1])
			className := m[2]
			sym := model.Node{
				ID:         model.NewNodeID(uri, className),
				Kind:       model.KindSymbol,
				Label:      className,
				SourceURI:  uri,
				Span:       &model.SourceSpan{StartLine: lineNum + 1},
				Provenance: model.ProvExtracted,
				Confidence: 1.0,
				Attrs:      map[string]any{"lang_kind": "class"},
			}
			addNode(sym)
			symbols[className] = sym.ID
			addEdge(containsEdge(fileNode.ID, sym.ID))

			// EXTENDS edge: class Foo(Bar):
			if reBase := regexp.MustCompile(`^class\s+\w+\s*\(([^)]+)\):`).FindStringSubmatch(line); reBase != nil {
				for _, base := range strings.Split(reBase[1], ",") {
					base = strings.TrimSpace(base)
					if base == "" || base == "object" {
						continue
					}
					bn := importNode(base)
					addNode(bn)
					addEdge(extendsEdge(sym.ID, bn.ID))
				}
			}

			stack = append(stack, stackEntry{indent: indent, kind: "class", name: className})
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

			sym := model.Node{
				ID:         model.NewNodeID(uri, label),
				Kind:       model.KindSymbol,
				Label:      label,
				SourceURI:  uri,
				Span:       &model.SourceSpan{StartLine: lineNum + 1},
				Provenance: model.ProvExtracted,
				Confidence: 1.0,
				Attrs:      map[string]any{"lang_kind": "func"},
			}
			addNode(sym)
			symbols[funcName] = sym.ID
			symbols[label] = sym.ID
			addEdge(containsEdge(fileNode.ID, sym.ID))

			funcs = append(funcs, funcRecord{lineStart: lineNum, indent: indent, id: sym.ID})
			stack = append(stack, stackEntry{indent: indent, kind: "func", name: funcName})
			continue
		}

		// Imports at module level (indent == 0).
		if cur == 0 {
			if m := rePyImport.FindStringSubmatch(line); m != nil {
				n := importNode(m[1])
				addNode(n)
				addEdge(importEdge(fileNode.ID, n.ID))
				continue
			}
			if m := rePyFromImport.FindStringSubmatch(line); m != nil {
				n := importNode(m[1])
				addNode(n)
				addEdge(importEdge(fileNode.ID, n.ID))
				continue
			}
		}
	}

	ambiguous := ambiguousBareNames(symbols)

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
			for _, m := range rePyCall.FindAllStringSubmatch(bodyLine, -1) {
				calleeName := m[1]
				calleeID, ok := symbols[calleeName]
				if !ok || calleeID == fr.id || ambiguous[calleeName] {
					continue
				}
				addEdge(callsEdge(fr.id, calleeID))
			}
		}
	}

	return nodes, edges, nil
}
