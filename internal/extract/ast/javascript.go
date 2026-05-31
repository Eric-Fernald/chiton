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

func (j *JavaScriptExtractor) ExtractFile(_ context.Context, path string) ([]model.Node, []model.Edge, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("read %s: %w", path, err)
	}

	absPath, _ := filepath.Abs(path)
	uri := "file://" + filepath.ToSlash(absPath)
	sum := sha256.Sum256(src)

	lang := "javascript"
	lower := strings.ToLower(path)
	if strings.HasSuffix(lower, ".ts") || strings.HasSuffix(lower, ".tsx") {
		lang = "typescript"
	}

	fileNode := model.Node{
		ID:            model.NewNodeID(uri, uri),
		Kind:          model.KindFile,
		Label:         filepath.Base(path),
		SourceURI:     uri,
		Provenance:    model.ProvExtracted,
		Confidence:    1.0,
		ContentSHA256: fmt.Sprintf("%x", sum[:]),
		Attrs:         map[string]any{"language": lang},
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

	symbols := map[string]string{}

	lines := strings.Split(strings.ReplaceAll(string(src), "\r\n", "\n"), "\n")

	type scopeFrame struct {
		kind      string // "class" or "func"
		name      string // for "class" this is the class name
		id        string // nodeID (for funcs/methods)
		pushDepth int    // brace depth before the opening {
	}

	type funcRecord struct {
		lineStart int
		id        string
	}

	var scopeStack []scopeFrame
	depth := 0
	var funcs []funcRecord

	top := func() *scopeFrame {
		if len(scopeStack) == 0 {
			return nil
		}
		return &scopeStack[len(scopeStack)-1]
	}

	enclosingClassName := func() string {
		for i := len(scopeStack) - 1; i >= 0; i-- {
			if scopeStack[i].kind == "class" {
				return scopeStack[i].name
			}
		}
		return ""
	}

	makeSymbol := func(label, langKind string, lineNum int) model.Node {
		return model.Node{
			ID:         model.NewNodeID(uri, label),
			Kind:       model.KindSymbol,
			Label:      label,
			SourceURI:  uri,
			Span:       &model.SourceSpan{StartLine: lineNum + 1},
			Provenance: model.ProvExtracted,
			Confidence: 1.0,
			Attrs:      map[string]any{"lang_kind": langKind},
		}
	}

	// Pass 1: collect symbols.
	inBlock := false
	for lineNum, line := range lines {
		pre := depth
		startedInBlock := inBlock
		var open, close_ int
		open, close_, inBlock = countLineBraces(line, inBlock)

		// A line wholly inside a /* */ block contributes no symbols; keep the
		// (now comment-adjusted) depth and scope stack current and move on.
		if startedInBlock {
			depth = pre + open - close_
			for len(scopeStack) > 0 && depth <= scopeStack[len(scopeStack)-1].pushDepth {
				scopeStack = scopeStack[:len(scopeStack)-1]
			}
			continue
		}

		// Determine if we're in a class body.
		t := top()
		inClassBody := t != nil && t.kind == "class" && pre > t.pushDepth

		handled := false

		if inClassBody {
			if m := reJSMethod.FindStringSubmatch(line); m != nil && !isJSKeyword(m[1]) {
				className := enclosingClassName()
				label := className + "." + m[1]
				sym := makeSymbol(label, "method", lineNum)
				addNode(sym)
				symbols[m[1]] = sym.ID
				symbols[label] = sym.ID
				addEdge(containsEdge(fileNode.ID, sym.ID))
				funcs = append(funcs, funcRecord{lineStart: lineNum, id: sym.ID})
				scopeStack = append(scopeStack, scopeFrame{kind: "func", name: label, id: sym.ID, pushDepth: pre})
				handled = true
			}
		}

		if !handled {
			name, kind := jsTopLevelDecl(line)
			if name != "" && !isJSKeyword(name) {
				sym := makeSymbol(name, kind, lineNum)
				addNode(sym)
				symbols[name] = sym.ID
				addEdge(containsEdge(fileNode.ID, sym.ID))

				switch kind {
				case "class":
					if m := reJSExtends.FindStringSubmatch(line); m != nil {
						base := importNode(m[1])
						addNode(base)
						addEdge(extendsEdge(sym.ID, base.ID))
					}
					if m := reJSImplements.FindStringSubmatch(line); m != nil {
						for _, iface := range strings.Split(m[1], ",") {
							iface = strings.TrimSpace(iface)
							if iface != "" {
								in := importNode(iface)
								addNode(in)
								addEdge(implementsEdge(sym.ID, in.ID))
							}
						}
					}
					scopeStack = append(scopeStack, scopeFrame{kind: "class", name: name, id: sym.ID, pushDepth: pre})
				case "func", "arrow":
					funcs = append(funcs, funcRecord{lineStart: lineNum, id: sym.ID})
					scopeStack = append(scopeStack, scopeFrame{kind: "func", name: name, id: sym.ID, pushDepth: pre})
				}
			}

			if m := reJSImportFrom.FindStringSubmatch(line); m != nil {
				n := importNode(m[1])
				addNode(n)
				addEdge(importEdge(fileNode.ID, n.ID))
			}
			if m := reJSRequire.FindStringSubmatch(line); m != nil {
				n := importNode(m[1])
				addNode(n)
				addEdge(importEdge(fileNode.ID, n.ID))
			}
		}

		depth = pre + open - close_
		// Pop scopes that were closed.
		for len(scopeStack) > 0 && depth <= scopeStack[len(scopeStack)-1].pushDepth {
			scopeStack = scopeStack[:len(scopeStack)-1]
		}
	}

	ambiguous := ambiguousBareNames(symbols)

	// Pass 2: CALLS edges.
	for _, fr := range funcs {
		bodyDepth := 0
		inBody := false
		inBlock := false
		declIndent := indentLen(lines[fr.lineStart])
		for i := fr.lineStart; i < len(lines); i++ {
			startedInBlock := inBlock
			var open, close_ int
			open, close_, inBlock = countLineBraces(lines[i], inBlock)
			bodyDepth += open
			if bodyDepth > 0 {
				inBody = true
			}
			if inBody && i > fr.lineStart && !startedInBlock {
				for _, m := range reJSCall.FindAllStringSubmatch(lines[i], -1) {
					calleeName := m[1]
					if isJSKeyword(calleeName) {
						continue
					}
					calleeID, ok := symbols[calleeName]
					if !ok || calleeID == fr.id || ambiguous[calleeName] {
						continue
					}
					addEdge(callsEdge(fr.id, calleeID))
				}
			}
			bodyDepth -= close_
			if inBody && bodyDepth <= 0 {
				break
			}
			// Brace-less arrow / single-expression function: once past the decl
			// line without entering a brace body, a non-empty line dedented back to
			// the declaration ends it — stop instead of scanning to EOF.
			if !inBody && i > fr.lineStart {
				if t := strings.TrimSpace(lines[i]); t != "" && indentLen(lines[i]) <= declIndent {
					break
				}
			}
		}
	}

	return nodes, edges, nil
}

// jsTopLevelDecl returns (name, kind) for a top-level declaration on this line.
func jsTopLevelDecl(line string) (string, string) {
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
