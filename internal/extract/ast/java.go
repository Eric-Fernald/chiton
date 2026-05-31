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
	reJavaImport    = regexp.MustCompile(`^\s*import\s+(?:static\s+)?([^;]+);`)
	reJavaClass     = regexp.MustCompile(`(?:^|\s)(?:public|protected|private|abstract|final|static|\s)*class\s+(\w+)`)
	reJavaInterface = regexp.MustCompile(`(?:^|\s)(?:public|protected|private|abstract|\s)*interface\s+(\w+)`)
	reJavaEnum      = regexp.MustCompile(`(?:^|\s)(?:public|protected|private|\s)*enum\s+(\w+)`)
	reJavaExtends   = regexp.MustCompile(`\bextends\s+([\w.]+)`)
	reJavaImplements = regexp.MustCompile(`\bimplements\s+([\w.,\s]+?)(?:\{|$)`)

	// Method: modifiers* ReturnType methodName ( — at indented level
	reJavaMethod = regexp.MustCompile(`^\s+(?:(?:public|protected|private|static|final|abstract|synchronized|native|default|override)\s+)*(?:<[^>]*>\s+)?(?:\w[\w.<>\[\]]*\s+)+(\w+)\s*\(`)

	reJavaCall = regexp.MustCompile(`\b([a-zA-Z_]\w*)\s*\(`)
)

// JavaExtractor parses Java source files using regex-based line scanning.
type JavaExtractor struct{}

func (j *JavaExtractor) ExtractFile(_ context.Context, path string) ([]model.Node, []model.Edge, error) {
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
		Attrs:         map[string]any{"language": "java"},
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
		kind      string // "class", "interface", "enum", "method"
		name      string // class/interface/enum name or method label
		id        string // nodeID
		pushDepth int
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
			k := scopeStack[i].kind
			if k == "class" || k == "interface" || k == "enum" {
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

		stripped := strings.TrimLeft(line, " \t")
		// Skip comment lines for symbol detection — but only AFTER updating the
		// block-comment state above, so a closing */ on a "*"-prefixed Javadoc
		// line still clears inBlock instead of leaving it stuck.
		if startedInBlock || strings.HasPrefix(stripped, "//") || strings.HasPrefix(stripped, "*") {
			depth = pre + open - close_
			for len(scopeStack) > 0 && depth <= scopeStack[len(scopeStack)-1].pushDepth {
				scopeStack = scopeStack[:len(scopeStack)-1]
			}
			continue
		}

		t := top()
		inTypeBody := t != nil &&
			(t.kind == "class" || t.kind == "interface" || t.kind == "enum") &&
			pre > t.pushDepth

		handled := false

		// Import (always at top level, depth == 0)
		if pre == 0 {
			if m := reJavaImport.FindStringSubmatch(line); m != nil {
				pkg := javaImportPkg(m[1])
				n := importNode(pkg)
				addNode(n)
				addEdge(importEdge(fileNode.ID, n.ID))
			}
		}

		// Type declarations (class/interface/enum) — can nest.
		if m := reJavaClass.FindStringSubmatch(line); m != nil {
			name := m[1]
			sym := makeSymbol(name, "class", lineNum)
			addNode(sym)
			symbols[name] = sym.ID
			addEdge(containsEdge(fileNode.ID, sym.ID))
			if me := reJavaExtends.FindStringSubmatch(line); me != nil {
				base := importNode(me[1])
				addNode(base)
				addEdge(extendsEdge(sym.ID, base.ID))
			}
			if mi := reJavaImplements.FindStringSubmatch(line); mi != nil {
				for _, iface := range strings.Split(mi[1], ",") {
					iface = strings.TrimSpace(iface)
					if iface != "" {
						in := importNode(iface)
						addNode(in)
						addEdge(implementsEdge(sym.ID, in.ID))
					}
				}
			}
			scopeStack = append(scopeStack, scopeFrame{kind: "class", name: name, id: sym.ID, pushDepth: pre})
			handled = true
		} else if m := reJavaInterface.FindStringSubmatch(line); m != nil {
			name := m[1]
			sym := makeSymbol(name, "interface", lineNum)
			addNode(sym)
			symbols[name] = sym.ID
			addEdge(containsEdge(fileNode.ID, sym.ID))
			scopeStack = append(scopeStack, scopeFrame{kind: "interface", name: name, id: sym.ID, pushDepth: pre})
			handled = true
		} else if m := reJavaEnum.FindStringSubmatch(line); m != nil {
			name := m[1]
			sym := makeSymbol(name, "enum", lineNum)
			addNode(sym)
			symbols[name] = sym.ID
			addEdge(containsEdge(fileNode.ID, sym.ID))
			scopeStack = append(scopeStack, scopeFrame{kind: "enum", name: name, id: sym.ID, pushDepth: pre})
			handled = true
		}

		// Method detection inside a type body.
		if !handled && inTypeBody {
			if m := reJavaMethod.FindStringSubmatch(line); m != nil {
				methodName := m[1]
				if !isJavaKeyword(methodName) {
					className := enclosingClassName()
					label := className + "." + methodName
					sym := makeSymbol(label, "method", lineNum)
					addNode(sym)
					symbols[methodName] = sym.ID
					symbols[label] = sym.ID
					addEdge(containsEdge(fileNode.ID, sym.ID))
					// Only record as a function if it has a body (not abstract/interface).
					if open > 0 || !strings.HasSuffix(strings.TrimRight(line, " \t"), ";") {
						funcs = append(funcs, funcRecord{lineStart: lineNum, id: sym.ID})
						scopeStack = append(scopeStack, scopeFrame{kind: "method", name: label, id: sym.ID, pushDepth: pre})
					}
				}
			}
		}

		depth = pre + open - close_
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
		for i := fr.lineStart; i < len(lines); i++ {
			startedInBlock := inBlock
			var open, close_ int
			open, close_, inBlock = countLineBraces(lines[i], inBlock)
			bodyDepth += open
			if bodyDepth > 0 {
				inBody = true
			}
			if inBody && i > fr.lineStart && !startedInBlock {
				for _, m := range reJavaCall.FindAllStringSubmatch(lines[i], -1) {
					calleeName := m[1]
					if isJavaKeyword(calleeName) {
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
		}
	}

	return nodes, edges, nil
}

// javaImportPkg simplifies a Java import path to just the package (drops the class name).
func javaImportPkg(imp string) string {
	imp = strings.TrimSpace(imp)
	// For "java.util.List" return "java.util.List" — keep full path for accuracy.
	return imp
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
