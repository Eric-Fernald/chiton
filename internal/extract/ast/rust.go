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
	reRsUse      = regexp.MustCompile(`^\s*(?:pub\s+)?use\s+([^;]+);`)
	reRsExtern   = regexp.MustCompile(`^\s*extern\s+crate\s+(\w+)`)
	reRsFn       = regexp.MustCompile(`^\s*(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)`)
	reRsStruct   = regexp.MustCompile(`^\s*(?:pub\s+)?struct\s+(\w+)`)
	reRsEnum     = regexp.MustCompile(`^\s*(?:pub\s+)?enum\s+(\w+)`)
	reRsTrait    = regexp.MustCompile(`^\s*(?:pub\s+)?trait\s+(\w+)`)
	reRsImpl     = regexp.MustCompile(`^\s*impl(?:<[^>]*>)?\s+(?:(\w+)\s+for\s+)?(\w+)`)
	reRsCall     = regexp.MustCompile(`\b([a-zA-Z_]\w*)\s*\(`)
	reRsMod      = regexp.MustCompile(`^\s*(?:pub\s+)?mod\s+(\w+)`)
)

// RustExtractor parses Rust source files using regex-based line scanning.
type RustExtractor struct{}

func (r *RustExtractor) ExtractFile(_ context.Context, path string) ([]model.Node, []model.Edge, error) {
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
		Attrs:         map[string]any{"language": "rust"},
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
		kind      string // "impl", "fn", "mod"
		typeName  string // for "impl", the type being implemented
		traitName string // for "impl Trait for Type"
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
		// block-comment state above, so /* */ braces never reach depth tracking.
		if startedInBlock || strings.HasPrefix(stripped, "//") || strings.HasPrefix(stripped, "///") {
			depth = pre + open - close_
			for len(scopeStack) > 0 && depth <= scopeStack[len(scopeStack)-1].pushDepth {
				scopeStack = scopeStack[:len(scopeStack)-1]
			}
			continue
		}

		t := top()
		inImplBody := t != nil && t.kind == "impl" && pre > t.pushDepth

		handled := false

		if inImplBody {
			// fn inside impl → method
			if m := reRsFn.FindStringSubmatch(line); m != nil {
				typeName := t.typeName
				label := typeName + "." + m[1]
				sym := makeSymbol(label, "method", lineNum)
				addNode(sym)
				symbols[m[1]] = sym.ID
				symbols[label] = sym.ID
				addEdge(containsEdge(fileNode.ID, sym.ID))
				funcs = append(funcs, funcRecord{lineStart: lineNum, id: sym.ID})
				scopeStack = append(scopeStack, scopeFrame{kind: "fn", pushDepth: pre})
				handled = true
			}
		}

		if !handled {
			switch {
			case reRsFn.MatchString(line):
				if m := reRsFn.FindStringSubmatch(line); m != nil {
					sym := makeSymbol(m[1], "func", lineNum)
					addNode(sym)
					symbols[m[1]] = sym.ID
					addEdge(containsEdge(fileNode.ID, sym.ID))
					funcs = append(funcs, funcRecord{lineStart: lineNum, id: sym.ID})
					scopeStack = append(scopeStack, scopeFrame{kind: "fn", pushDepth: pre})
				}

			case reRsImpl.MatchString(line):
				if m := reRsImpl.FindStringSubmatch(line); m != nil {
					traitName := m[1] // may be empty for bare impl
					typeName := m[2]
					frame := scopeFrame{kind: "impl", typeName: typeName, traitName: traitName, pushDepth: pre}
					scopeStack = append(scopeStack, frame)

					if traitName != "" {
						// impl Trait for Type → IMPLEMENTS edge
						typeID := symbols[typeName]
						if typeID == "" {
							typeID = model.NewNodeID(uri, typeName)
						}
						traitNode := importNode(traitName)
						addNode(traitNode)
						addEdge(implementsEdge(typeID, traitNode.ID))
					}
				}

			case reRsStruct.MatchString(line):
				if m := reRsStruct.FindStringSubmatch(line); m != nil {
					sym := makeSymbol(m[1], "struct", lineNum)
					addNode(sym)
					symbols[m[1]] = sym.ID
					addEdge(containsEdge(fileNode.ID, sym.ID))
					scopeStack = append(scopeStack, scopeFrame{kind: "mod", pushDepth: pre})
				}

			case reRsEnum.MatchString(line):
				if m := reRsEnum.FindStringSubmatch(line); m != nil {
					sym := makeSymbol(m[1], "enum", lineNum)
					addNode(sym)
					symbols[m[1]] = sym.ID
					addEdge(containsEdge(fileNode.ID, sym.ID))
					scopeStack = append(scopeStack, scopeFrame{kind: "mod", pushDepth: pre})
				}

			case reRsTrait.MatchString(line):
				if m := reRsTrait.FindStringSubmatch(line); m != nil {
					sym := makeSymbol(m[1], "trait", lineNum)
					addNode(sym)
					symbols[m[1]] = sym.ID
					addEdge(containsEdge(fileNode.ID, sym.ID))
					scopeStack = append(scopeStack, scopeFrame{kind: "mod", pushDepth: pre})
				}

			case reRsMod.MatchString(line):
				if m := reRsMod.FindStringSubmatch(line); m != nil {
					sym := makeSymbol(m[1], "mod", lineNum)
					addNode(sym)
					symbols[m[1]] = sym.ID
					addEdge(containsEdge(fileNode.ID, sym.ID))
					if open > 0 {
						scopeStack = append(scopeStack, scopeFrame{kind: "mod", pushDepth: pre})
					}
				}
			}

			// use declarations → import nodes
			if m := reRsUse.FindStringSubmatch(line); m != nil {
				pkg := rustUsePath(m[1])
				if pkg != "" {
					n := importNode(pkg)
					addNode(n)
					addEdge(importEdge(fileNode.ID, n.ID))
				}
			}
			if m := reRsExtern.FindStringSubmatch(line); m != nil {
				n := importNode(m[1])
				addNode(n)
				addEdge(importEdge(fileNode.ID, n.ID))
			}
		}

		depth = pre + open - close_
		for len(scopeStack) > 0 && depth <= scopeStack[len(scopeStack)-1].pushDepth {
			scopeStack = scopeStack[:len(scopeStack)-1]
		}
	}

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
				for _, m := range reRsCall.FindAllStringSubmatch(lines[i], -1) {
					calleeName := m[1]
					if isRsKeyword(calleeName) {
						continue
					}
					calleeID, ok := symbols[calleeName]
					if !ok || calleeID == fr.id {
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

// rustUsePath extracts the crate name from a use path like "std::fs::File" → "std".
func rustUsePath(s string) string {
	s = strings.TrimSpace(s)
	// Handle grouped imports: {A, B} → skip
	if strings.Contains(s, "{") {
		idx := strings.Index(s, "::")
		if idx < 0 {
			return ""
		}
		return strings.TrimSpace(s[:idx])
	}
	parts := strings.SplitN(s, "::", 2)
	if len(parts) == 0 {
		return ""
	}
	return strings.TrimSpace(parts[0])
}

var rsKeywords = map[string]bool{
	"if": true, "else": true, "for": true, "while": true, "loop": true,
	"match": true, "return": true, "let": true, "mut": true, "const": true,
	"pub": true, "fn": true, "struct": true, "enum": true, "trait": true,
	"impl": true, "use": true, "mod": true, "where": true, "type": true,
	"move": true, "async": true, "await": true, "dyn": true, "box": true,
	"ref": true, "in": true, "super": true, "self": true, "crate": true,
	"Some": true, "None": true, "Ok": true, "Err": true, "true": true, "false": true,
	"vec": true, "println": true, "eprintln": true, "print": true, "format": true,
	"panic": true, "assert": true, "dbg": true, "todo": true, "unimplemented": true,
}

func isRsKeyword(s string) bool { return rsKeywords[s] }
