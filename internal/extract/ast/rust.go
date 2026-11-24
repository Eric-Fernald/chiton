package ast

import (
	"context"
	"regexp"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

var (
	reRsUse    = regexp.MustCompile(`^\s*(?:pub\s+)?use\s+([^;]+);`)
	reRsExtern = regexp.MustCompile(`^\s*extern\s+crate\s+(\w+)`)
	reRsFn     = regexp.MustCompile(`^\s*(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)`)
	reRsStruct = regexp.MustCompile(`^\s*(?:pub\s+)?struct\s+(\w+)`)
	reRsEnum   = regexp.MustCompile(`^\s*(?:pub\s+)?enum\s+(\w+)`)
	reRsTrait  = regexp.MustCompile(`^\s*(?:pub\s+)?trait\s+(\w+)`)
	reRsImpl   = regexp.MustCompile(`^\s*impl(?:<[^>]*>)?\s+(?:(\w+)\s+for\s+)?(\w+)`)
	reRsCall   = regexp.MustCompile(`\b([a-zA-Z_]\w*)\s*\(`)
	reRsMod    = regexp.MustCompile(`^\s*(?:pub\s+)?mod\s+(\w+)`)
)

// RustExtractor parses Rust source files using regex-based line scanning.
type RustExtractor struct{}

func (r *RustExtractor) ExtractFile(ctx context.Context, path string) ([]model.Node, []model.Edge, error) {
	return extractFile(ctx, r, path)
}

func (r *RustExtractor) ExtractSource(_ context.Context, path string, src []byte) ([]model.Node, []model.Edge, error) {
	g := newFileGraph(newFileNode(path, src, map[string]any{"language": "rust"}))
	lines := splitLines(src)
	var scopes braceScopes // scope kinds are "impl" (named for the implemented type), "fn", and "mod"
	var funcs []funcRecord

	// Pass 1 collects symbols and skips // lines, though scan still tracks their braces and /* */ state.
	isComment := func(s string) bool { return strings.HasPrefix(s, "//") }
	scopes.scan(lines, isComment, func(i int, line string, pre, open int) {
		fn := reRsFn.FindStringSubmatch(line)
		if t := scopes.top(); t != nil && t.kind == "impl" && pre > t.pushDepth && fn != nil {
			// fn inside impl → method
			label := t.name + "." + fn[1]
			funcs = append(funcs, funcRecord{lineStart: i, id: g.declare(label, "method", i+1, fn[1])})
			scopes.push("fn", "", pre)
			return
		}

		if fn != nil {
			funcs = append(funcs, funcRecord{lineStart: i, id: g.declare(fn[1], "func", i+1)})
			scopes.push("fn", "", pre)
		} else if m := reRsImpl.FindStringSubmatch(line); m != nil {
			traitName := m[1] // empty for a bare impl
			typeName := m[2]
			scopes.push("impl", typeName, pre)

			if traitName != "" {
				// impl Trait for Type becomes an IMPLEMENTS edge
				typeID := g.symbols[typeName]
				if typeID == "" {
					typeID = model.NewNodeID(g.uri, typeName)
				}
				g.addExternal(typeID, traitName, implementsEdge)
			}
		} else if m := reRsStruct.FindStringSubmatch(line); m != nil {
			g.declare(m[1], "struct", i+1)
			scopes.push("mod", "", pre)
		} else if m := reRsEnum.FindStringSubmatch(line); m != nil {
			g.declare(m[1], "enum", i+1)
			scopes.push("mod", "", pre)
		} else if m := reRsTrait.FindStringSubmatch(line); m != nil {
			g.declare(m[1], "trait", i+1)
			scopes.push("mod", "", pre)
		} else if m := reRsMod.FindStringSubmatch(line); m != nil {
			g.declare(m[1], "mod", i+1)
			if open > 0 {
				scopes.push("mod", "", pre)
			}
		}

		// use declarations become import nodes
		if m := reRsUse.FindStringSubmatch(line); m != nil {
			if pkg := rustUsePath(m[1]); pkg != "" {
				g.addImport(pkg)
			}
		}
		if m := reRsExtern.FindStringSubmatch(line); m != nil {
			g.addImport(m[1])
		}
	})

	g.addBraceCalls(lines, funcs, false, reRsCall, isRsKeyword)
	return g.nodes, g.edges, nil
}

// rustUsePath extracts the crate name from a use path like "std::fs::File" → "std".
func rustUsePath(s string) string {
	s = strings.TrimSpace(s)
	head, _, found := strings.Cut(s, "::")
	if !found && strings.Contains(s, "{") {
		return "" // a bare grouped import like {A, B} names no crate
	}
	return strings.TrimSpace(head)
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
