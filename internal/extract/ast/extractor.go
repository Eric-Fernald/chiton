// Package ast extracts graph nodes and edges from source files.
package ast

import (
	"context"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// Extractor extracts graph nodes and edges from a single source file.
type Extractor interface {
	ExtractFile(ctx context.Context, path string) ([]model.Node, []model.Edge, error)
	// ExtractSource is ExtractFile for a caller that has already read path's contents into src.
	ExtractSource(ctx context.Context, path string, src []byte) ([]model.Node, []model.Edge, error)
}

// extractFile is every extractor's ExtractFile: read path, then hand the bytes to ExtractSource.
func extractFile(ctx context.Context, ex Extractor, path string) ([]model.Node, []model.Edge, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("read %s: %w", path, err)
	}
	return ex.ExtractSource(ctx, path, src)
}

// ForFile returns the Extractor for the given file's language, or nil if unsupported.
func ForFile(path string) Extractor {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".go":
		return &GoExtractor{}
	case ".py":
		return &PythonExtractor{}
	case ".js", ".ts", ".jsx", ".tsx":
		return &JavaScriptExtractor{}
	case ".rs":
		return &RustExtractor{}
	case ".java":
		return &JavaExtractor{}
	case ".md", ".markdown":
		return &MarkdownExtractor{}
	case ".png", ".jpg", ".jpeg", ".gif", ".webp":
		return &ImageExtractor{}
	}
	return nil
}

// --- shared helpers ---

// newFileNode builds the file node an extractor roots its graph at, keyed on the absolute file:// URI.
func newFileNode(path string, src []byte, attrs map[string]any) model.Node {
	uri := model.FileURI(path)
	return model.Node{
		ID:            model.NewNodeID(uri, uri),
		Kind:          model.KindFile,
		Label:         filepath.Base(path),
		SourceURI:     uri,
		Provenance:    model.ProvExtracted,
		Confidence:    1.0,
		ContentSHA256: model.ContentSHA(src),
		Attrs:         attrs,
	}
}

// findWith runs re on line only if line contains lit, a literal every match of re includes; the cheap
// substring check spares unanchored patterns a full-line scan on the many lines that cannot match.
func findWith(re *regexp.Regexp, line, lit string) []string {
	if !strings.Contains(line, lit) {
		return nil
	}
	return re.FindStringSubmatch(line)
}

// splitLines splits src into lines, treating CRLF as LF.
func splitLines(src []byte) []string {
	return strings.Split(strings.ReplaceAll(string(src), "\r\n", "\n"), "\n")
}

// fileGraph accumulates one file's nodes and edges, dropping repeats by ID. symbols maps each declared
// symbol's bare and qualified names to its node ID for CALLS resolution.
type fileGraph struct {
	uri, fileID string
	nodes       []model.Node
	edges       []model.Edge
	symbols     map[string]string
	seenNode    map[string]bool
	seenEdge    map[string]bool
}

func newFileGraph(file model.Node) *fileGraph {
	return &fileGraph{
		uri:      file.SourceURI,
		fileID:   file.ID,
		nodes:    []model.Node{file},
		symbols:  map[string]string{},
		seenNode: map[string]bool{file.ID: true},
		seenEdge: map[string]bool{},
	}
}

// declare adds a symbol node declared at 1-based line plus the file's CONTAINS edge to it, registers it
// under label and each alias, and returns its ID.
func (g *fileGraph) declare(label, langKind string, line int, aliases ...string) string {
	sym := model.Node{
		ID:         model.NewNodeID(g.uri, label),
		Kind:       model.KindSymbol,
		Label:      label,
		SourceURI:  g.uri,
		Span:       &model.SourceSpan{StartLine: line},
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		Attrs:      map[string]any{"lang_kind": langKind},
	}
	g.addNode(sym)
	for _, a := range aliases {
		g.symbols[a] = sym.ID
	}
	g.symbols[label] = sym.ID
	g.addEdge(containsEdge(g.fileID, sym.ID))
	return sym.ID
}

func (g *fileGraph) addNode(n model.Node) {
	if !g.seenNode[n.ID] {
		g.seenNode[n.ID] = true
		g.nodes = append(g.nodes, n)
	}
}

func (g *fileGraph) addEdge(e model.Edge) {
	if !g.seenEdge[e.ID] {
		g.seenEdge[e.ID] = true
		g.edges = append(g.edges, e)
	}
}

// addExternal adds the external symbol node for pkg and an edge (built by mkEdge) from srcID to it.
func (g *fileGraph) addExternal(srcID, pkg string, mkEdge func(srcID, dstID string) model.Edge) {
	n := importNode(pkg)
	g.addNode(n)
	g.addEdge(mkEdge(srcID, n.ID))
}

// addImport adds an external package node and the file's IMPORTS edge to it.
func (g *fileGraph) addImport(pkg string) {
	g.addExternal(g.fileID, pkg, importEdge)
}

// addHeritage adds symID's EXTENDS edge to the base reExt names on line and an IMPLEMENTS edge to each
// comma-separated interface reImpl names.
func (g *fileGraph) addHeritage(symID, line string, reExt, reImpl *regexp.Regexp) {
	if m := reExt.FindStringSubmatch(line); m != nil {
		g.addExternal(symID, m[1], extendsEdge)
	}
	if m := reImpl.FindStringSubmatch(line); m != nil {
		for iface := range strings.SplitSeq(m[1], ",") {
			if iface = strings.TrimSpace(iface); iface != "" {
				g.addExternal(symID, iface, implementsEdge)
			}
		}
	}
}

// addLineCalls adds a CALLS edge from callerID for each call on line naming exactly one known symbol.
func (g *fileGraph) addLineCalls(line, callerID string, reCall *regexp.Regexp, isKeyword func(string) bool,
	ambiguous map[string]bool) {
	if strings.IndexByte(line, '(') < 0 {
		return // every call pattern requires "("
	}
	for _, m := range reCall.FindAllStringSubmatch(line, -1) {
		name := m[1]
		if isKeyword != nil && isKeyword(name) {
			continue
		}
		calleeID, ok := g.symbols[name]
		if !ok || calleeID == callerID || ambiguous[name] {
			continue
		}
		g.addEdge(callsEdge(callerID, calleeID))
	}
}

// funcRecord is a function whose body pass 2 scans for calls; indent is used only by Python.
type funcRecord struct {
	lineStart, indent int
	id                string
}

// addBraceCalls adds CALLS edges from each of funcs to the known symbols its brace-delimited body calls.
func (g *fileGraph) addBraceCalls(lines []string, funcs []funcRecord, dedentEnds bool, reCall *regexp.Regexp,
	isKeyword func(string) bool) {
	ambiguous := ambiguousBareNames(g.symbols)
	for _, fr := range funcs {
		for line := range braceBodyLines(lines, fr.lineStart, dedentEnds) {
			g.addLineCalls(line, fr.id, reCall, isKeyword, ambiguous)
		}
	}
}

// scopeFrame is an open brace scope; pushDepth is the brace depth before its opening {.
type scopeFrame struct {
	kind, name string
	pushDepth  int
}

// braceScopes is the scope stack a brace language's symbol pass keeps while scanning.
type braceScopes []scopeFrame

func (s *braceScopes) push(kind, name string, pushDepth int) {
	*s = append(*s, scopeFrame{kind: kind, name: name, pushDepth: pushDepth})
}

// top returns the innermost open scope, or nil.
func (s braceScopes) top() *scopeFrame {
	if len(s) == 0 {
		return nil
	}
	return &s[len(s)-1]
}

// enclosing returns the name of the innermost open scope of one of kinds, or "".
func (s braceScopes) enclosing(kinds ...string) string {
	for i := len(s) - 1; i >= 0; i-- {
		if slices.Contains(kinds, s[i].kind) {
			return s[i].name
		}
	}
	return ""
}

// scan calls visit(i, line, depth before line, its { count) for each line not opening inside /* */ nor
// matched by skip (given the line left-trimmed), then pops the scopes the line closed.
func (s *braceScopes) scan(lines []string, skip func(trimmed string) bool, visit func(i int, line string, pre, open int)) {
	depth, inBlock := 0, false
	for i, line := range lines {
		startedInBlock := inBlock
		var open, closeN int
		open, closeN, inBlock = countLineBraces(line, inBlock)
		if !startedInBlock && (skip == nil || !skip(strings.TrimLeft(line, " \t"))) {
			visit(i, line, depth, open)
		}
		depth += open - closeN
		for len(*s) > 0 && depth <= (*s)[len(*s)-1].pushDepth {
			*s = (*s)[:len(*s)-1]
		}
	}
}

// braceBodyLines yields the body lines (past the declaration, not opening inside /* */) of the block declared
// at lines[start]. With dedentEnds, a brace-less body (JS arrow) ends at a non-blank line not indented past it.
func braceBodyLines(lines []string, start int, dedentEnds bool) iter.Seq[string] {
	return func(yield func(string) bool) {
		depth, inBody, inBlock := 0, false, false
		declIndent := indentLen(lines[start])
		for i := start; i < len(lines); i++ {
			startedInBlock := inBlock
			var open, closeN int
			open, closeN, inBlock = countLineBraces(lines[i], inBlock)
			depth += open
			if depth > 0 {
				inBody = true
			}
			if inBody && i > start && !startedInBlock && !yield(lines[i]) {
				return
			}
			depth -= closeN
			if inBody && depth <= 0 {
				return
			}
			if dedentEnds && !inBody && i > start {
				if t := strings.TrimSpace(lines[i]); t != "" && indentLen(lines[i]) <= declIndent {
					return
				}
			}
		}
	}
}

func containsEdge(fileID, symID string) model.Edge {
	return model.Edge{
		ID:         model.NewEdgeID(fileID, symID, model.EdgeContains),
		Src:        fileID,
		Dst:        symID,
		Kind:       model.EdgeContains,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
	}
}

func callsEdge(srcID, dstID string) model.Edge {
	return model.Edge{
		ID:         model.NewEdgeID(srcID, dstID, model.EdgeCalls),
		Src:        srcID,
		Dst:        dstID,
		Kind:       model.EdgeCalls,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
	}
}

func extendsEdge(srcID, dstID string) model.Edge {
	return model.Edge{
		ID:         model.NewEdgeID(srcID, dstID, model.EdgeExtends),
		Src:        srcID,
		Dst:        dstID,
		Kind:       model.EdgeExtends,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
	}
}

func implementsEdge(srcID, dstID string) model.Edge {
	return model.Edge{
		ID:         model.NewEdgeID(srcID, dstID, model.EdgeImplements),
		Src:        srcID,
		Dst:        dstID,
		Kind:       model.EdgeImplements,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
	}
}

func importNode(pkg string) model.Node {
	pkgURI := "pkg://" + pkg
	return model.Node{
		ID:         model.NewNodeID(pkgURI, pkg),
		Kind:       model.KindSymbol,
		Label:      pkg,
		SourceURI:  pkgURI,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		Attrs:      map[string]any{"external": true},
	}
}

func importEdge(fileID, pkgNodeID string) model.Edge {
	return model.Edge{
		ID:         model.NewEdgeID(fileID, pkgNodeID, model.EdgeImports),
		Src:        fileID,
		Dst:        pkgNodeID,
		Kind:       model.EdgeImports,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
	}
}

// countLineBraces counts { and } outside strings and comments; inBlock threads /* */ state across lines so a
// brace inside a comment (JSDoc, commented-out code) never pops the enclosing scope.
func countLineBraces(line string, inBlock bool) (open, close int, stillInBlock bool) {
	var inStr byte // 0, '"', '\'', or '`'
	blk := inBlock
	for i := 0; i < len(line); i++ {
		c := line[i]
		if blk {
			// Inside /* ... */ — everything is comment until the closing */.
			if c == '*' && i+1 < len(line) && line[i+1] == '/' {
				blk = false
				i++
			}
			continue
		}
		if inStr != 0 {
			if c == '\\' {
				i++ // skip escaped char
			} else if c == inStr {
				inStr = 0
			}
			continue
		}
		switch c {
		case '"', '\'', '`':
			inStr = c
		case '/':
			if i+1 < len(line) && line[i+1] == '/' {
				return open, close, false // line comment — rest of line ignored
			}
			if i+1 < len(line) && line[i+1] == '*' {
				blk = true
				i++
			}
		case '{':
			open++
		case '}':
			close++
		}
	}
	return open, close, blk
}

// indentLen returns the number of leading spaces/tabs in s.
func indentLen(s string) int {
	return len(s) - len(strings.TrimLeft(s, " \t"))
}

// ambiguousBareNames returns bare names shared by more than one symbol (two classes' "parse" methods), so
// pass-2 callers skip the CALLS edge rather than point it at whichever symbol was registered last.
func ambiguousBareNames(symbols map[string]string) map[string]bool {
	ambiguous := map[string]bool{}
	firstID := map[string]string{}
	for key, id := range symbols {
		bare := key
		if i := strings.LastIndex(key, "."); i >= 0 {
			bare = key[i+1:]
		}
		if prev, ok := firstID[bare]; ok {
			if prev != id {
				ambiguous[bare] = true
			}
		} else {
			firstID[bare] = id
		}
	}
	return ambiguous
}
