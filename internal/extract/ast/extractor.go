// Package ast extracts graph nodes and edges from source files.
package ast

import (
	"context"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// Extractor extracts graph nodes and edges from a single source file.
type Extractor interface {
	ExtractFile(ctx context.Context, path string) ([]model.Node, []model.Edge, error)
}

// ForFile returns the Extractor for the given file's language, or nil if unsupported.
func ForFile(path string) Extractor {
	lower := strings.ToLower(path)
	switch {
	case strings.HasSuffix(lower, ".go"):
		return &GoExtractor{}
	case strings.HasSuffix(lower, ".py"):
		return &PythonExtractor{}
	case strings.HasSuffix(lower, ".js"),
		strings.HasSuffix(lower, ".ts"),
		strings.HasSuffix(lower, ".jsx"),
		strings.HasSuffix(lower, ".tsx"):
		return &JavaScriptExtractor{}
	case strings.HasSuffix(lower, ".rs"):
		return &RustExtractor{}
	case strings.HasSuffix(lower, ".java"):
		return &JavaExtractor{}
	case strings.HasSuffix(lower, ".md"),
		strings.HasSuffix(lower, ".markdown"):
		return &MarkdownExtractor{}
	case strings.HasSuffix(lower, ".png"),
		strings.HasSuffix(lower, ".jpg"),
		strings.HasSuffix(lower, ".jpeg"),
		strings.HasSuffix(lower, ".gif"),
		strings.HasSuffix(lower, ".webp"):
		return &ImageExtractor{}
	}
	return nil
}

// --- shared helpers ---

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

// countLineBraces counts unescaped { and } in a line, ignoring string literals,
// // line comments, and /* */ block comments. inBlock says whether the line
// begins inside a block comment opened on an earlier line; stillInBlock reports
// whether it ends inside one, so callers thread the state line-to-line. Without
// this, a brace inside a comment (e.g. JSDoc or commented-out code) is miscounted
// and prematurely pops the enclosing scope.
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
