package ast

import (
	"bytes"
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// utf8BOM is the UTF-8 byte-order mark, common on Windows-authored files. We
// strip it so a leading "# Heading" isn't hidden behind the BOM bytes.
var utf8BOM = []byte{0xEF, 0xBB, 0xBF}

// MarkdownExtractor emits a file node plus one doc_section per ATX heading, nested via CONTAINS by heading
// level. It keeps each section's body in attrs["text"] for embedding and subagent concept extraction.
type MarkdownExtractor struct{}

func (m *MarkdownExtractor) ExtractFile(ctx context.Context, path string) ([]model.Node, []model.Edge, error) {
	return extractFile(ctx, m, path)
}

func (m *MarkdownExtractor) ExtractSource(_ context.Context, path string, src []byte) ([]model.Node, []model.Edge, error) {
	src = bytes.TrimPrefix(src, utf8BOM)

	fileNode := newFileNode(path, src, map[string]any{"language": "markdown"})
	uri := fileNode.SourceURI

	nodes := []model.Node{fileNode}
	var edges []model.Edge

	lines := splitLines(src)

	type heading struct {
		level int
		text  string
		line  int // 1-based
	}
	var headings []heading
	inFence := false
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			inFence = !inFence
			continue
		}
		if inFence {
			continue
		}
		if lvl, text, ok := parseATXHeading(line); ok {
			headings = append(headings, heading{level: lvl, text: text, line: i + 1})
		}
	}

	// A doc with no headings still becomes one embeddable section covering the
	// whole file, so its prose is retrievable.
	if len(headings) == 0 {
		body := strings.TrimSpace(string(src))
		if body == "" {
			return nodes, edges, nil
		}
		sec := docSectionNode(uri, filepath.Base(path), 1, 0, body)
		nodes = append(nodes, sec)
		edges = append(edges, containsEdge(fileNode.ID, sec.ID))
		return nodes, edges, nil
	}

	// Build a section per heading; body is the lines between this heading and
	// the next. A stack of (level, nodeID) reconstructs nesting for CONTAINS.
	type stackEntry struct {
		level int
		id    string
	}
	var stack []stackEntry

	for hi, h := range headings {
		end := len(lines)
		if hi+1 < len(headings) {
			end = headings[hi+1].line - 1
		}
		// h.line is 1-based, so lines[h.line] is the first line after the heading.
		body := strings.TrimSpace(strings.Join(lines[h.line:end], "\n"))

		sec := docSectionNode(uri, h.text, h.line, h.level, body)
		nodes = append(nodes, sec)

		// Pop deeper-or-equal sections; parent is the nearest shallower one.
		for len(stack) > 0 && stack[len(stack)-1].level >= h.level {
			stack = stack[:len(stack)-1]
		}
		parentID := fileNode.ID
		if len(stack) > 0 {
			parentID = stack[len(stack)-1].id
		}
		edges = append(edges, containsEdge(parentID, sec.ID))
		stack = append(stack, stackEntry{level: h.level, id: sec.ID})
	}

	return nodes, edges, nil
}

// docSectionNode builds a doc_section node. The canonical label embeds the
// start line so duplicate headings produce distinct, stable IDs.
func docSectionNode(uri, heading string, startLine, level int, body string) model.Node {
	canonical := fmt.Sprintf("section@L%d:%s", startLine, heading)
	attrs := map[string]any{
		"heading": heading,
		"level":   level,
	}
	if body != "" {
		attrs["text"] = body
	}
	return model.Node{
		ID:         model.NewNodeID(uri, canonical),
		Kind:       model.KindDocSection,
		Label:      heading,
		SourceURI:  uri,
		Span:       &model.SourceSpan{StartLine: startLine},
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		Attrs:      attrs,
	}
}

// parseATXHeading recognises "# Heading" through "###### Heading". It requires
// a space after the hashes (CommonMark) and strips trailing closing hashes.
func parseATXHeading(line string) (level int, text string, ok bool) {
	i := 0
	for i < len(line) && line[i] == '#' {
		i++
	}
	if i == 0 || i > 6 {
		return 0, "", false
	}
	if i >= len(line) || (line[i] != ' ' && line[i] != '\t') {
		return 0, "", false
	}
	text = strings.TrimSpace(line[i:])
	text = strings.TrimRight(text, "#")
	text = strings.TrimSpace(text)
	if text == "" {
		return 0, "", false
	}
	return i, text, true
}
