package ast

import (
	"context"
	"path/filepath"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// ImageExtractor emits a single image node per image file. Pixels are never described at build time; the
// subagent vision pass picks up image nodes without attrs["extraction_status"] and reads attrs["path"].
type ImageExtractor struct{}

func (im *ImageExtractor) ExtractFile(ctx context.Context, path string) ([]model.Node, []model.Edge, error) {
	return extractFile(ctx, im, path)
}

// ExtractSource hashes src for the content cache but keeps the bytes out of the graph. attrs["path"] is
// the absolute, forward-slash path (the URI minus its scheme).
func (im *ImageExtractor) ExtractSource(_ context.Context, path string, src []byte) ([]model.Node, []model.Edge, error) {
	node := newFileNode(path, src, nil)
	node.Kind = model.KindImage
	node.Attrs = map[string]any{"path": strings.TrimPrefix(node.SourceURI, "file://"), "format": imageFormat(path)}
	return []model.Node{node}, nil, nil
}

func imageFormat(path string) string {
	ext := filepath.Ext(path)
	if len(ext) > 0 {
		return ext[1:]
	}
	return ""
}
