package ast

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// ImageExtractor emits a single image node per image file. It deliberately
// does NOT read or describe the pixels at build time — that is the job of the
// vision pass in subagent mode (internal/extract/subagent), which picks up
// image nodes whose attrs["extraction_status"] is unset and emits a
// vision task spec referencing attrs["path"].
type ImageExtractor struct{}

func (im *ImageExtractor) ExtractFile(_ context.Context, path string) ([]model.Node, []model.Edge, error) {
	// Hash the bytes for the content cache, but keep them out of the graph.
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("read %s: %w", path, err)
	}
	sum := sha256.Sum256(data)

	absPath, _ := filepath.Abs(path)
	uri := "file://" + filepath.ToSlash(absPath)

	node := model.Node{
		ID:            model.NewNodeID(uri, uri),
		Kind:          model.KindImage,
		Label:         filepath.Base(path),
		SourceURI:     uri,
		Provenance:    model.ProvExtracted,
		Confidence:    1.0,
		ContentSHA256: fmt.Sprintf("%x", sum[:]),
		Attrs: map[string]any{
			"path":   filepath.ToSlash(absPath),
			"format": imageFormat(path),
		},
	}
	return []model.Node{node}, nil, nil
}

func imageFormat(path string) string {
	ext := filepath.Ext(path)
	if len(ext) > 0 {
		return ext[1:]
	}
	return ""
}
