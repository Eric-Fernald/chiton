// Package embed provides pluggable embedding providers (local Ollama by default, or a deterministic
// HashEmbedder for offline runs, CI, and tests) and the helpers that turn graph nodes into embeddable text.
package embed

import (
	"context"
	"fmt"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// Embedder turns text into fixed-dimension float32 vectors. Implementations
// must return one vector per input string, in order, each of length Dim().
type Embedder interface {
	// Embed returns an embedding for each input text, in the same order.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Dim is the dimensionality of every vector this embedder returns.
	Dim() int
	// Model is a human-readable identifier (e.g. "nomic-embed-text" or "hash").
	Model() string
}

// Config selects and configures an embedding provider. It mirrors the
// embed.* keys in chiton's config file.
type Config struct {
	Provider string // "ollama" | "hash"
	Model    string
	Endpoint string
	Dim      int
}

// New returns the Embedder named by cfg.Provider. An empty provider defaults
// to "ollama". Unknown providers are an error.
func New(cfg Config) (Embedder, error) {
	switch strings.ToLower(strings.TrimSpace(cfg.Provider)) {
	case "", "ollama":
		return NewOllama(cfg), nil
	case "hash":
		dim := cfg.Dim
		if dim <= 0 {
			dim = DefaultHashDim
		}
		return NewHashEmbedder(dim), nil
	default:
		return nil, fmt.Errorf("embed: unknown provider %q (want ollama|hash)", cfg.Provider)
	}
}

// TextFor returns a node's embeddable text: a symbol's label and docstring, a doc section's label and text,
// or a concept's label and description. It returns "" for any other node kind.
func TextFor(n model.Node) string {
	switch n.Kind {
	case model.KindSymbol:
		return join(n.Label, model.AttrString(n.Attrs, "docstring"))
	case model.KindDocSection:
		if t := model.AttrString(n.Attrs, "text"); t != "" {
			return join(n.Label, t)
		}
		return n.Label
	case model.KindConcept:
		return join(n.Label, model.AttrString(n.Attrs, "description"))
	default:
		return ""
	}
}

// NodeEmbeddings embeds each node that TextFor gives text for in one Embed call and returns results as
// TextEmbeddings does.
func NodeEmbeddings(ctx context.Context, e Embedder, nodes []model.Node) (embs []store.Embedding, exact bool, err error) {
	var ids, texts []string
	for _, n := range nodes {
		if t := TextFor(n); t != "" {
			ids = append(ids, n.ID)
			texts = append(texts, t)
		}
	}
	return TextEmbeddings(ctx, e, ids, texts)
}

// TextEmbeddings embeds texts in one call, skipped when texts is empty, and pairs vectors with ids in order
// up to the shorter list. exact reports whether the embedder returned exactly one vector per text.
func TextEmbeddings(ctx context.Context, e Embedder, ids, texts []string) (embs []store.Embedding, exact bool, err error) {
	if len(texts) == 0 {
		return nil, true, nil
	}
	vecs, err := e.Embed(ctx, texts)
	if err != nil {
		return nil, false, err
	}
	embs = make([]store.Embedding, min(len(ids), len(vecs)))
	for i := range embs {
		embs[i] = store.Embedding{NodeID: ids[i], Vec: vecs[i]}
	}
	return embs, len(vecs) == len(texts), nil
}

// join joins the trimmed, non-empty parts with a newline.
func join(a, b string) string {
	a, b = strings.TrimSpace(a), strings.TrimSpace(b)
	switch {
	case a == "":
		return b
	case b == "":
		return a
	}
	return a + "\n" + b
}
