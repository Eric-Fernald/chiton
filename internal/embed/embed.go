// Package embed provides a pluggable embedding-provider interface and the
// helpers chiton uses to turn graph nodes into embeddable text.
//
// The default provider is Ollama (local, no API key). A deterministic
// HashEmbedder is provided for offline/CI use and for tests, so the whole
// semantic layer is exercisable without a running model.
package embed

import (
	"context"
	"fmt"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
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

// TextFor returns the text chiton embeds for a node, or "" if the node kind
// is not embedded. We embed symbols (label + docstring), doc sections (their
// text), and concepts (label + description).
func TextFor(n model.Node) string {
	switch n.Kind {
	case model.KindSymbol:
		return join(n.Label, attrString(n, "docstring"))
	case model.KindDocSection:
		if t := attrString(n, "text"); t != "" {
			return join(n.Label, t)
		}
		return n.Label
	case model.KindConcept:
		return join(n.Label, attrString(n, "description"))
	default:
		return ""
	}
}

// Embeddable reports whether TextFor would return a non-empty string.
func Embeddable(n model.Node) bool { return TextFor(n) != "" }

func attrString(n model.Node, key string) string {
	if n.Attrs == nil {
		return ""
	}
	if v, ok := n.Attrs[key].(string); ok {
		return v
	}
	return ""
}

func join(parts ...string) string {
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if s := strings.TrimSpace(p); s != "" {
			out = append(out, s)
		}
	}
	return strings.Join(out, "\n")
}
