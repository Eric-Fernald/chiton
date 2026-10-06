// Package embed provides pluggable embedding providers (local Ollama by default, or a deterministic
// HashEmbedder for offline runs, CI, and tests) and the helpers that turn graph nodes into embeddable text.
package embed

import (
	"context"
	"fmt"
	"slices"
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
// or a concept's label and description. It returns "" for external package stubs and any other node kind.
func TextFor(n model.Node) string {
	switch n.Kind {
	case model.KindSymbol:
		if model.IsExternal(n) {
			return "" // a label-only stub for an import or cross-package call; graph expansion still reaches it
		}
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
		embs[i] = store.Embedding{NodeID: ids[i], Vec: vecs[i], Model: e.Model(), TextSHA: TextSHA(texts[i])}
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

// TextSHA is the hash stored with a vector to detect that its node's text changed since it was embedded.
func TextSHA(text string) string { return model.ContentSHA([]byte(text)) }

// Plan picks the nodes `chiton embed` must embed for modelID: those without a vector, with edited text, or with a
// vector from another or unknown model (all: every embeddable node). stale lists vectors whose node is gone or has no text.
func Plan(nodes []model.Node, have []store.EmbeddingMeta, modelID string, all bool) (work []model.Node, stale []string) {
	byID := make(map[string]store.EmbeddingMeta, len(have))
	for _, m := range have {
		byID[m.NodeID] = m
	}
	for _, n := range nodes {
		m, ok := byID[n.ID]
		delete(byID, n.ID)
		text := TextFor(n)
		switch {
		case text == "":
			if ok {
				stale = append(stale, n.ID)
			}
		case all || !ok || m.Model != modelID || m.TextSHA != TextSHA(text):
			work = append(work, n)
		}
	}
	for id := range byID {
		stale = append(stale, id)
	}
	slices.Sort(stale)
	return work, stale
}

// IndexReport says what KeepModel left out, and how many legacy vectors it kept on trust.
type IndexReport struct {
	Legacy   int            // kept: stored before models were recorded, so assumed to match
	Other    map[string]int // dropped: count per other model id
	WrongDim int            // dropped: dimension differs from the kept majority
}

// KeepModel filters embs in place to non-empty vectors made by modelID or of unknown model, then to their most
// common dimension, since cosine scores across models or dimensions are meaningless.
func KeepModel(embs []store.Embedding, modelID string) ([]store.Embedding, IndexReport) {
	rep := IndexReport{Other: map[string]int{}}
	kept := embs[:0]
	for _, e := range embs {
		switch {
		case len(e.Vec) == 0:
			continue
		case e.Model != "" && e.Model != modelID:
			rep.Other[e.Model]++
			continue
		}
		kept = append(kept, e)
	}
	dim := MajorityDim(kept)
	out := kept[:0]
	for _, e := range kept {
		if len(e.Vec) != dim {
			rep.WrongDim++
			continue
		}
		if e.Model == "" {
			rep.Legacy++
		}
		out = append(out, e)
	}
	clear(embs[len(out):]) // drop the excluded vectors so the GC can free them
	return out, rep
}

// MajorityDim returns the most common non-zero vector length in embs, the larger on a tie, so an index keeps the
// bulk of a store whose embedder changed and skips the stragglers.
func MajorityDim(embs []store.Embedding) int {
	counts := map[int]int{}
	best := 0
	for _, e := range embs {
		d := len(e.Vec)
		if d == 0 {
			continue
		}
		counts[d]++
		if c, b := counts[d], counts[best]; c > b || (c == b && d > best) {
			best = d
		}
	}
	return best
}
