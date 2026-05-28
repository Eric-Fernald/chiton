package embed

import (
	"context"
	"hash/fnv"
	"math"
	"strings"
	"unicode"
)

// DefaultHashDim is the dimensionality of the HashEmbedder when none is given.
// Smaller than a real model's 768 — it only needs enough buckets to keep
// distinct tokens from colliding too often for tests and offline use.
const DefaultHashDim = 256

// HashEmbedder is a deterministic, dependency-free embedder. It tokenises text
// on non-alphanumeric boundaries, hashes each token into one of Dim buckets
// (with a signed contribution so unrelated tokens can cancel rather than only
// add), and L2-normalises the result. It captures lexical overlap — enough to
// drive and test the retrieval pipeline — but is not a semantic model.
// Use it for offline/CI runs (--provider hash); use Ollama for real recall.
type HashEmbedder struct {
	dim int
}

// NewHashEmbedder returns a HashEmbedder of the given dimension (defaulting to
// DefaultHashDim when dim <= 0).
func NewHashEmbedder(dim int) *HashEmbedder {
	if dim <= 0 {
		dim = DefaultHashDim
	}
	return &HashEmbedder{dim: dim}
}

func (h *HashEmbedder) Dim() int      { return h.dim }
func (h *HashEmbedder) Model() string { return "hash" }

// Embed returns one normalized vector per input text.
func (h *HashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = h.embedOne(t)
	}
	return out, nil
}

func (h *HashEmbedder) embedOne(text string) []float32 {
	vec := make([]float32, h.dim)
	for _, tok := range tokenize(text) {
		bucket, sign := bucketAndSign(tok, h.dim)
		vec[bucket] += sign
	}
	normalize(vec)
	return vec
}

// tokenize lowercases and splits on any non-letter/non-digit rune.
func tokenize(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

// bucketAndSign hashes tok to a bucket index and a ±1 sign. The sign comes
// from a separate salt so it is independent of the bucket choice.
func bucketAndSign(tok string, dim int) (int, float32) {
	hb := fnv.New32a()
	_, _ = hb.Write([]byte(tok))
	bucket := int(hb.Sum32()) % dim
	if bucket < 0 {
		bucket += dim
	}

	hs := fnv.New32a()
	_, _ = hs.Write([]byte("sign:" + tok))
	if hs.Sum32()&1 == 0 {
		return bucket, 1
	}
	return bucket, -1
}

// normalize scales vec to unit L2 length in place. A zero vector is left as-is.
func normalize(vec []float32) {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range vec {
		vec[i] *= inv
	}
}
