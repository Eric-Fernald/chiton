package embed

import (
	"context"
	"math"
	"strings"
	"unicode"
)

// DefaultHashDim is the dimensionality of the HashEmbedder when none is given; it only needs enough
// buckets to keep distinct tokens from colliding too often for tests and offline use.
const DefaultHashDim = 256

// HashEmbedder is a deterministic, dependency-free embedder for offline/CI runs (--provider hash): signed
// token hashing into Dim buckets, L2-normalized. It captures lexical overlap only; use Ollama for recall.
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
	// Tokens are lowercased runs of letters/digits.
	for tok := range strings.FieldsFuncSeq(strings.ToLower(text), isTokenSep) {
		bucket, sign := bucketAndSign(tok, h.dim)
		vec[bucket] += sign
	}
	normalize(vec)
	return vec
}

func isTokenSep(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }

// FNV-1a (32-bit) parameters, matching hash/fnv.New32a so existing hash embeddings stay valid.
const (
	fnvOffset32 = 2166136261
	fnvPrime32  = 16777619
)

// fnvSignSeed is the FNV-1a state after hashing the "sign:" salt.
var fnvSignSeed = fnv32a(fnvOffset32, "sign:")

// fnv32a continues an FNV-1a hash from state h over s, without allocating.
func fnv32a(h uint32, s string) uint32 {
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= fnvPrime32
	}
	return h
}

// bucketAndSign hashes tok to a bucket index and a ±1 sign. The sign comes
// from a separate salt so it is independent of the bucket choice.
func bucketAndSign(tok string, dim int) (int, float32) {
	bucket := int(fnv32a(fnvOffset32, tok)) % dim
	if bucket < 0 {
		bucket += dim
	}
	if fnv32a(fnvSignSeed, tok)&1 == 0 {
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
