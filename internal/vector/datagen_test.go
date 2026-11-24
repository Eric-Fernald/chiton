package vector

import (
	"fmt"
	"math/rand"
)

// embeddingLike draws n vectors shaped like text embeddings rather than iid noise: points from a mixture
// of clusters in a 48-dimensional latent space, linearly projected to dim and lightly perturbed. iid
// Gaussian vectors in hundreds of dimensions have no neighbourhood structure (every point is nearly
// equidistant), so they measure the worst case for any approximate index rather than chiton's data.
type embeddingLike struct {
	proj    [][]float32 // dim x latent
	centers [][]float64
	rng     *rand.Rand
}

const latentDim = 48

func newEmbeddingLike(rng *rand.Rand, dim, clusters int) *embeddingLike {
	e := &embeddingLike{rng: rng, proj: make([][]float32, dim), centers: make([][]float64, clusters)}
	for i := range e.proj {
		e.proj[i] = make([]float32, latentDim)
		for j := range e.proj[i] {
			e.proj[i][j] = float32(rng.NormFloat64())
		}
	}
	for i := range e.centers {
		e.centers[i] = make([]float64, latentDim)
		for j := range e.centers[i] {
			e.centers[i][j] = rng.NormFloat64() * 2
		}
	}
	return e
}

func (e *embeddingLike) vec() []float32 {
	c := e.centers[e.rng.Intn(len(e.centers))]
	z := make([]float64, latentDim)
	for j := range z {
		z[j] = c[j] + e.rng.NormFloat64()
	}
	v := make([]float32, len(e.proj))
	for i, row := range e.proj {
		var s float64
		for j, a := range row {
			s += float64(a) * z[j]
		}
		v[i] = float32(s) + float32(e.rng.NormFloat64())*4
	}
	return v
}

func (e *embeddingLike) items(n int) []Item {
	items := make([]Item, n)
	for i := range items {
		items[i] = Item{ID: fmt.Sprintf("n%07d", i), Vec: e.vec()}
	}
	return items
}

func (e *embeddingLike) queries(n int) [][]float32 {
	qs := make([][]float32, n)
	for i := range qs {
		qs[i] = e.vec()
	}
	return qs
}
