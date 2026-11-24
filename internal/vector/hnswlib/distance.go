package hnswlib

import (
	"fmt"

	"github.com/viterin/vek/vek32"
)

// Metric selects the distance a graph is built on; smaller is closer. Its value is stored by WriteTo, so
// existing values must never be renumbered.
type Metric uint8

const (
	// Cosine is 1 minus cosine similarity.
	Cosine Metric = iota + 1
	// Euclidean is L2 distance.
	Euclidean
	// UnitCosine is 1 minus the dot product: cosine distance for vectors already scaled to unit length,
	// without recomputing their norms on every comparison.
	UnitCosine
)

// Distance compares a and b, which must have equal length. vek32 uses AVX2 when the CPU has it.
func (m Metric) Distance(a, b []float32) float32 {
	switch m {
	case Euclidean:
		return vek32.Distance(a, b)
	case UnitCosine:
		return 1 - vek32.Dot(a, b)
	default:
		return 1 - vek32.CosineSimilarity(a, b)
	}
}

func (m Metric) valid() bool { return m >= Cosine && m <= UnitCosine }

func (m Metric) String() string {
	switch m {
	case Cosine:
		return "cosine"
	case Euclidean:
		return "euclidean"
	case UnitCosine:
		return "unit-cosine"
	}
	return fmt.Sprintf("Metric(%d)", uint8(m))
}
