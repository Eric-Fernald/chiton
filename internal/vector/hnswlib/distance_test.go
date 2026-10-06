package hnswlib

import (
	"math"
	"testing"

	"github.com/viterin/vek/vek32"
)

func TestMetricDistance(t *testing.T) {
	s := float32(math.Sqrt(0.5))
	for _, tc := range []struct {
		m    Metric
		a, b []float32
		want float64
	}{
		{Euclidean, []float32{1, 2, 3}, []float32{4, 5, 6}, math.Sqrt(27)},
		{Cosine, []float32{1, 1, 1}, []float32{0.8, 0.8, 0.8}, 0}, // same direction, different length
		{Cosine, []float32{1, 0}, []float32{0, 1}, 1},
		{Cosine, []float32{1, 0}, []float32{-2, 0}, 2},
		{UnitCosine, []float32{1, 0}, []float32{s, s}, 1 - float64(s)},
		{UnitCosine, []float32{0, 1}, []float32{0, 1}, 0},
	} {
		if d := tc.m.Distance(tc.a, tc.b); math.Abs(float64(d)-tc.want) > 1e-6 {
			t.Errorf("%v.Distance(%v, %v) = %v, want %v", tc.m, tc.a, tc.b, d, tc.want)
		}
	}
}

// TestMetricDistanceWithoutSIMD runs the pure-Go kernels that non-amd64 targets (arm64 releases) use, and
// checks they agree with the AVX2 ones on this machine.
func TestMetricDistanceWithoutSIMD(t *testing.T) {
	a, b := make([]float32, 768), make([]float32, 768)
	for i := range a {
		a[i], b[i] = float32(i%7)-3, float32(i%5)-2
	}
	var simd [3]float32
	for i, m := range []Metric{Cosine, Euclidean, UnitCosine} {
		simd[i] = m.Distance(a, b)
	}
	vek32.SetAcceleration(false)
	defer vek32.SetAcceleration(true)
	for i, m := range []Metric{Cosine, Euclidean, UnitCosine} {
		if d := m.Distance(a, b); math.Abs(float64(d-simd[i])) > 1e-3*math.Max(1, math.Abs(float64(d))) {
			t.Errorf("%v: pure Go %v, SIMD %v", m, d, simd[i])
		}
	}
}

func TestMetricString(t *testing.T) {
	if Cosine.String() != "cosine" || UnitCosine.String() != "unit-cosine" || Metric(0).String() != "Metric(0)" {
		t.Fatal("unexpected Metric names")
	}
}
