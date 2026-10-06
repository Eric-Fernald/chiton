package hnswlib

import (
	"math"
	"testing"
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

func TestMetricString(t *testing.T) {
	if Cosine.String() != "cosine" || UnitCosine.String() != "unit-cosine" || Metric(0).String() != "Metric(0)" {
		t.Fatal("unexpected Metric names")
	}
}
