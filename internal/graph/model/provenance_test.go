package model

import "testing"

func TestResolveProvenance(t *testing.T) {
	cases := []struct {
		conf     float32
		wantProv Provenance
		wantConf float32
	}{
		{0.9, ProvInferred, 0.9},
		{0.5, ProvInferred, 0.5},    // at threshold → INFERRED
		{0.49, ProvAmbiguous, 0.49}, // just below threshold → AMBIGUOUS
		{-1, ProvAmbiguous, 0},      // clamped to 0
		{2, ProvInferred, 1},        // clamped to 1
	}
	for _, c := range cases {
		p, conf := ResolveProvenance(c.conf, DefaultConfidenceThreshold)
		if p != c.wantProv || conf != c.wantConf {
			t.Fatalf("ResolveProvenance(%v) = (%s,%v), want (%s,%v)", c.conf, p, conf, c.wantProv, c.wantConf)
		}
	}
}
