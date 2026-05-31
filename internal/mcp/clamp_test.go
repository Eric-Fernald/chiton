package mcpsrv

import "testing"

func TestClampInt(t *testing.T) {
	cases := []struct {
		v, lo, hi, want int
	}{
		{2, 1, 5, 2},        // in range
		{0, 1, 5, 1},        // below floor
		{100000, 1, 5, 5},   // above ceiling (depth=100000 → 5)
		{1, 1, 5, 1},        // at floor
		{5, 1, 5, 5},        // at ceiling
		{0, 0, 3, 0},        // hops floor of 0
		{4000, 500, 16000, 4000},
	}
	for _, c := range cases {
		if got := clampInt(c.v, c.lo, c.hi); got != c.want {
			t.Errorf("clampInt(%d, %d, %d) = %d, want %d", c.v, c.lo, c.hi, got, c.want)
		}
	}
}
