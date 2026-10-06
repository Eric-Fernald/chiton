package vector

import (
	"fmt"
	"strings"
)

// DefaultHNSWThreshold is the vector count at which the auto backend switches from Flat to HNSW. Below
// it an exact scan is still only a few milliseconds per query and needs no build time.
const DefaultHNSWThreshold = 100_000

// Options picks and tunes the backend NewIndex returns.
type Options struct {
	Backend   string // auto | flat | hnsw; empty means auto
	Threshold int    // auto uses HNSW at or above this many vectors; 0 means DefaultHNSWThreshold
	HNSW      HNSWConfig
}

// NewIndex returns an empty index of the backend opts selects for n vectors.
func NewIndex(opts Options, n int) (Index, error) {
	switch strings.ToLower(strings.TrimSpace(opts.Backend)) {
	case "", "auto":
		if n < cmpOr(opts.Threshold, DefaultHNSWThreshold) {
			return NewFlatSized(n), nil
		}
		return NewHNSW(opts.HNSW), nil
	case "flat":
		return NewFlatSized(n), nil
	case "hnsw":
		return NewHNSW(opts.HNSW), nil
	}
	return nil, fmt.Errorf("vector: unknown index backend %q (want auto, flat, or hnsw)", opts.Backend)
}

// BackendName names idx's implementation for logs: "flat", "hnsw", or its Go type.
func BackendName(idx Index) string {
	switch idx.(type) {
	case *Flat:
		return "flat"
	case *HNSW:
		return "hnsw"
	}
	return fmt.Sprintf("%T", idx)
}
