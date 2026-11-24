//go:build !windows

package model

// canonicalPath returns abs unchanged; only Windows paths get case normalisation.
func canonicalPath(abs string) string { return abs }
