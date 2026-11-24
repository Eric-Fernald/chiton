package main

import (
	"errors"
	"testing"
)

// TestCloseGraphStoreReportsCloseFailure checks a failed close fails a command that otherwise succeeded (a
// writable store flushes on close), and never hides the command's own error.
func TestCloseGraphStoreReportsCloseFailure(t *testing.T) {
	boom := errors.New("disk full")
	failing := func() error { return boom }

	var err error
	closeGraphStore(failing, &err)
	if !errors.Is(err, boom) {
		t.Errorf("err = %v, want the close failure", err)
	}

	first := errors.New("build failed")
	err = first
	closeGraphStore(failing, &err)
	if !errors.Is(err, first) || errors.Is(err, boom) {
		t.Errorf("err = %v, want the command's own error kept", err)
	}

	err = nil
	closeGraphStore(func() error { return nil }, &err)
	if err != nil {
		t.Errorf("err = %v after a clean close", err)
	}
}
