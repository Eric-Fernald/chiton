//go:build !msgraph

// The msgraph-free counterpart of outlook.go: the Graph SDK adds ~117 MB to the binary, so `chiton email
// outlook` needs `-tags msgraph`; local and Gmail ingest are always available.

package email

import (
	"context"
	"errors"
)

// errNoGraph is returned by every entry point that would need the Graph SDK.
var errNoGraph = errors.New("outlook: this chiton build was compiled without Microsoft Graph support; rebuild with 'go build -tags msgraph' (or 'make build-msgraph')")

// OutlookSource is the stub stand-in. The real type's fields are unexported, so
// an empty struct is API-identical from outside the package.
type OutlookSource struct{}

// NewOutlookSource always fails in a build without the msgraph tag.
func NewOutlookSource(OutlookConfig) (*OutlookSource, error) { return nil, errNoGraph }

// NewOutlookSourceFromEnv always fails in a build without the msgraph tag; cmd/chiton checks this before
// opening the store for writing.
func NewOutlookSourceFromEnv() (*OutlookSource, error) { return nil, errNoGraph }

// Name satisfies Source, and keeps reporting the same source kind so any
// persisted "outlook" data still round-trips through a default build.
func (s *OutlookSource) Name() string { return "outlook" }

// Messages satisfies Source.
func (s *OutlookSource) Messages(context.Context, Query) ([]Message, error) {
	return nil, errNoGraph
}

// Keep the stub honest: it must implement the same seam as the real source.
var _ Source = (*OutlookSource)(nil)
