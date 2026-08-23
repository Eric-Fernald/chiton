//go:build !msgraph

// The msgraph-free counterpart of outlook.go. The reasoning is the same as for
// internal/ingest/teams/graph_stub.go: the Microsoft Graph SDK costs 117 MB of
// binary for two API calls, so it is behind the `msgraph` build tag.
//
// `chiton email outlook` needs `go build -tags msgraph` (or `make
// build-msgraph`). `chiton email ingest` (local .eml/.mbox/maildir) and
// `chiton email gmail` are pure Go and always available.
package email

import (
	"context"
	"errors"
)

// errNoGraph is returned by every entry point that would need the Graph SDK.
var errNoGraph = errors.New("outlook: this chiton build was compiled without Microsoft Graph support; rebuild with 'go build -tags msgraph' (or 'make build-msgraph')")

// OutlookConfig mirrors the real configuration struct so callers compile
// identically in both build configurations.
type OutlookConfig struct {
	TenantID     string
	ClientID     string
	ClientSecret string
	// UserID is the mailbox owner's id or UPN: messages are read under
	// /users/{UserID}/messages (app-only has no /me).
	UserID string
}

// OutlookSource is the stub stand-in. The real type's fields are unexported, so
// an empty struct is API-identical from outside the package.
type OutlookSource struct{}

// NewOutlookSource always fails in a build without the msgraph tag.
func NewOutlookSource(OutlookConfig) (*OutlookSource, error) { return nil, errNoGraph }

// NewOutlookSourceFromEnv always fails in a build without the msgraph tag. The
// caller in cmd/chiton checks this error before opening the graph store for
// writing, so the failure costs nothing but a message.
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
