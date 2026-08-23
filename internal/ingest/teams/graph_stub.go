//go:build !msgraph

// The msgraph-free counterpart of graph_client.go.
//
// Microsoft Graph's Go SDK is reached through msgraphsdk.GraphServiceClient,
// whose type transitively references request builders for every top-level Graph
// resource — admin, devicemanagement, education, planner, print, security and a
// hundred more. chiton calls exactly two of those paths, but the linker cannot
// strip the rest: kiota registers its serialization types through interface
// tables, so the whole surface lands in the binary's function and type metadata.
// Measured cost: 117 MB, 83% of the stripped binary.
//
// So it is excluded by default. `chiton teams fetch` and `chiton teams
// serve-webhook` need a binary built with `go build -tags msgraph` (or
// `make build-msgraph`); everything else in package teams — the transcript
// parser, the ingester, the webhook receiver — is pure Go and always available.
package teams

import (
	"context"
	"errors"
)

// errNoGraph is returned by every entry point that would need the Graph SDK.
var errNoGraph = errors.New("teams: this chiton build was compiled without Microsoft Graph support; rebuild with 'go build -tags msgraph' (or 'make build-msgraph')")

// GraphConfig mirrors the real configuration struct so callers compile
// identically in both build configurations.
type GraphConfig struct {
	TenantID     string
	ClientID     string
	ClientSecret string
	// UserID is the meeting organizer's id or UPN: transcripts are read under
	// /users/{UserID}/onlineMeetings/... (app-only has no /me).
	UserID string
}

// GraphSource is the stub stand-in. The real type's fields are unexported, so
// an empty struct is API-identical from outside the package.
type GraphSource struct{}

// NewGraphSource always fails in a build without the msgraph tag.
func NewGraphSource(GraphConfig) (*GraphSource, error) { return nil, errNoGraph }

// NewGraphSourceFromEnv always fails in a build without the msgraph tag. Both
// callers in cmd/chiton check this error before opening the graph store, so the
// failure costs nothing but a message.
func NewGraphSourceFromEnv() (*GraphSource, error) { return nil, errNoGraph }

// ListTranscripts satisfies TranscriptSource.
func (g *GraphSource) ListTranscripts(context.Context, string) ([]TranscriptMeta, error) {
	return nil, errNoGraph
}

// FetchContent satisfies TranscriptSource.
func (g *GraphSource) FetchContent(context.Context, string, string) ([]byte, error) {
	return nil, errNoGraph
}

// Keep the stub honest: it must implement the same seam as the real client.
var _ TranscriptSource = (*GraphSource)(nil)
