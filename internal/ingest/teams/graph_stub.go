//go:build !msgraph

// This is the msgraph-free counterpart of graph_client.go. Kiota registers types via interface tables, so
// Graph SDK request builders can't be dead-stripped and add ~117 MB. `chiton teams` needs -tags msgraph.

package teams

import (
	"context"
	"errors"
)

// errNoGraph is returned by every entry point that would need the Graph SDK.
var errNoGraph = errors.New("teams: this chiton build was compiled without Microsoft Graph support; rebuild with 'go build -tags msgraph' (or 'make build-msgraph')")

// GraphSource is the stub stand-in. The real type's fields are unexported, so
// an empty struct is API-identical from outside the package.
type GraphSource struct{}

// NewGraphSource always fails in a build without the msgraph tag.
func NewGraphSource(GraphConfig) (*GraphSource, error) { return nil, errNoGraph }

// NewGraphSourceFromEnv always fails in a build without the msgraph tag. Both cmd/chiton callers check this
// before opening the graph store.
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
