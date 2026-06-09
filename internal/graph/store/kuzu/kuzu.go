// Package kuzu will hold the Kuzu embedded graph store implementation.
// Kuzu requires CGo and a C compiler; on platforms where those are unavailable
// use the badger store instead.
package kuzu

import (
	"context"
	"errors"
	"io"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// ErrUnavailable is returned by all methods when Kuzu is not compiled in.
var ErrUnavailable = errors.New("kuzu: CGo not available on this platform; use the badger store instead")

// Store is a placeholder that satisfies store.GraphStore.
// Replace this file with the real Kuzu implementation once CGo is available.
type Store struct{}

func Open(_ string) (*Store, error)                                         { return nil, ErrUnavailable }
func (s *Store) Close() error                                               { return ErrUnavailable }
func (s *Store) UpsertNodes(_ context.Context, _ []model.Node) error        { return ErrUnavailable }
func (s *Store) UpsertEdges(_ context.Context, _ []model.Edge) error        { return ErrUnavailable }
func (s *Store) GetNode(_ context.Context, _ string) (*model.Node, error)   { return nil, ErrUnavailable }
func (s *Store) Snapshot(_ context.Context, _ io.Writer) error              { return ErrUnavailable }
func (s *Store) DeleteBySourceURI(_ context.Context, _ string) (int, error) { return 0, ErrUnavailable }
func (s *Store) SearchNodes(_ context.Context, _ string, _ int) ([]model.Node, error) {
	return nil, ErrUnavailable
}

func (s *Store) Neighbors(_ context.Context, _ string, _ int, _ []model.EdgeKind) (store.Subgraph, error) {
	return store.Subgraph{}, ErrUnavailable
}

func (s *Store) ShortestPath(_ context.Context, _, _ string) (store.Path, error) {
	return store.Path{}, ErrUnavailable
}
