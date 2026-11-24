package badger

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/crypto"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// TestEncryptedRoundTrip verifies that data written through an encrypted store
// is readable when reopened with the same key and unreadable with no key.
func TestEncryptedRoundTrip(t *testing.T) {
	dir := t.TempDir()
	graph := filepath.Join(dir, "graph")
	kf, err := crypto.DefaultKeyFile()
	if err != nil {
		t.Fatal(err)
	}
	key, err := kf.DeriveKey("test-passphrase")
	if err != nil {
		t.Fatal(err)
	}

	s, err := OpenWith(graph, Options{EncryptionKey: key})
	if err != nil {
		t.Fatalf("OpenWith: %v", err)
	}
	now := time.Now()
	node := model.Node{
		ID:        "abcd1234",
		Kind:      model.KindSymbol,
		Label:     "secret-symbol",
		SourceURI: "file:///secret.go",
		CreatedAt: now,
		UpdatedAt: now,
	}
	if err := s.UpsertNodes(context.Background(), []model.Node{node}); err != nil {
		t.Fatalf("UpsertNodes: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Re-open with the correct key — should round-trip.
	s2, err := OpenWith(graph, Options{EncryptionKey: key})
	if err != nil {
		t.Fatalf("re-open with key: %v", err)
	}
	got, err := s2.GetNode(context.Background(), node.ID)
	if err != nil {
		t.Fatalf("GetNode: %v", err)
	}
	if got == nil || got.Label != "secret-symbol" {
		t.Errorf("expected round-trip node, got %+v", got)
	}
	s2.Close()

	// Open with NO key — Badger must reject.
	if _, err := OpenWith(graph, Options{}); err == nil {
		t.Errorf("expected error opening encrypted store with no key")
	}
}

// TestPlaintextStill verifies the no-key path is unchanged for users who
// never opt in to encryption.
func TestPlaintextStill(t *testing.T) {
	dir := t.TempDir()
	graph := filepath.Join(dir, "graph")
	s, err := Open(graph)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	now := time.Now()
	if err := s.UpsertNodes(context.Background(), []model.Node{{ID: "x", Kind: model.KindFile, Label: "a", SourceURI: "file://a", CreatedAt: now, UpdatedAt: now}}); err != nil {
		t.Fatalf("UpsertNodes: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}
