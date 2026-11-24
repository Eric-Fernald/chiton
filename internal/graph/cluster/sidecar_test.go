package cluster

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	chitoncrypto "github.com/Eric-Fernald/chiton/internal/crypto"
)

func sampleResults() *Results {
	return &Results{NodeCount: 1, CommunityCount: 1, Nodes: map[string]NodeResult{
		"n1": {Label: "Re: acquisition of Initech", SourceURI: "email://msg-1", Degree: 1},
	}}
}

// An encrypted store's sidecar must not hold its labels and source URIs in plaintext.
func TestSidecarSealedWithKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cluster.json")
	key := bytes.Repeat([]byte{7}, 32)
	if err := WriteSidecar(path, sampleResults(), key); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte("Initech")) || bytes.Contains(raw, []byte("email://")) {
		t.Error("sealed sidecar leaks labels or source URIs")
	}
	if _, err := LoadSidecar(path, nil); !errors.Is(err, chitoncrypto.ErrSealedNoKey) {
		t.Errorf("LoadSidecar without a key err = %v, want ErrSealedNoKey", err)
	}
	got, err := LoadSidecar(path, key)
	if err != nil {
		t.Fatal(err)
	}
	if got.Nodes["n1"].Label != "Re: acquisition of Initech" {
		t.Errorf("round-trip label = %q", got.Nodes["n1"].Label)
	}
}

// A plaintext sidecar written before encryption stays readable with a key; its next write seals it.
func TestSidecarLegacyPlaintextReadWithKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cluster.json")
	if err := WriteSidecar(path, sampleResults(), nil); err != nil {
		t.Fatal(err)
	}
	got, err := LoadSidecar(path, bytes.Repeat([]byte{7}, 32))
	if err != nil || got.NodeCount != 1 {
		t.Fatalf("LoadSidecar(plaintext, key) = %+v, %v", got, err)
	}
}

// A cluster.json from before the compact, sealable format (indented JSON, never sealed) still loads, with or
// without a key.
func TestSidecarLoadsIndentedLegacyFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cluster.json")
	b, err := json.MarshalIndent(sampleResults(), "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, key := range [][]byte{nil, bytes.Repeat([]byte{7}, 32)} {
		got, err := LoadSidecar(path, key)
		if err != nil || got.Nodes["n1"].SourceURI != "email://msg-1" {
			t.Errorf("LoadSidecar(indented legacy, key=%v) = %+v, %v", key != nil, got, err)
		}
	}
}
