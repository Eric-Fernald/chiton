package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/config"
)

func TestNewEmbedderRefusesRepoChosenRemoteEndpoint(t *testing.T) {
	cfg := &config.Config{Embed: config.EmbedConfig{
		Provider:       "ollama",
		Endpoint:       "https://embed.attacker.example",
		EndpointOrigin: config.OriginProjectRC,
	}}
	if _, err := newEmbedder(cfg, "", ""); err == nil {
		t.Fatal("newEmbedder accepted a remote endpoint chosen by the project .chitonrc")
	}
	// The hash provider never contacts the endpoint, so it stays available.
	if _, err := newEmbedder(cfg, "hash", ""); err != nil {
		t.Errorf("newEmbedder with provider hash: %v", err)
	}
}

// A refused embed endpoint must disable embeddings only; MCP extraction needs no embedder.
func TestBuildSemanticDepsKeepsExtractorWhenEndpointRefused(t *testing.T) {
	t.Setenv(plaintextOKEnv, "1")
	cfg := &config.Config{
		GraphPath: filepath.Join(t.TempDir(), "graph.badger"),
		Embed: config.EmbedConfig{
			Provider:       "ollama",
			Endpoint:       "https://embed.attacker.example",
			EndpointOrigin: config.OriginProjectRC,
		},
	}
	gs, err := openGraphStore(cfg.GraphPath, false)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	cmd := &cobra.Command{}
	var stderr bytes.Buffer
	cmd.SetErr(&stderr)
	deps := buildSemanticDeps(cmd, cfg, gs, nil)
	if deps.Extractor == nil {
		t.Error("Extractor is nil: a refused embed endpoint disabled MCP extraction")
	}
	if deps.Retriever != nil {
		t.Error("Retriever set despite the refused endpoint")
	}
	if !strings.Contains(stderr.String(), "CHITON_EMBED_ENDPOINT") {
		t.Errorf("stderr = %q, want a warning naming CHITON_EMBED_ENDPOINT", stderr.String())
	}
}
