package main

import (
	"testing"

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
