package config

import (
	"strings"

	"github.com/spf13/viper"
)

// EmbedConfig configures the embedding provider used by `chiton embed`,
// `chiton search`, and the semantic MCP tools.
type EmbedConfig struct {
	Provider string `mapstructure:"provider"` // ollama | hash
	Model    string `mapstructure:"model"`
	Endpoint string `mapstructure:"endpoint"`
	Dim      int    `mapstructure:"dim"`
}

// SemanticConfig configures subagent-mode semantic extraction.
type SemanticConfig struct {
	// ConfidenceThreshold splits INFERRED from AMBIGUOUS for LLM-extracted
	// concepts and edges.
	ConfidenceThreshold float32 `mapstructure:"confidence_threshold"`
}

type Config struct {
	GraphPath string         `mapstructure:"graph_path"`
	LogLevel  string         `mapstructure:"log_level"`
	Embed     EmbedConfig    `mapstructure:"embed"`
	Semantic  SemanticConfig `mapstructure:"semantic"`
}

// Load reads .chitonrc (TOML) from the current directory, merges environment
// overrides (CHITON_* prefix), and returns a Config with sane defaults.
func Load() (*Config, error) {
	v := viper.New()
	v.SetConfigName(".chitonrc")
	v.SetConfigType("toml")
	v.AddConfigPath(".")
	v.AddConfigPath("$HOME")

	v.SetDefault("graph_path", "./chiton-out/graph.kuzu")
	v.SetDefault("log_level", "info")

	// Semantic-layer defaults (Phase 2). Embeddings default to local Ollama;
	// nothing leaves the machine unless the user points us at a remote provider.
	v.SetDefault("embed.provider", "ollama")
	v.SetDefault("embed.model", "nomic-embed-text")
	v.SetDefault("embed.endpoint", "http://localhost:11434")
	// No embed.dim default on purpose: leaving it 0 lets each provider apply its
	// own native dimension (Ollama 768, hash 256) instead of forcing the Ollama
	// size onto the hash provider. An explicit embed.dim / CHITON_EMBED_DIM wins.
	v.SetDefault("semantic.confidence_threshold", 0.5)

	// Allow CHITON_EMBED_PROVIDER, CHITON_SEMANTIC_CONFIDENCE_THRESHOLD, etc.
	// The key replacer maps nested keys (embed.provider) to flat env vars
	// (CHITON_EMBED_PROVIDER).
	v.SetEnvPrefix("CHITON")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	// Config file is optional.
	_ = v.ReadInConfig()

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}
