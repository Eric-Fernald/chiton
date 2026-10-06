package config

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/viper"

	"github.com/Eric-Fernald/chiton/internal/vector"
)

// ProjectDirEnv anchors chiton to a workspace, since assistant-launched MCP servers may start elsewhere
// and read the wrong store. When set, .chitonrc is read there and a relative graph_path resolves against it.
const ProjectDirEnv = "CHITON_PROJECT_DIR"

// ProjectDir returns $CHITON_PROJECT_DIR, or "." when unset. A value still holding a "${...}" template
// (a launcher placeholder left unsubstituted) counts as unset, since anchoring to it breaks the store path.
func ProjectDir() string {
	if dir := strings.TrimSpace(os.Getenv(ProjectDirEnv)); dir != "" && !strings.Contains(dir, "${") {
		return dir
	}
	return "."
}

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

// VectorConfig selects and tunes the nearest-neighbour index behind semantic search.
type VectorConfig struct {
	Index              string `mapstructure:"index"`          // auto | flat | hnsw
	HNSWThreshold      int    `mapstructure:"hnsw_threshold"` // auto switches to HNSW at this many vectors
	HNSWM              int    `mapstructure:"hnsw_m"`
	HNSWEfSearch       int    `mapstructure:"hnsw_ef_search"`
	HNSWEfConstruction int    `mapstructure:"hnsw_ef_construction"`
}

type Config struct {
	GraphPath string         `mapstructure:"graph_path"`
	LogLevel  string         `mapstructure:"log_level"`
	Embed     EmbedConfig    `mapstructure:"embed"`
	Semantic  SemanticConfig `mapstructure:"semantic"`
	Vector    VectorConfig   `mapstructure:"vector"`
}

// Load reads .chitonrc (TOML) from ProjectDir, merges CHITON_* environment overrides, and returns a
// Config with sane defaults. See LoadFrom.
func Load() (*Config, error) {
	return LoadFrom(ProjectDir())
}

// LoadFrom is Load anchored at dir. It resolves a relative graph_path against dir only when dir is not
// ".", so plain CLI invocations keep their relative default.
func LoadFrom(dir string) (*Config, error) {
	if dir == "" {
		dir = "."
	}

	v := viper.New()
	v.SetConfigName(".chitonrc")
	v.SetConfigType("toml")
	v.AddConfigPath(dir)
	v.AddConfigPath("$HOME")

	v.SetDefault("graph_path", "./chiton-out/graph.kuzu")
	v.SetDefault("log_level", "info")

	// Semantic-layer defaults (Phase 2). Embeddings default to local Ollama;
	// nothing leaves the machine unless the user points us at a remote provider.
	v.SetDefault("embed.provider", "ollama")
	v.SetDefault("embed.model", "nomic-embed-text")
	v.SetDefault("embed.endpoint", "http://localhost:11434")
	// embed.dim gets no default, so 0 lets each provider use its native dimension (Ollama 768, hash 256).
	// It must still be bound, as Unmarshal only walks AllKeys() and would silently drop CHITON_EMBED_DIM.
	if err := v.BindEnv("embed.dim"); err != nil {
		return nil, err
	}
	v.SetDefault("semantic.confidence_threshold", 0.5)
	v.SetDefault("vector.index", "auto")
	v.SetDefault("vector.hnsw_threshold", vector.DefaultHNSWThreshold)
	v.SetDefault("vector.hnsw_m", vector.DefaultHNSWM)
	v.SetDefault("vector.hnsw_ef_search", vector.DefaultHNSWEfSearch)
	v.SetDefault("vector.hnsw_ef_construction", vector.DefaultHNSWEfConstruction)

	// The replacer maps nested keys to flat env vars, e.g. embed.provider to CHITON_EMBED_PROVIDER.
	v.SetEnvPrefix("CHITON")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	// Config file is optional.
	_ = v.ReadInConfig()

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, err
	}
	cfg.GraphPath = resolveAgainst(dir, cfg.GraphPath)
	return &cfg, nil
}

// resolveAgainst makes a relative path absolute against dir. It is a no-op for absolute or empty paths
// and for dir ".", so CLI runs keep printing the familiar relative "./chiton-out/graph.kuzu".
func resolveAgainst(dir, path string) string {
	if path == "" || dir == "" || dir == "." || filepath.IsAbs(path) {
		return path
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return path
	}
	return filepath.Join(abs, path)
}
