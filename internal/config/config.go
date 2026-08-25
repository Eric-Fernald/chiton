package config

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/viper"
)

// ProjectDirEnv names the environment variable that anchors chiton to a
// workspace regardless of the process working directory.
//
// Assistant-launched MCP servers do not reliably start in the workspace root —
// the Claude Code plugin sets this to the project directory — so without an
// anchor a server can end up reading a different store than `chiton build`
// wrote. When it is set, `.chitonrc` is looked up there and a relative
// graph_path is resolved against it.
const ProjectDirEnv = "CHITON_PROJECT_DIR"

// ProjectDir returns the anchoring directory: $CHITON_PROJECT_DIR when set,
// otherwise "." (the process working directory), preserving the historical
// behaviour for CLI use.
//
// A value still carrying a "${...}" template is treated as unset: the launcher
// passed the placeholder through without substituting it (Claude Code does this
// with ${CLAUDE_PROJECT_DIR} when the variable is undefined), and anchoring to a
// literal "${CLAUDE_PROJECT_DIR}" directory only turns a working store path into
// one that cannot be opened.
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

type Config struct {
	GraphPath string         `mapstructure:"graph_path"`
	LogLevel  string         `mapstructure:"log_level"`
	Embed     EmbedConfig    `mapstructure:"embed"`
	Semantic  SemanticConfig `mapstructure:"semantic"`
}

// Load reads .chitonrc (TOML) from the project directory, merges environment
// overrides (CHITON_* prefix), and returns a Config with sane defaults.
//
// The project directory is $CHITON_PROJECT_DIR when set, else the process
// working directory. See LoadFrom.
func Load() (*Config, error) {
	return LoadFrom(ProjectDir())
}

// LoadFrom is Load anchored at an explicit directory. A relative graph_path is
// resolved against dir only when dir is not the working directory, so plain CLI
// invocations keep returning the relative default they always have.
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
	// No embed.dim default on purpose: leaving it 0 lets each provider apply its
	// own native dimension (Ollama 768, hash 256) instead of forcing the Ollama
	// size onto the hash provider. An explicit embed.dim / CHITON_EMBED_DIM wins.
	//
	// It still has to be bound explicitly, though. Unmarshal only walks
	// AllKeys(), and AutomaticEnv alone does not put a key there — so without
	// this BindEnv, embed.dim is the one setting where the environment override
	// is silently dropped while every other CHITON_* var works.
	if err := v.BindEnv("embed.dim"); err != nil {
		return nil, err
	}
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
	cfg.GraphPath = resolveAgainst(dir, cfg.GraphPath)
	return &cfg, nil
}

// resolveAgainst makes a relative path absolute against dir. It is a no-op for
// absolute paths, for an empty path, and for dir == "." — the last so that CLI
// runs keep printing the familiar relative "./chiton-out/graph.kuzu".
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
