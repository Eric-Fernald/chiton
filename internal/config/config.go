package config

import (
	"errors"
	"fmt"
	"io/fs"
	"net"
	"net/url"
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

// StateDirName is the per-project directory for the keyfile, finding baselines and LLM-rule state.
const StateDirName = ".chiton"

// StateDir returns ProjectDir's .chiton, so a server started in any directory finds the same state as the CLI.
// It stays the relative ".chiton" when CHITON_PROJECT_DIR is unset.
func StateDir() string { return filepath.Join(ProjectDir(), StateDirName) }

// Origin names the layer that supplied a config value.
type Origin string

const (
	OriginDefault   Origin = "default"
	OriginEnv       Origin = "environment"
	OriginUserRC    Origin = "$HOME/.chitonrc"
	OriginProjectRC Origin = "project .chitonrc"
)

// embedEndpointEnv is the variable viper maps onto embed.endpoint.
const embedEndpointEnv = "CHITON_EMBED_ENDPOINT"

// EmbedConfig configures the embedding provider used by `chiton embed`,
// `chiton search`, and the semantic MCP tools.
type EmbedConfig struct {
	Provider string `mapstructure:"provider"` // ollama | hash
	Model    string `mapstructure:"model"`
	Endpoint string `mapstructure:"endpoint"`
	Dim      int    `mapstructure:"dim"`
	// EndpointOrigin records which layer set Endpoint, so CheckEndpoint can refuse a repository-chosen host.
	EndpointOrigin Origin `mapstructure:"-"`
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

	// The config file is optional, but one that exists and fails to read or parse must stop the command,
	// since falling back to defaults would silently open a different store than the file names.
	if err := v.ReadInConfig(); err != nil {
		var notFound viper.ConfigFileNotFoundError
		if !errors.As(err, &notFound) && !errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("read config %s: %w", v.ConfigFileUsed(), err)
		}
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, err
	}
	cfg.GraphPath = resolveAgainst(dir, cfg.GraphPath)
	cfg.Embed.EndpointOrigin = endpointOrigin(v, dir)
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

// endpointOrigin reports which layer set embed.endpoint. Viper reads one .chitonrc, the project's before $HOME's.
func endpointOrigin(v *viper.Viper, dir string) Origin {
	switch {
	case os.Getenv(embedEndpointEnv) != "":
		return OriginEnv
	case !v.InConfig("embed.endpoint"):
		return OriginDefault
	case sameDir(filepath.Dir(v.ConfigFileUsed()), dir):
		return OriginProjectRC
	default:
		return OriginUserRC
	}
}

// sameDir reports whether a and b are one directory. A failed stat counts as a match, so a config file that
// cannot be placed gets the stricter project treatment.
func sameDir(a, b string) bool {
	ai, errA := os.Stat(a)
	bi, errB := os.Stat(b)
	return errA != nil || errB != nil || os.SameFile(ai, bi)
}

// CheckEndpoint refuses a non-loopback endpoint set by the project's .chitonrc, since a cloned repository must
// not choose where chiton sends document text and queries. The environment and $HOME/.chitonrc may point anywhere.
func (e EmbedConfig) CheckEndpoint() error {
	switch e.EndpointOrigin {
	case OriginEnv, OriginUserRC, OriginDefault:
		return nil
	}
	if isLoopbackURL(e.Endpoint) {
		return nil
	}
	return fmt.Errorf("embed.endpoint %q comes from the project's .chitonrc and is not a loopback address; "+
		"chiton will not send document text or queries off-device because a repository asked. "+
		"If you trust that server, set %s to it yourself", e.Endpoint, embedEndpointEnv)
}

// isLoopbackURL reports whether raw is an http(s) URL whose host is localhost or a loopback IP. Names are not
// resolved, so a DNS answer cannot turn a remote host into an allowed one.
func isLoopbackURL(raw string) bool {
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
		return false
	}
	host := u.Hostname()
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
