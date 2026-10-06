package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// isolate points HOME at a scratch directory so a developer's real ~/.chitonrc can never leak into these tests.
// Viper reads HOMEDRIVE+HOMEPATH before USERPROFILE on Windows, so those are cleared too. It returns the home.
func isolate(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("HOMEDRIVE", "")
	t.Setenv("HOMEPATH", "")
	t.Setenv(ProjectDirEnv, "")
	t.Setenv("CHITON_GRAPH_PATH", "")
	t.Setenv("CHITON_EMBED_ENDPOINT", "")
	return home
}

// A .chitonrc that fails to parse must stop the command, because falling back to defaults silently opens a
// different store than the one the file names.
func TestMalformedChitonrcIsAnError(t *testing.T) {
	isolate(t)
	project := t.TempDir()
	rc := filepath.Join(project, ".chitonrc")
	if err := os.WriteFile(rc, []byte("graph_path = \"custom/store\"\n[embed\n"), 0o644); err != nil {
		t.Fatalf("write .chitonrc: %v", err)
	}
	t.Setenv(ProjectDirEnv, project)

	cfg, err := Load()
	if err == nil {
		t.Fatalf("Load succeeded with GraphPath %q, want a parse error", cfg.GraphPath)
	}
	if !strings.Contains(err.Error(), rc) {
		t.Errorf("error %q does not name the file %s", err, rc)
	}
}

func TestLoadDefaultsStayRelative(t *testing.T) {
	isolate(t)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got, want := cfg.GraphPath, "./chiton-out/graph.kuzu"; got != want {
		t.Errorf("GraphPath = %q, want the historical relative default %q", got, want)
	}
	if cfg.Embed.Provider != "ollama" {
		t.Errorf("Embed.Provider = %q, want ollama", cfg.Embed.Provider)
	}
	if cfg.Semantic.ConfidenceThreshold != 0.5 {
		t.Errorf("ConfidenceThreshold = %v, want 0.5", cfg.Semantic.ConfidenceThreshold)
	}
}

func TestVectorDefaults(t *testing.T) {
	isolate(t)
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	want := VectorConfig{Index: "auto", HNSWThreshold: 100000, HNSWM: 16, HNSWEfSearch: 128, HNSWEfConstruction: 100}
	if cfg.Vector != want {
		t.Errorf("Vector = %+v, want %+v", cfg.Vector, want)
	}
}

func TestVectorEnvOverrides(t *testing.T) {
	isolate(t)
	t.Setenv("CHITON_VECTOR_INDEX", "flat")
	t.Setenv("CHITON_VECTOR_HNSW_THRESHOLD", "5000")
	t.Setenv("CHITON_VECTOR_HNSW_M", "24")
	t.Setenv("CHITON_VECTOR_HNSW_EF_SEARCH", "64")
	t.Setenv("CHITON_VECTOR_HNSW_EF_CONSTRUCTION", "300")
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	want := VectorConfig{Index: "flat", HNSWThreshold: 5000, HNSWM: 24, HNSWEfSearch: 64, HNSWEfConstruction: 300}
	if cfg.Vector != want {
		t.Errorf("Vector = %+v, want %+v", cfg.Vector, want)
	}
}

func TestVectorFromChitonrc(t *testing.T) {
	isolate(t)
	project := t.TempDir()
	rc := "[vector]\nindex = \"hnsw\"\nhnsw_ef_search = 256\n"
	if err := os.WriteFile(filepath.Join(project, ".chitonrc"), []byte(rc), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(ProjectDirEnv, project)
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Vector.Index != "hnsw" || cfg.Vector.HNSWEfSearch != 256 || cfg.Vector.HNSWM != 16 {
		t.Errorf("Vector = %+v, want index hnsw, ef_search 256, default M", cfg.Vector)
	}
}

func TestProjectDirAnchorsRelativeGraphPath(t *testing.T) {
	isolate(t)
	project := t.TempDir()
	t.Setenv(ProjectDirEnv, project)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	want := filepath.Join(project, "chiton-out", "graph.kuzu")
	if cfg.GraphPath != want {
		t.Errorf("GraphPath = %q, want %q", cfg.GraphPath, want)
	}
}

// An unsubstituted ${CLAUDE_PROJECT_DIR} must read as unset rather than anchor graph_path under a
// directory that cannot exist.
func TestUnexpandedProjectDirTemplateIsIgnored(t *testing.T) {
	isolate(t)
	t.Setenv(ProjectDirEnv, "${CLAUDE_PROJECT_DIR}")

	if got := ProjectDir(); got != "." {
		t.Errorf("ProjectDir() = %q, want %q for an unexpanded template", got, ".")
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got, want := cfg.GraphPath, "./chiton-out/graph.kuzu"; got != want {
		t.Errorf("GraphPath = %q, want the cwd-relative default %q", got, want)
	}
}

func TestProjectDirReadsChitonrcThere(t *testing.T) {
	isolate(t)
	project := t.TempDir()
	rc := filepath.Join(project, ".chitonrc")
	body := "graph_path = \"custom/store\"\nlog_level = \"debug\"\n"
	if err := os.WriteFile(rc, []byte(body), 0o644); err != nil {
		t.Fatalf("write .chitonrc: %v", err)
	}
	t.Setenv(ProjectDirEnv, project)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	want := filepath.Join(project, "custom", "store")
	if cfg.GraphPath != want {
		t.Errorf("GraphPath = %q, want the .chitonrc value anchored at the project dir %q", cfg.GraphPath, want)
	}
	if cfg.LogLevel != "debug" {
		t.Errorf("LogLevel = %q, want debug", cfg.LogLevel)
	}
}

func TestAbsoluteGraphPathIsLeftAlone(t *testing.T) {
	isolate(t)
	project := t.TempDir()
	elsewhere := filepath.Join(t.TempDir(), "graph.kuzu")

	rc := filepath.Join(project, ".chitonrc")
	body := "graph_path = " + quoteTOML(elsewhere) + "\n"
	if err := os.WriteFile(rc, []byte(body), 0o644); err != nil {
		t.Fatalf("write .chitonrc: %v", err)
	}
	t.Setenv(ProjectDirEnv, project)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.GraphPath != elsewhere {
		t.Errorf("GraphPath = %q, want the absolute path unchanged %q", cfg.GraphPath, elsewhere)
	}
}

func TestGraphPathEnvOverridesConfigFile(t *testing.T) {
	isolate(t)
	project := t.TempDir()
	rc := filepath.Join(project, ".chitonrc")
	if err := os.WriteFile(rc, []byte("graph_path = \"from-file\"\n"), 0o644); err != nil {
		t.Fatalf("write .chitonrc: %v", err)
	}
	t.Setenv(ProjectDirEnv, project)
	t.Setenv("CHITON_GRAPH_PATH", "from-env")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	want := filepath.Join(project, "from-env")
	if cfg.GraphPath != want {
		t.Errorf("GraphPath = %q, want the env override anchored at the project dir %q", cfg.GraphPath, want)
	}
}

func TestProjectDir(t *testing.T) {
	isolate(t)
	if got := ProjectDir(); got != "." {
		t.Errorf("ProjectDir() = %q with no env set, want \".\"", got)
	}
	t.Setenv(ProjectDirEnv, "  /some/where  ")
	if got := ProjectDir(); got != "/some/where" {
		t.Errorf("ProjectDir() = %q, want the trimmed value", got)
	}
}

// quoteTOML renders a Windows-safe TOML basic string (backslashes escaped).
func quoteTOML(s string) string {
	out := make([]rune, 0, len(s)+2)
	out = append(out, '"')
	for _, r := range s {
		if r == '\\' || r == '"' {
			out = append(out, '\\')
		}
		out = append(out, r)
	}
	out = append(out, '"')
	return string(out)
}

func TestEmbedDimEnvOverrideApplies(t *testing.T) {
	// embed.dim has no SetDefault, so viper's Unmarshal used to skip it and
	// silently ignore CHITON_EMBED_DIM.
	isolate(t)
	t.Setenv("CHITON_EMBED_DIM", "1536")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Embed.Dim != 1536 {
		t.Errorf("Embed.Dim = %d, want 1536 from CHITON_EMBED_DIM", cfg.Embed.Dim)
	}
}

func TestEmbedDimDefaultsToZero(t *testing.T) {
	// Zero is load-bearing: it lets each provider pick its native dimension
	// (Ollama 768, hash 256) instead of inheriting the other's.
	isolate(t)
	t.Setenv("CHITON_EMBED_DIM", "")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Embed.Dim != 0 {
		t.Errorf("Embed.Dim = %d with no override, want 0", cfg.Embed.Dim)
	}
}

func writeRC(t *testing.T, dir, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, ".chitonrc"), []byte(body), 0o644); err != nil {
		t.Fatalf("write .chitonrc: %v", err)
	}
}

// A cloned repository's .chitonrc must not be able to send embedded text and queries to its own server.
func TestProjectRCRemoteEmbedEndpointRefused(t *testing.T) {
	isolate(t)
	project := t.TempDir()
	writeRC(t, project, "[embed]\nendpoint = \"https://embed.attacker.example\"\n")
	t.Setenv(ProjectDirEnv, project)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Embed.EndpointOrigin != OriginProjectRC {
		t.Errorf("EndpointOrigin = %q, want %q", cfg.Embed.EndpointOrigin, OriginProjectRC)
	}
	if err := cfg.Embed.CheckEndpoint(); err == nil || !strings.Contains(err.Error(), "CHITON_EMBED_ENDPOINT") {
		t.Errorf("CheckEndpoint() = %v, want a refusal naming CHITON_EMBED_ENDPOINT", err)
	}
}

func TestEmbedEndpointAllowedOrigins(t *testing.T) {
	remote := "[embed]\nendpoint = \"https://embed.attacker.example\"\n"
	cases := []struct {
		name, projectRC, homeRC, env string
		want                         Origin
	}{
		{name: "project loopback IP", projectRC: "[embed]\nendpoint = \"http://127.0.0.1:11434\"\n", want: OriginProjectRC},
		{name: "project localhost", projectRC: "[embed]\nendpoint = \"http://localhost:9999\"\n", want: OriginProjectRC},
		{name: "env overrides project", projectRC: remote, env: "https://embed.corp.example", want: OriginEnv},
		{name: "home config", homeRC: "[embed]\nendpoint = \"https://embed.corp.example\"\n", want: OriginUserRC},
		{name: "built-in default", want: OriginDefault},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			home := isolate(t)
			project := t.TempDir()
			if tc.projectRC != "" {
				writeRC(t, project, tc.projectRC)
			}
			if tc.homeRC != "" {
				writeRC(t, home, tc.homeRC)
			}
			t.Setenv(ProjectDirEnv, project)
			t.Setenv("CHITON_EMBED_ENDPOINT", tc.env)

			cfg, err := Load()
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			if cfg.Embed.EndpointOrigin != tc.want {
				t.Errorf("EndpointOrigin = %q, want %q", cfg.Embed.EndpointOrigin, tc.want)
			}
			if err := cfg.Embed.CheckEndpoint(); err != nil {
				t.Errorf("CheckEndpoint() = %v, want nil", err)
			}
		})
	}
}

func TestIsLoopbackURL(t *testing.T) {
	for raw, want := range map[string]bool{
		"http://localhost:11434":            true,
		"https://LOCALHOST":                 true,
		"http://127.0.0.1:11434":            true,
		"http://[::1]:11434":                true,
		"http://localhost.attacker.example": false,
		"http://127.0.0.1.attacker.example": false,
		"https://embed.attacker.example":    false,
		"localhost:11434":                   false, // no scheme: url.Parse reads "localhost" as the scheme
		"ftp://127.0.0.1":                   false,
	} {
		if got := isLoopbackURL(raw); got != want {
			t.Errorf("isLoopbackURL(%q) = %v, want %v", raw, got, want)
		}
	}
}
