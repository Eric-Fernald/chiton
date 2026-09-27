package config

import (
	"os"
	"path/filepath"
	"testing"
)

// isolate points HOME at a scratch directory so a developer's real ~/.chitonrc
// can never leak into these tests.
func isolate(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv(ProjectDirEnv, "")
	t.Setenv("CHITON_GRAPH_PATH", "")
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
