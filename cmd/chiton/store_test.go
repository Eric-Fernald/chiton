package main

import (
	"context"
	"io"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/config"
	chitoncrypto "github.com/Eric-Fernald/chiton/internal/crypto"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

// testProject anchors chiton at a scratch project, as the plugin anchors an assistant-launched server, and
// returns it. The working directory is left alone to prove nothing depends on it.
func testProject(t *testing.T) string {
	t.Helper()
	project := t.TempDir()
	t.Setenv(config.ProjectDirEnv, project)
	t.Setenv("CHITON_GRAPH_PATH", "")
	t.Setenv(chitoncrypto.EnvVarPassphrase, "")
	return project
}

// diff_findings baselines and LLM-rule state must live under CHITON_PROJECT_DIR, not the server's cwd.
func TestServeStateFollowsProjectDir(t *testing.T) {
	project := testProject(t)
	gs, err := badgerstore.Open(filepath.Join(project, "chiton-out", "graph.kuzu"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())
	cmd.SetErr(io.Discard)
	cfg := &config.Config{Embed: config.EmbedConfig{Provider: "hash"}}

	deps := buildSemanticDeps(cmd, cfg, gs, nil)
	if want := filepath.Join(project, ".chiton", "baselines"); deps.BaselineDir != want {
		t.Errorf("BaselineDir = %q, want %q", deps.BaselineDir, want)
	}
}
