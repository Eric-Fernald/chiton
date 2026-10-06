package main

import (
	"context"
	"errors"
	"io"
	"os"
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

// writeKeyfile saves a keyfile with no verifier, as chiton wrote before verifiers, into project/.chiton.
func writeKeyfile(t *testing.T, project string) string {
	t.Helper()
	path := chitoncrypto.ResolveKeyfilePath(filepath.Join(project, config.StateDirName))
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	kf, err := chitoncrypto.DefaultKeyFile()
	if err != nil {
		t.Fatal(err)
	}
	if err := kf.Save(path, false); err != nil {
		t.Fatal(err)
	}
	return path
}

// init must record a verifier, so a later typo fails instead of creating a store under the wrong key.
func TestInitKeyfileRejectsMistypedPassphrase(t *testing.T) {
	project := testProject(t)
	t.Setenv(chitoncrypto.EnvVarPassphrase, "correct horse")
	initc := initCmd()
	initc.SetArgs([]string{"--dir", project})
	initc.SetOut(io.Discard)
	if err := initc.Execute(); err != nil {
		t.Fatalf("init: %v", err)
	}

	t.Setenv(chitoncrypto.EnvVarPassphrase, "correct hrose")
	graph := filepath.Join(project, "chiton-out", "graph.kuzu")
	gs, err := openGraphStore(graph, false)
	if !errors.Is(err, chitoncrypto.ErrWrongPassphrase) {
		if gs != nil {
			gs.Close()
		}
		t.Fatalf("openGraphStore with a typo: err = %v, want ErrWrongPassphrase", err)
	}
	if _, err := os.Stat(graph); !os.IsNotExist(err) {
		t.Errorf("a store was created under the mistyped passphrase (stat err %v)", err)
	}
}

// A keyfile from before verifiers existed gains one on the first open the store accepts.
func TestOpenGraphStoreUpgradesLegacyKeyfile(t *testing.T) {
	project := testProject(t)
	kfPath := writeKeyfile(t, project)
	t.Setenv(chitoncrypto.EnvVarPassphrase, "pw")

	gs, err := openGraphStore(filepath.Join(project, "chiton-out", "graph.kuzu"), false)
	if err != nil {
		t.Fatalf("openGraphStore: %v", err)
	}
	gs.Close()

	kf, err := chitoncrypto.LoadKeyFile(kfPath)
	if err != nil {
		t.Fatal(err)
	}
	if kf.Verifier == "" {
		t.Error("legacy keyfile was not upgraded with a verifier after a successful open")
	}
}
