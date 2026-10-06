package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/config"
	chitoncrypto "github.com/Eric-Fernald/chiton/internal/crypto"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
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
	t.Setenv(plaintextOKEnv, "1")
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

// An assistant-launched server has no terminal, so its error must say where to supply the passphrase.
func TestOpenGraphStoreExplainsMissingPassphrase(t *testing.T) {
	project := testProject(t)
	writeKeyfile(t, project)

	gs, err := openGraphStore(filepath.Join(project, "chiton-out", "graph.kuzu"), true)
	if err == nil {
		gs.Close()
		t.Fatal("openGraphStore succeeded without a passphrase")
	}
	if !errors.Is(err, chitoncrypto.ErrNoPassphrase) {
		t.Errorf("err = %v, want it to wrap ErrNoPassphrase", err)
	}
	if !strings.Contains(err.Error(), "environment that starts your assistant") {
		t.Errorf("err = %q, want the assistant fix spelled out", err)
	}
}

func TestInitNotesAssistantPassphrase(t *testing.T) {
	project := testProject(t)
	t.Setenv(chitoncrypto.EnvVarPassphrase, "pw")
	var out bytes.Buffer
	initc := initCmd()
	initc.SetArgs([]string{"--dir", project})
	initc.SetOut(&out)
	if err := initc.Execute(); err != nil {
		t.Fatalf("init: %v", err)
	}
	if !strings.Contains(out.String(), "serve --stdio") {
		t.Errorf("init output lacks the assistant passphrase note:\n%s", out.String())
	}
}

func TestOpenGraphStoreWritesOutGitignore(t *testing.T) {
	project := testProject(t)
	gs, err := openGraphStore(filepath.Join(project, "chiton-out", "graph.kuzu"), false)
	if err != nil {
		t.Fatal(err)
	}
	gs.Close()
	b, err := os.ReadFile(filepath.Join(project, "chiton-out", ".gitignore"))
	if err != nil || string(b) != "*\n" {
		t.Errorf("chiton-out/.gitignore = %q, %v; want \"*\n\"", b, err)
	}

	// A custom store directory may hold the user's own tracked files, so it is left alone.
	gs, err = openGraphStore(filepath.Join(project, "graphs", "g"), false)
	if err != nil {
		t.Fatal(err)
	}
	gs.Close()
	if _, err := os.Stat(filepath.Join(project, "graphs", ".gitignore")); !os.IsNotExist(err) {
		t.Errorf("wrote a .gitignore outside chiton-out (stat err %v)", err)
	}
}

func TestPlaintextNotice(t *testing.T) {
	t.Setenv(plaintextOKEnv, "")
	if msg := plaintextNotice("chiton-out/graph.kuzu"); !strings.Contains(msg, "not encrypted") || !strings.Contains(msg, plaintextOKEnv) {
		t.Errorf("notice = %q, want it to say the store is not encrypted and name %s", msg, plaintextOKEnv)
	}
	t.Setenv(plaintextOKEnv, "1")
	if msg := plaintextNotice("chiton-out/graph.kuzu"); msg != "" {
		t.Errorf("notice with %s=1 = %q, want silence", plaintextOKEnv, msg)
	}
}

// An encrypted project's cluster sidecar is written sealed and read back through the same resolved key.
func TestClusterSidecarSealedForEncryptedProject(t *testing.T) {
	project := testProject(t)
	writeKeyfile(t, project)
	t.Setenv(chitoncrypto.EnvVarPassphrase, "pw")
	graph := filepath.Join(project, "chiton-out", "graph.kuzu")
	gs, err := openGraphStore(graph, false)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	r := &cluster.Results{NodeCount: 1, Nodes: map[string]cluster.NodeResult{"n": {Label: "Initech acquisition"}}}
	path, err := writeClusterSidecar(graph, r)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cluster.LoadSidecar(path, nil); !errors.Is(err, chitoncrypto.ErrSealedNoKey) {
		t.Errorf("sidecar readable without the key (err %v); want it sealed", err)
	}
	got, err := loadClusterSidecar(graph)
	if err != nil || got.Nodes["n"].Label != "Initech acquisition" {
		t.Errorf("loadClusterSidecar = %+v, %v", got, err)
	}
}
