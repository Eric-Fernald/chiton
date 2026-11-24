package main

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestReadSecretPrefersFileAndTrims(t *testing.T) {
	t.Setenv("CHITON_TEST_SECRET", "from-env")
	p := filepath.Join(t.TempDir(), "secret")
	if err := os.WriteFile(p, []byte("  from-file\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got, err := readSecret(p, "CHITON_TEST_SECRET"); err != nil || got != "from-file" {
		t.Errorf("readSecret(file) = %q, %v; want from-file", got, err)
	}
	if got, err := readSecret("", "CHITON_TEST_SECRET"); err != nil || got != "from-env" {
		t.Errorf("readSecret(env) = %q, %v; want from-env", got, err)
	}
	bom := filepath.Join(t.TempDir(), "bom")
	if err := os.WriteFile(bom, []byte("\ufefffrom-file\r\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got, err := readSecret(bom, "CHITON_TEST_SECRET"); err != nil || got != "from-file" {
		t.Errorf("readSecret(BOM file) = %q, %v; want from-file without the BOM", got, err)
	}
	empty := filepath.Join(t.TempDir(), "empty")
	if err := os.WriteFile(empty, []byte("\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readSecret(empty, "CHITON_TEST_SECRET"); err == nil {
		t.Error("readSecret accepted an empty secret file")
	}
}

// isolateWebhooks keeps these commands from reaching a real store or vendor API if a check regresses.
func isolateWebhooks(t *testing.T) {
	t.Helper()
	t.Setenv("CHITON_PROJECT_DIR", t.TempDir())
	for _, v := range []string{"CHITON_GRAPH_PATH", "CHITON_SLACK_TOKEN", "CHITON_SLACK_SIGNING_SECRET", "CHITON_TEAMS_CLIENT_STATE",
		"CHITON_TEAMS_TENANT_ID", "CHITON_TEAMS_CLIENT_ID", "CHITON_TEAMS_CLIENT_SECRET", "CHITON_TEAMS_USER_ID"} {
		t.Setenv(v, "")
	}
}

func execQuiet(cmd *cobra.Command, args ...string) error {
	cmd.SetArgs(args)
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	return cmd.Execute()
}

// A secret passed as a flag value lands in ps and shell history, so the old flags must refuse and say what to use.
func TestSecretFlagsRemoved(t *testing.T) {
	isolateWebhooks(t)
	if err := execQuiet(slackEventsCmd(), "--signing-secret", "x"); err == nil || !strings.Contains(err.Error(), "--signing-secret-file") {
		t.Errorf("slack serve-events --signing-secret: err = %v, want a pointer to --signing-secret-file", err)
	}
	if err := execQuiet(teamsWebhookCmd(), "--client-state", "x"); err == nil || !strings.Contains(err.Error(), "--client-state-file") {
		t.Errorf("teams serve-webhook --client-state: err = %v, want a pointer to --client-state-file", err)
	}
}

// The file flag must satisfy the secret check; the command then stops at the missing Slack token.
func TestSigningSecretFileIsUsed(t *testing.T) {
	isolateWebhooks(t)
	p := filepath.Join(t.TempDir(), "signing")
	if err := os.WriteFile(p, []byte("s3cret\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	err := execQuiet(slackEventsCmd(), "--signing-secret-file", p)
	if err == nil || strings.Contains(err.Error(), "signing secret is required") || strings.Contains(err.Error(), "unknown flag") {
		t.Errorf("err = %v, want the secret accepted and a later failure (missing CHITON_SLACK_TOKEN)", err)
	}
}
