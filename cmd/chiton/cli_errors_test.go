package main

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/config"
)

// runCLI runs the root command with args in an empty project and returns everything cobra printed plus the error.
func runCLI(t *testing.T, args ...string) (string, error) {
	t.Helper()
	t.Setenv(config.ProjectDirEnv, t.TempDir())
	t.Setenv(plaintextOKEnv, "1")
	var out bytes.Buffer
	cmd := rootCmd()
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs(args)
	err := cmd.ExecuteContext(context.Background())
	return out.String(), err
}

// TestRuntimeErrorPrintsNoUsage checks a command that fails while running leaves the error to main, which
// prints it once, and does not dump the flag help.
func TestRuntimeErrorPrintsNoUsage(t *testing.T) {
	out, err := runCLI(t, "query", "anything")
	if err == nil {
		t.Fatal("query on a project with no graph succeeded")
	}
	if strings.Contains(out, "Usage:") || strings.Contains(out, "Error:") {
		t.Errorf("a runtime error printed usage or a second copy of the error:\n%s", out)
	}
}

// TestUsageErrorStillPrintsUsage checks a mistyped flag or missing argument still shows the command's help.
func TestUsageErrorStillPrintsUsage(t *testing.T) {
	for _, args := range [][]string{{"query"}, {"query", "x", "--no-such-flag"}} {
		out, err := runCLI(t, args...)
		if err == nil || !strings.Contains(out, "Usage:") {
			t.Errorf("chiton %v: err = %v, output has no usage:\n%s", args, err, out)
		}
	}
}

// emptyStdin points os.Stdin at an empty file for the test, so a stdio server sees EOF and returns.
func emptyStdin(t *testing.T) {
	t.Helper()
	f, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stdin
	os.Stdin = f
	t.Cleanup(func() {
		os.Stdin = old
		f.Close()
	})
}

// TestServeStdioStartsWithoutAGraph checks an assistant-launched server starts in a project nobody has
// indexed yet. Failing there would only show up as chiton's tools being absent.
func TestServeStdioStartsWithoutAGraph(t *testing.T) {
	emptyStdin(t)
	out, err := runCLI(t, "serve", "--stdio")
	if err != nil {
		t.Fatalf("serve --stdio on a project with no graph failed: %v", err)
	}
	if !strings.Contains(out, "chiton build .") {
		t.Errorf("serve printed %q, want a note that names 'chiton build .'", out)
	}
}

// TestServeDaemonStillNeedsAGraph checks the daemon transports, which a person starts by hand, keep failing
// with the fix when there is no graph.
func TestServeDaemonStillNeedsAGraph(t *testing.T) {
	_, err := runCLI(t, "serve", "--http", "127.0.0.1:0")
	if err == nil || !strings.Contains(err.Error(), "chiton build") {
		t.Errorf("serve --http on a project with no graph: err = %v, want a hint to run 'chiton build'", err)
	}
}

// TestReadOnlyOpenOfUnbuiltStoreSaysToBuild checks reading a project that was never indexed names the fix,
// not Badger's missing-directory error.
func TestReadOnlyOpenOfUnbuiltStoreSaysToBuild(t *testing.T) {
	_, err := runCLI(t, "query", "anything")
	if err == nil {
		t.Fatal("query on a project with no graph succeeded")
	}
	if msg := err.Error(); !strings.Contains(msg, "chiton build") || strings.Contains(msg, "Cannot find directory") {
		t.Errorf("error = %q, want a hint to run 'chiton build' without Badger's raw message", msg)
	}
}
