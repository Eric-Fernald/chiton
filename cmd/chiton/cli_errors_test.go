package main

import (
	"bytes"
	"context"
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
