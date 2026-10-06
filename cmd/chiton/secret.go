package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

// readSecret returns the trimmed contents of file when it is set, else $env. Secrets never come from a flag
// value, which ps and shell history would record.
func readSecret(file, env string) (string, error) {
	if file == "" {
		return os.Getenv(env), nil
	}
	b, err := os.ReadFile(file)
	if err != nil {
		return "", fmt.Errorf("read secret file: %w", err)
	}
	// PowerShell 5.1's `Out-File -Encoding utf8` prepends a BOM, which would otherwise end up in the secret.
	s := strings.TrimSpace(strings.TrimPrefix(string(b), "\ufeff"))
	if s == "" {
		return "", fmt.Errorf("secret file %s is empty", file)
	}
	return s, nil
}

// secretFlagRemoved errors when the removed secret-bearing flag name was passed, naming what replaced it.
func secretFlagRemoved(cmd *cobra.Command, name, fileFlag, env string) error {
	if !cmd.Flags().Changed(name) {
		return nil
	}
	return fmt.Errorf("--%s was removed because a secret on the command line shows up in ps and shell history; "+
		"use --%s <path> or set %s", name, fileFlag, env)
}
