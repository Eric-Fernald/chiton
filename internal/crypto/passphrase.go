package crypto

import (
	"errors"
	"fmt"
	"os"

	"golang.org/x/term"
)

// EnvVarPassphrase is the env var read before any interactive prompt. Set this
// in CI / non-interactive environments to avoid the prompt.
const EnvVarPassphrase = "CHITON_PASSPHRASE"

// ErrNoPassphrase is returned by GetPassphrase when neither the env var is
// set nor an interactive terminal is available.
var ErrNoPassphrase = errors.New("no passphrase available: set CHITON_PASSPHRASE or run interactively")

// GetPassphrase returns the chiton passphrase from CHITON_PASSPHRASE, or by
// prompting on the controlling terminal (no echo). The prompt argument is the
// label shown when interactive — typically "Enter chiton passphrase:".
//
// Confirm=true asks the user to enter the passphrase twice and verifies they
// match — appropriate when creating a new keyfile.
func GetPassphrase(prompt string, confirm bool) (string, error) {
	if env := os.Getenv(EnvVarPassphrase); env != "" {
		return env, nil
	}
	if !term.IsTerminal(int(os.Stdin.Fd())) {
		return "", ErrNoPassphrase
	}
	fmt.Fprint(os.Stderr, prompt+" ")
	first, err := term.ReadPassword(int(os.Stdin.Fd()))
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return "", fmt.Errorf("read passphrase: %w", err)
	}
	if len(first) == 0 {
		return "", fmt.Errorf("empty passphrase")
	}
	if !confirm {
		return string(first), nil
	}
	fmt.Fprint(os.Stderr, "Confirm: ")
	second, err := term.ReadPassword(int(os.Stdin.Fd()))
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return "", fmt.Errorf("read confirmation: %w", err)
	}
	if string(first) != string(second) {
		return "", fmt.Errorf("passphrases do not match")
	}
	return string(first), nil
}
