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

// GetPassphrase returns CHITON_PASSPHRASE, else prompts with prompt on the terminal without echo.
// With confirm set, it asks twice and requires a match. Set confirm when creating a keyfile.
func GetPassphrase(prompt string, confirm bool) (string, error) {
	if env := os.Getenv(EnvVarPassphrase); env != "" {
		return env, nil
	}
	fd := int(os.Stdin.Fd())
	if !term.IsTerminal(fd) {
		return "", ErrNoPassphrase
	}
	fmt.Fprint(os.Stderr, prompt+" ")
	first, err := term.ReadPassword(fd)
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
	second, err := term.ReadPassword(fd)
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return "", fmt.Errorf("read confirmation: %w", err)
	}
	if string(first) != string(second) {
		return "", fmt.Errorf("passphrases do not match")
	}
	return string(first), nil
}
