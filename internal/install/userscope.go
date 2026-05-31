package install

import (
	"os"
	"path/filepath"
)

// homeDir resolves the current user's home directory. It is a package var so
// tests can redirect user-scope writes (Codex, Antigravity, Copilot CLI) to a
// temp directory without touching the real home.
var homeDir = os.UserHomeDir

// userPath joins parts onto the user's home directory. Installers that write
// user/global config (e.g. ~/.codex/config.toml) resolve their target through
// this so the home lookup is overridable in tests.
func userPath(parts ...string) (string, error) {
	home, err := homeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(append([]string{home}, parts...)...), nil
}

// copilotConfigDir returns the base directory GitHub Copilot CLI reads its
// config from. COPILOT_HOME, when set, REPLACES the entire ~/.copilot path
// (per Copilot CLI docs); otherwise it is ~/.copilot.
func copilotConfigDir() (string, error) {
	if h := os.Getenv("COPILOT_HOME"); h != "" {
		return h, nil
	}
	return userPath(".copilot")
}
