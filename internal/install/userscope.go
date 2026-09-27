package install

import (
	"os"
	"path/filepath"
)

// homeDir resolves the current user's home directory; it is a var so tests can redirect user-scope
// writes to a temp directory.
var homeDir = os.UserHomeDir

// userPath joins parts onto the user's home directory (via homeDir, so tests can override it).
func userPath(parts ...string) (string, error) {
	home, err := homeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(append([]string{home}, parts...)...), nil
}

// copilotConfigDir returns Copilot CLI's config directory: $COPILOT_HOME, which replaces the whole
// ~/.copilot path when set, else ~/.copilot.
func copilotConfigDir() (string, error) {
	if h := os.Getenv("COPILOT_HOME"); h != "" {
		return h, nil
	}
	return userPath(".copilot")
}
