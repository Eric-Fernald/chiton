package browser

import (
	"os"
	"path/filepath"
	"runtime"
)

// browserRoot is a browser's "User Data" root, under which per-profile dirs live.
type browserRoot struct {
	name string
	root string
}

// DetectDBs returns the History databases of installed Chromium-family browsers
// on the current OS — the Default profile plus any numbered "Profile N" dirs —
// filtered to files that actually exist. Returns nil when none are found.
func DetectDBs() []DetectedDB {
	var out []DetectedDB
	for _, b := range browserRoots() {
		if b.root == "" {
			continue
		}
		for _, prof := range profileDirs(b.root) {
			p := filepath.Join(prof, "History")
			if isFile(p) {
				out = append(out, DetectedDB{Browser: b.name, Path: p})
			}
		}
	}
	return out
}

// browserRoots returns the per-OS "User Data" roots for Chrome, Edge, and Brave.
func browserRoots() []browserRoot {
	home, _ := os.UserHomeDir()
	switch runtime.GOOS {
	case "windows":
		local := os.Getenv("LOCALAPPDATA")
		if local == "" && home != "" {
			local = filepath.Join(home, "AppData", "Local")
		}
		return []browserRoot{
			{"chrome", filepath.Join(local, "Google", "Chrome", "User Data")},
			{"edge", filepath.Join(local, "Microsoft", "Edge", "User Data")},
			{"brave", filepath.Join(local, "BraveSoftware", "Brave-Browser", "User Data")},
		}
	case "darwin":
		if home == "" {
			return nil
		}
		support := filepath.Join(home, "Library", "Application Support")
		return []browserRoot{
			{"chrome", filepath.Join(support, "Google", "Chrome")},
			{"edge", filepath.Join(support, "Microsoft Edge")},
			{"brave", filepath.Join(support, "BraveSoftware", "Brave-Browser")},
		}
	default: // linux and other unix
		if home == "" {
			return nil
		}
		cfg := os.Getenv("XDG_CONFIG_HOME")
		if cfg == "" {
			cfg = filepath.Join(home, ".config")
		}
		return []browserRoot{
			{"chrome", filepath.Join(cfg, "google-chrome")},
			{"edge", filepath.Join(cfg, "microsoft-edge")},
			{"brave", filepath.Join(cfg, "BraveSoftware", "Brave-Browser")},
		}
	}
}

// profileDirs returns the "Default" profile dir plus any "Profile N" dirs under
// root. Missing dirs are harmless — DetectDBs filters to existing History files.
func profileDirs(root string) []string {
	dirs := []string{filepath.Join(root, "Default")}
	if matches, err := filepath.Glob(filepath.Join(root, "Profile *")); err == nil {
		dirs = append(dirs, matches...)
	}
	return dirs
}

func isFile(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && !fi.IsDir()
}
