package crypto

import (
	"fmt"
	"path/filepath"
)

// KeyfileName is the filename chiton writes next to .chiton/schema_version
// (or wherever the user's graph lives).
const KeyfileName = "keyfile.json"

// ResolveKeyfilePath returns the conventional keyfile location for a given
// chiton directory. The chitonDir argument is typically <repo>/.chiton.
func ResolveKeyfilePath(chitonDir string) string {
	return filepath.Join(chitonDir, KeyfileName)
}

// ResolveKeyForGraph finds a keyfile beside graphPath, in a .chiton beside its directory, or in stateDir
// (normally config.StateDir()), then derives the key via GetPassphrase. With no keyfile it returns (nil, nil, nil).
func ResolveKeyForGraph(graphPath, stateDir string) ([]byte, *KeyFile, error) {
	candidates := []string{
		ResolveKeyfilePath(filepath.Dir(graphPath)),
		ResolveKeyfilePath(filepath.Join(filepath.Dir(graphPath), "..", ".chiton")),
		ResolveKeyfilePath(stateDir),
	}
	var found string
	var kf *KeyFile
	for _, candidate := range candidates {
		loaded, err := LoadKeyFile(candidate)
		if err != nil {
			return nil, nil, err
		}
		if loaded != nil {
			found = candidate
			kf = loaded
			break
		}
	}
	if kf == nil {
		return nil, nil, nil
	}
	passphrase, err := GetPassphrase("Enter chiton passphrase:", false)
	if err != nil {
		return nil, kf, fmt.Errorf("%s: %w", found, err)
	}
	key, err := kf.DeriveKey(passphrase)
	if err != nil {
		return nil, kf, fmt.Errorf("derive key from %s: %w", found, err)
	}
	return key, kf, nil
}
