// Package crypto handles chiton's at-rest encryption: an Argon2id key from a passphrase that is never
// stored (the keyfile holds only salt+params, so the same inputs reopen BadgerDB) and AES-256-GCM files.
package crypto

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"

	"golang.org/x/crypto/argon2"

	"github.com/Eric-Fernald/chiton/internal/fsutil"
)

// KeyFileVersion is bumped when the on-disk keyfile format changes in a way
// that breaks readers.
const KeyFileVersion = 1

// Floors (not defaults) for a loaded keyfile's Argon2id params: a corrupt or tampered keyfile must
// fail clearly rather than panic inside argon2 (iterations/parallelism < 1) or derive a weak key.
const (
	minMemoryKiB   = 8 * 1024 // 8 MiB floor — well below the 64 MiB default
	minIterations  = 1
	minParallelism = 1
	minSaltBytes   = 16
	keyLengthBytes = 32 // AES-256; every chiton consumer requires exactly this
)

// KeyFile is the on-disk metadata for deriving a chiton encryption key from a passphrase (never
// persisted). Defaults follow OWASP 2024 Argon2id guidance: 64 MiB, 3 iterations, parallelism 4.
type KeyFile struct {
	Version     int    `json:"version"`
	KDF         string `json:"kdf"`         // "argon2id"
	SaltHex     string `json:"salt_hex"`    // 16-byte salt, hex-encoded
	MemoryKiB   uint32 `json:"memory_kib"`  // Argon2id memory cost
	Iterations  uint32 `json:"iterations"`  // Argon2id time cost
	Parallelism uint8  `json:"parallelism"` // Argon2id parallelism
	KeyLength   uint32 `json:"key_length"`  // bytes (32 for AES-256)
	Cipher      string `json:"cipher"`      // "AES-256-GCM"
}

// DefaultKeyFile returns a KeyFile with a fresh random salt and chiton's
// recommended Argon2id parameters.
func DefaultKeyFile() (*KeyFile, error) {
	salt := make([]byte, 16)
	if _, err := rand.Read(salt); err != nil {
		return nil, fmt.Errorf("generate salt: %w", err)
	}
	return &KeyFile{
		Version:     KeyFileVersion,
		KDF:         "argon2id",
		SaltHex:     hex.EncodeToString(salt),
		MemoryKiB:   64 * 1024, // 64 MiB
		Iterations:  3,
		Parallelism: 4,
		KeyLength:   32,
		Cipher:      "AES-256-GCM",
	}, nil
}

// Salt decodes the keyfile's hex salt.
func (kf *KeyFile) Salt() ([]byte, error) {
	b, err := hex.DecodeString(kf.SaltHex)
	if err != nil {
		return nil, fmt.Errorf("decode salt: %w", err)
	}
	return b, nil
}

// validate rejects malformed or sub-floor params: argon2.IDKey panics when Iterations or Parallelism
// is below 1, and low cost params or a short salt would silently weaken the key.
func (kf *KeyFile) validate() error {
	if kf.KDF != "argon2id" {
		return fmt.Errorf("unsupported KDF %q (want argon2id)", kf.KDF)
	}
	if kf.KeyLength != keyLengthBytes {
		return fmt.Errorf("key_length must be %d for AES-256, got %d", keyLengthBytes, kf.KeyLength)
	}
	if kf.Iterations < minIterations {
		return fmt.Errorf("iterations must be >= %d, got %d", minIterations, kf.Iterations)
	}
	if kf.Parallelism < minParallelism {
		return fmt.Errorf("parallelism must be >= %d, got %d", minParallelism, kf.Parallelism)
	}
	if kf.MemoryKiB < minMemoryKiB {
		return fmt.Errorf("memory_kib must be >= %d (8 MiB), got %d", minMemoryKiB, kf.MemoryKiB)
	}
	salt, err := kf.Salt()
	if err != nil {
		return err
	}
	if len(salt) < minSaltBytes {
		return fmt.Errorf("salt must be >= %d bytes, got %d", minSaltBytes, len(salt))
	}
	return nil
}

// DeriveKey runs Argon2id over the passphrase and the keyfile's parameters.
// Returns a key of length kf.KeyLength.
func (kf *KeyFile) DeriveKey(passphrase string) ([]byte, error) {
	if err := kf.validate(); err != nil {
		return nil, fmt.Errorf("keyfile: %w", err)
	}
	salt, err := kf.Salt()
	if err != nil {
		return nil, err
	}
	return argon2.IDKey(
		[]byte(passphrase),
		salt,
		kf.Iterations,
		kf.MemoryKiB,
		kf.Parallelism,
		kf.KeyLength,
	), nil
}

// Save atomically writes the keyfile to path with mode 0600. An existing file is kept unless force is set:
// regenerating the salt would lock the user out of existing encrypted data.
func (kf *KeyFile) Save(path string, force bool) error {
	if !force {
		if _, err := os.Stat(path); err == nil {
			return fmt.Errorf("keyfile already exists at %s — pass --force-keyfile to overwrite (LOSES ACCESS TO EXISTING ENCRYPTED DATA)", path)
		}
	}
	b, err := json.MarshalIndent(kf, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal keyfile: %w", err)
	}
	if err := fsutil.WriteFileAtomic(path, append(b, '\n'), 0o600); err != nil {
		return fmt.Errorf("write keyfile: %w", err)
	}
	return nil
}

// LoadKeyFile reads and validates the keyfile at path. A missing file returns (nil, nil), meaning
// "no encryption configured" as distinct from a parse error.
func LoadKeyFile(path string) (*KeyFile, error) {
	b, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read keyfile %s: %w", path, err)
	}
	var kf KeyFile
	if err := json.Unmarshal(b, &kf); err != nil {
		return nil, fmt.Errorf("parse keyfile %s: %w", path, err)
	}
	if kf.Version > KeyFileVersion {
		return nil, fmt.Errorf("keyfile %s: version %d not supported (this chiton speaks v%d)", path, kf.Version, KeyFileVersion)
	}
	if err := kf.validate(); err != nil {
		return nil, fmt.Errorf("keyfile %s: %w", path, err)
	}
	return &kf, nil
}
