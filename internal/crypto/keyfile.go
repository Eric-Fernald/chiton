// Package crypto handles chiton's at-rest encryption: passphrase prompting,
// Argon2id key derivation, AES-256-GCM file encryption for ad-hoc artifacts
// (cache.json, audit.jsonl, cluster.json), and key material for the graph
// store's built-in encryption.
//
// Design:
//
//   - .chiton/keyfile.json holds the salt and Argon2id parameters. The
//     passphrase itself is never written to disk. Without the passphrase,
//     the keyfile alone reveals nothing.
//   - The derived key is 32 bytes (AES-256). The same passphrase + same
//     keyfile always produce the same key, so we can re-open BadgerDB
//     (which stores encrypted SSTs) deterministically.
//   - For one-off files (cache.json, audit.jsonl), we wrap content in
//     AES-256-GCM with a fresh nonce per write. The nonce is prepended to
//     the ciphertext on disk.
package crypto

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"

	"golang.org/x/crypto/argon2"
)

// KeyFileVersion is bumped when the on-disk keyfile format changes in a way
// that breaks readers.
const KeyFileVersion = 1

// Minimum Argon2id parameters accepted from a loaded keyfile. These are floors,
// not the recommended values (see DefaultKeyFile) — they exist so a corrupt or
// tampered keyfile fails with a clear error instead of panicking inside argon2
// (which aborts when iterations or parallelism are below 1) or silently deriving
// a trivially weak key.
const (
	minMemoryKiB   = 8 * 1024 // 8 MiB floor — well below the 64 MiB default
	minIterations  = 1
	minParallelism = 1
	minSaltBytes   = 16
	keyLengthBytes = 32 // AES-256; every chiton consumer requires exactly this
)

// KeyFile is the on-disk metadata for deriving a chiton encryption key from
// a passphrase. The passphrase itself is never persisted.
//
// Argon2id parameters follow OWASP's 2024 guidance for general use:
// 64 MiB memory, 3 iterations, parallelism 4, 32-byte output. These can be
// raised on newer hardware by writing a higher Version with stronger params.
type KeyFile struct {
	Version     int    `json:"version"`
	KDF         string `json:"kdf"`           // "argon2id"
	SaltHex     string `json:"salt_hex"`      // 16-byte salt, hex-encoded
	MemoryKiB   uint32 `json:"memory_kib"`    // Argon2id memory cost
	Iterations  uint32 `json:"iterations"`    // Argon2id time cost
	Parallelism uint8  `json:"parallelism"`   // Argon2id parallelism
	KeyLength   uint32 `json:"key_length"`    // bytes (32 for AES-256)
	Cipher      string `json:"cipher"`        // "AES-256-GCM"
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

// validate checks the keyfile's parameters are well-formed and meet chiton's
// minimum strength, returning a descriptive error otherwise. It guards against
// corrupt or tampered keyfiles: argon2.IDKey panics when Iterations or
// Parallelism is below 1, and sub-floor cost params or a short salt would
// silently weaken the derived key.
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

// Save writes the keyfile to path with restrictive permissions. Existing
// files are not overwritten unless force is true — protecting users from
// regenerating a salt and locking themselves out.
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
	if err := os.WriteFile(path, append(b, '\n'), 0o600); err != nil {
		return fmt.Errorf("write keyfile: %w", err)
	}
	return nil
}

// LoadKeyFile reads and parses a keyfile from path. Missing files return
// (nil, nil) so callers can tell "no encryption configured" apart from a
// parse error.
func LoadKeyFile(path string) (*KeyFile, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
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
