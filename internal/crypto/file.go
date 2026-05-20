package crypto

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"fmt"
)

// Envelope is the on-disk layout for an encrypted file written by EncryptBytes:
// a 1-byte version, a 12-byte GCM nonce, then GCM-sealed ciphertext (which
// includes the auth tag). The version byte lets us evolve the format without
// breaking existing files.
const envelopeVersion byte = 1

const (
	gcmNonceLen   = 12
	headerLen     = 1 + gcmNonceLen
)

// EncryptBytes seals plaintext with AES-256-GCM under key (must be 32 bytes).
// The returned bytes are version | nonce | ciphertext+tag and are safe to
// write directly to a file.
func EncryptBytes(key, plaintext []byte) ([]byte, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("crypto: key length is %d, want 32", len(key))
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("aes.NewCipher: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("cipher.NewGCM: %w", err)
	}
	nonce := make([]byte, gcmNonceLen)
	if _, err := rand.Read(nonce); err != nil {
		return nil, fmt.Errorf("rand: %w", err)
	}
	out := make([]byte, 0, headerLen+len(plaintext)+gcm.Overhead())
	out = append(out, envelopeVersion)
	out = append(out, nonce...)
	out = gcm.Seal(out, nonce, plaintext, nil)
	return out, nil
}

// DecryptBytes reverses EncryptBytes. Returns an error on truncation, version
// mismatch, or auth-tag failure.
func DecryptBytes(key, ciphertext []byte) ([]byte, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("crypto: key length is %d, want 32", len(key))
	}
	if len(ciphertext) < headerLen {
		return nil, fmt.Errorf("crypto: ciphertext too short (%d bytes)", len(ciphertext))
	}
	if ciphertext[0] != envelopeVersion {
		return nil, fmt.Errorf("crypto: unsupported envelope version %d", ciphertext[0])
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("aes.NewCipher: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("cipher.NewGCM: %w", err)
	}
	nonce := ciphertext[1:headerLen]
	sealed := ciphertext[headerLen:]
	plain, err := gcm.Open(nil, nonce, sealed, nil)
	if err != nil {
		return nil, fmt.Errorf("gcm.Open: %w", err)
	}
	return plain, nil
}
