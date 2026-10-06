package crypto

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/hkdf"
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"fmt"
)

// FileKey derives the key for files chiton seals beside the graph store from the store's own key, so the
// store and those files never share a key.
func FileKey(storeKey []byte) ([]byte, error) {
	if len(storeKey) == 0 {
		return nil, fmt.Errorf("crypto: empty store key")
	}
	return hkdf.Key(sha256.New, storeKey, nil, "chiton file key v1", 32)
}

// envelopeVersion leads EncryptBytes' on-disk layout: version (1 byte) | GCM nonce (12) |
// ciphertext+tag. It lets the format evolve without breaking existing files.
const envelopeVersion byte = 1

const (
	gcmNonceLen = 12
	headerLen   = 1 + gcmNonceLen
)

// EncryptBytes seals plaintext with AES-256-GCM under key, which must be 32 bytes, and returns
// version | nonce | ciphertext+tag, ready to write to a file.
func EncryptBytes(key, plaintext []byte) ([]byte, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("crypto: key length is %d, want 32", len(key))
	}
	gcm, err := newGCM(key)
	if err != nil {
		return nil, err
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
	gcm, err := newGCM(key)
	if err != nil {
		return nil, err
	}
	nonce := ciphertext[1:headerLen]
	sealed := ciphertext[headerLen:]
	plain, err := gcm.Open(nil, nonce, sealed, nil)
	if err != nil {
		return nil, fmt.Errorf("gcm.Open: %w", err)
	}
	return plain, nil
}

func newGCM(key []byte) (cipher.AEAD, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("aes.NewCipher: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("cipher.NewGCM: %w", err)
	}
	return gcm, nil
}

// sealMagic prefixes Seal's output so Unseal can tell a sealed file from one written before encryption was on.
var sealMagic = []byte("CHITON-SEALED-1\x00")

// ErrSealedNoKey means a file is sealed but no keyfile was found to open it.
var ErrSealedNoKey = errors.New("file is encrypted but no keyfile was found for this store")

// Seal returns data unchanged for an empty key (a plaintext store), else sealMagic followed by EncryptBytes'
// envelope. key is a FileKey, never the store key itself.
func Seal(key, data []byte) ([]byte, error) {
	if len(key) == 0 {
		return data, nil
	}
	ct, err := EncryptBytes(key, data)
	if err != nil {
		return nil, err
	}
	out := make([]byte, 0, len(sealMagic)+len(ct))
	return append(append(out, sealMagic...), ct...), nil
}

// Unseal reverses Seal. Unsealed data passes through even with a key, so a file written before encryption stays
// readable until its next write. Sealed data without a key is ErrSealedNoKey.
func Unseal(key, data []byte) ([]byte, error) {
	if !bytes.HasPrefix(data, sealMagic) {
		return data, nil
	}
	if len(key) == 0 {
		return nil, ErrSealedNoKey
	}
	plain, err := DecryptBytes(key, data[len(sealMagic):])
	if err != nil {
		return nil, fmt.Errorf("decrypt (wrong passphrase or corrupt file): %w", err)
	}
	return plain, nil
}
