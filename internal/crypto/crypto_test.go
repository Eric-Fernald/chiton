package crypto

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestDeriveKey_Deterministic(t *testing.T) {
	kf, err := DefaultKeyFile()
	if err != nil {
		t.Fatal(err)
	}
	a, err := kf.DeriveKey("hunter2")
	if err != nil {
		t.Fatal(err)
	}
	b, err := kf.DeriveKey("hunter2")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, b) {
		t.Errorf("same passphrase+keyfile should produce same key")
	}
	if len(a) != 32 {
		t.Errorf("expected 32-byte key, got %d", len(a))
	}
}

func TestDeriveKey_PassphraseSensitive(t *testing.T) {
	kf, _ := DefaultKeyFile()
	a, _ := kf.DeriveKey("hunter2")
	b, _ := kf.DeriveKey("hunter3")
	if bytes.Equal(a, b) {
		t.Errorf("different passphrases produced same key")
	}
}

func TestKeyFile_SaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "keyfile.json")
	kf, _ := DefaultKeyFile()
	if err := kf.Save(path, false); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadKeyFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded == nil {
		t.Fatal("expected loaded keyfile, got nil")
	}
	if loaded.SaltHex != kf.SaltHex || loaded.MemoryKiB != kf.MemoryKiB {
		t.Errorf("round-trip mismatch")
	}

	// Re-save without force fails.
	if err := kf.Save(path, false); err == nil {
		t.Errorf("expected error overwriting keyfile without --force")
	}
	if err := kf.Save(path, true); err != nil {
		t.Errorf("expected force=true to overwrite: %v", err)
	}
}

// TestDeriveKey_RejectsBadParams verifies DeriveKey errors cleanly instead of panicking inside
// argon2 (iterations/parallelism < 1) or silently deriving a weak/short key.
func TestDeriveKey_RejectsBadParams(t *testing.T) {
	base := func() *KeyFile { kf, _ := DefaultKeyFile(); return kf }
	cases := map[string]func(*KeyFile){
		"zero iterations":  func(kf *KeyFile) { kf.Iterations = 0 },
		"zero parallelism": func(kf *KeyFile) { kf.Parallelism = 0 },
		"low memory":       func(kf *KeyFile) { kf.MemoryKiB = 8 },
		"non-32 keylen":    func(kf *KeyFile) { kf.KeyLength = 16 },
		"empty salt":       func(kf *KeyFile) { kf.SaltHex = "" },
		"short salt":       func(kf *KeyFile) { kf.SaltHex = "0011" },
		"bad kdf":          func(kf *KeyFile) { kf.KDF = "scrypt" },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			kf := base()
			mutate(kf)
			// Must not panic, and must return an error.
			if _, err := kf.DeriveKey("hunter2"); err == nil {
				t.Errorf("DeriveKey accepted invalid keyfile (%s)", name)
			}
		})
	}
}

// TestLoadKeyFile_RejectsTampered verifies a hand-edited/corrupt keyfile fails
// at load time instead of flowing into argon2 and panicking later.
func TestLoadKeyFile_RejectsTampered(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keyfile.json")
	tampered := `{"version":1,"kdf":"argon2id","salt_hex":"00112233445566778899aabbccddeeff","memory_kib":64,"iterations":0,"parallelism":0,"key_length":32,"cipher":"AES-256-GCM"}`
	if err := os.WriteFile(path, []byte(tampered), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadKeyFile(path); err == nil {
		t.Error("LoadKeyFile accepted a keyfile with iterations=0/parallelism=0")
	}
}

func TestLoadKeyFile_Missing(t *testing.T) {
	got, err := LoadKeyFile(filepath.Join(t.TempDir(), "absent.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got != nil {
		t.Errorf("expected nil on missing keyfile, got %v", got)
	}
}

func TestEncryptDecrypt_RoundTrip(t *testing.T) {
	key := bytes.Repeat([]byte{0x42}, 32)
	plain := []byte("the quick brown fox jumps over the lazy dog")
	ct, err := EncryptBytes(key, plain)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(ct[headerLen:], plain) {
		t.Errorf("ciphertext leaks plaintext")
	}
	got, err := DecryptBytes(key, ct)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, plain) {
		t.Errorf("round-trip mismatch: %q vs %q", got, plain)
	}
}

func TestDecrypt_TamperingDetected(t *testing.T) {
	key := bytes.Repeat([]byte{0x42}, 32)
	ct, err := EncryptBytes(key, []byte("payload"))
	if err != nil {
		t.Fatal(err)
	}
	// Flip a byte in the ciphertext body.
	ct[len(ct)-1] ^= 0xff
	if _, err := DecryptBytes(key, ct); err == nil {
		t.Errorf("expected auth-tag failure on tampered ciphertext")
	}
}

func TestDecrypt_WrongKey(t *testing.T) {
	keyA := bytes.Repeat([]byte{0x42}, 32)
	keyB := bytes.Repeat([]byte{0x43}, 32)
	ct, _ := EncryptBytes(keyA, []byte("payload"))
	if _, err := DecryptBytes(keyB, ct); err == nil {
		t.Errorf("expected decryption failure with wrong key")
	}
}

func TestGetPassphrase_FromEnv(t *testing.T) {
	t.Setenv(EnvVarPassphrase, "from-env")
	got, err := GetPassphrase("ignored", false)
	if err != nil {
		t.Fatal(err)
	}
	if got != "from-env" {
		t.Errorf("expected env passphrase, got %q", got)
	}
}

func TestGetPassphrase_NonInteractive(t *testing.T) {
	t.Setenv(EnvVarPassphrase, "")
	// Stdin in the test runner is not a terminal — should error.
	if _, err := GetPassphrase("prompt", false); err == nil {
		t.Errorf("expected ErrNoPassphrase in non-interactive context")
	}
}

func TestResolveKeyForGraph_NoKeyfile(t *testing.T) {
	graph := filepath.Join(t.TempDir(), "graph.kuzu")
	key, kf, err := ResolveKeyForGraph(graph)
	if err != nil {
		t.Fatal(err)
	}
	if key != nil || kf != nil {
		t.Errorf("expected nil key when no keyfile present")
	}
}

func TestResolveKeyForGraph_FindsKeyfileNextToGraph(t *testing.T) {
	dir := t.TempDir()
	graph := filepath.Join(dir, "graph.kuzu")
	if err := os.WriteFile(filepath.Join(dir, KeyfileName), mustMarshal(t), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(EnvVarPassphrase, "test-passphrase")
	key, kf, err := ResolveKeyForGraph(graph)
	if err != nil {
		t.Fatalf("ResolveKeyForGraph: %v", err)
	}
	if kf == nil {
		t.Fatal("expected keyfile to be found")
	}
	if len(key) != 32 {
		t.Errorf("expected 32-byte key, got %d", len(key))
	}
}

func mustMarshal(t *testing.T) []byte {
	t.Helper()
	kf, err := DefaultKeyFile()
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(saveTempKeyfile(t, kf))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func saveTempKeyfile(t *testing.T, kf *KeyFile) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "kf.json")
	if err := kf.Save(path, false); err != nil {
		t.Fatal(err)
	}
	return path
}
