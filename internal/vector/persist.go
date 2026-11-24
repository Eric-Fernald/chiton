package vector

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"iter"
	"math"
	"os"
	"path/filepath"

	chitoncrypto "github.com/Eric-Fernald/chiton/internal/crypto"
	"github.com/Eric-Fernald/chiton/internal/fsutil"
)

// The saved index is sidecarMagic, sidecarVersion, a Fingerprint, then the graph's topology. Vectors are
// left out: they already live in the graph store, and LoadHNSW takes them back from there.
const (
	sidecarMagic   = "chiton-hnsw-index"
	sidecarVersion = 1
)

// ErrStale reports a saved index built from other embeddings or settings than the caller's.
var ErrStale = errors.New("vector: saved hnsw index is stale")

// SidecarPath returns where the saved HNSW index for a graph store lives: hnsw.idx beside the store, as
// cluster.json is.
func SidecarPath(graphPath string) string {
	return filepath.Join(filepath.Dir(graphPath), "hnsw.idx")
}

// Fingerprint identifies embeddings, in order, together with the settings that shape an HNSW graph built
// from them. A saved index is valid only for the fingerprint it was saved with.
func Fingerprint(cfg HNSWConfig, embs iter.Seq2[string, []float32]) [sha256.Size]byte {
	h := sha256.New()
	var num [binary.MaxVarintLen64]byte
	putUint := func(x uint64) { h.Write(num[:binary.PutUvarint(num[:], x)]) }
	putUint(sidecarVersion)
	putUint(uint64(cfg.Dim))
	putUint(uint64(cmpOr(cfg.M, DefaultHNSWM)))
	putUint(uint64(cmpOr(cfg.EfConstruction, DefaultHNSWEfConstruction)))
	putUint(uint64(cfg.Seed))
	var buf []byte
	for id, vec := range embs {
		putUint(uint64(len(id)))
		h.Write([]byte(id))
		putUint(uint64(len(vec)))
		buf = buf[:0]
		for _, f := range vec {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(f))
		}
		h.Write(buf)
	}
	var sum [sha256.Size]byte
	h.Sum(sum[:0])
	return sum
}

// SaveHNSW writes h's graph to path, replacing the file atomically. fp is the Fingerprint of the
// embeddings and settings h was built from. A non-empty storeKey encrypts the file with a key derived from
// it, since node IDs name the code they index.
func SaveHNSW(path string, h *HNSW, fp [sha256.Size]byte, storeKey []byte) error {
	var buf bytes.Buffer
	buf.WriteString(sidecarMagic)
	buf.WriteByte(sidecarVersion)
	buf.Write(fp[:])
	if _, err := h.g.WriteTopology(&buf); err != nil {
		return fmt.Errorf("encode hnsw index: %w", err)
	}
	data := buf.Bytes()
	if len(storeKey) > 0 {
		key, err := chitoncrypto.FileKey(storeKey)
		if err != nil {
			return err
		}
		if data, err = chitoncrypto.EncryptBytes(key, data); err != nil {
			return fmt.Errorf("encrypt hnsw index: %w", err)
		}
	}
	if err := fsutil.WriteFileAtomic(path, data, 0o600); err != nil {
		return fmt.Errorf("write hnsw index %s: %w", path, err)
	}
	return nil
}

// LoadHNSW reads the index SaveHNSW wrote to path, taking each node's raw vector from vec and keeping a
// normalized copy. cfg supplies the query-time settings. It returns an error wrapping ErrStale when the
// file was saved for another fingerprint, and one wrapping fs.ErrNotExist when there is no file.
func LoadHNSW(path string, cfg HNSWConfig, fp [sha256.Size]byte, storeKey []byte,
	vec func(id string) ([]float32, bool)) (*HNSW, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(storeKey) > 0 {
		key, err := chitoncrypto.FileKey(storeKey)
		if err != nil {
			return nil, err
		}
		if data, err = chitoncrypto.DecryptBytes(key, data); err != nil {
			return nil, fmt.Errorf("decrypt hnsw index %s: %w", path, err)
		}
	}
	header := len(sidecarMagic) + 1 + sha256.Size
	if len(data) < header || string(data[:len(sidecarMagic)]) != sidecarMagic {
		return nil, fmt.Errorf("%s is not a saved hnsw index", path)
	}
	if v := data[len(sidecarMagic)]; v != sidecarVersion {
		return nil, fmt.Errorf("%s: %w: format version %d", path, ErrStale, v)
	}
	if !bytes.Equal(data[len(sidecarMagic)+1:header], fp[:]) {
		return nil, fmt.Errorf("%s: %w: built from other embeddings or settings", path, ErrStale)
	}

	h := NewHNSW(cfg)
	_, err = h.g.ReadTopology(bytes.NewReader(data[header:]), func(id string) ([]float32, bool) {
		raw, ok := vec(id)
		if !ok {
			return nil, false
		}
		return h.prepare(raw)
	})
	if err != nil {
		return nil, fmt.Errorf("read hnsw index %s: %w", path, err)
	}
	h.g.EfSearch = cmpOr(cfg.EfSearch, DefaultHNSWEfSearch)
	h.dim = h.g.Dim()
	return h, nil
}
