package vector

import (
	"bytes"
	"errors"
	"io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"
)

func itemSeq(items []Item) func(yield func(string, []float32) bool) {
	return func(yield func(string, []float32) bool) {
		for _, it := range items {
			if !yield(it.ID, it.Vec) {
				return
			}
		}
	}
}

func lookupIn(items []Item) func(string) ([]float32, bool) {
	byID := map[string][]float32{}
	for _, it := range items {
		byID[it.ID] = it.Vec
	}
	return func(id string) ([]float32, bool) { v, ok := byID[id]; return v, ok }
}

func TestSaveLoadHNSW(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	items := newEmbeddingLike(rng, 32, 20).items(3000)
	cfg := HNSWConfig{Seed: 1}
	h := NewHNSW(cfg)
	h.Add(items)
	fp := Fingerprint(cfg, itemSeq(items))
	path := filepath.Join(t.TempDir(), "hnsw.idx")
	if err := SaveHNSW(path, h, fp, nil); err != nil {
		t.Fatal(err)
	}

	got, err := LoadHNSW(path, HNSWConfig{EfSearch: 40}, fp, nil, lookupIn(items))
	if err != nil {
		t.Fatal(err)
	}
	if got.Len() != h.Len() || got.Dim() != 32 || got.g.EfSearch != 40 {
		t.Fatalf("Len %d Dim %d EfSearch %d; want %d, 32, and the configured 40", got.Len(), got.Dim(), got.g.EfSearch, h.Len())
	}
	got.g.EfSearch = h.g.EfSearch
	for _, q := range newEmbeddingLike(rand.New(rand.NewSource(13)), 32, 20).queries(20) {
		if a, b := h.Search(q, 8), got.Search(q, 8); !reflect.DeepEqual(a, b) {
			t.Fatalf("loaded index searches differently:\n%v\n%v", a, b)
		}
	}

	other := fp
	other[0] ^= 1
	if _, err := LoadHNSW(path, cfg, other, nil, lookupIn(items)); !errors.Is(err, ErrStale) {
		t.Errorf("load with another fingerprint: err = %v, want ErrStale", err)
	}
	if _, err := LoadHNSW(filepath.Join(t.TempDir(), "none.idx"), cfg, fp, nil, lookupIn(items)); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("load of a missing file: err = %v, want fs.ErrNotExist", err)
	}
	if _, err := LoadHNSW(path, cfg, fp, nil, lookupIn(items[1:])); err == nil {
		t.Error("load with a vector missing succeeded")
	}
}

func TestSaveHNSWEncrypted(t *testing.T) {
	items := randomItems(rand.New(rand.NewSource(14)), 200, 8)
	h := NewHNSW(HNSWConfig{Seed: 1})
	h.Add(items)
	fp := Fingerprint(HNSWConfig{Seed: 1}, itemSeq(items))
	key := bytes.Repeat([]byte{7}, 32)
	path := filepath.Join(t.TempDir(), "hnsw.idx")
	if err := SaveHNSW(path, h, fp, key); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(data, []byte(sidecarMagic)) || bytes.Contains(data, []byte(items[0].ID)) {
		t.Fatal("encrypted index holds plaintext")
	}
	if got, err := LoadHNSW(path, HNSWConfig{}, fp, key, lookupIn(items)); err != nil || got.Len() != 200 {
		t.Fatalf("load with the key: %v", err)
	}
	for name, k := range map[string][]byte{"no key": nil, "wrong key": bytes.Repeat([]byte{8}, 32)} {
		if _, err := LoadHNSW(path, HNSWConfig{}, fp, k, lookupIn(items)); err == nil || errors.Is(err, ErrStale) {
			t.Errorf("%s: err = %v, want a read failure", name, err)
		}
	}
}

func TestFingerprintCoversInputs(t *testing.T) {
	items := randomItems(rand.New(rand.NewSource(15)), 50, 4)
	cfg := HNSWConfig{Dim: 4, Seed: 1}
	base := Fingerprint(cfg, itemSeq(items))
	if Fingerprint(cfg, itemSeq(slices.Clone(items))) != base {
		t.Fatal("same input, different fingerprint")
	}
	if Fingerprint(HNSWConfig{Dim: 4, Seed: 1, M: DefaultHNSWM, EfConstruction: DefaultHNSWEfConstruction}, itemSeq(items)) != base {
		t.Error("explicit defaults changed the fingerprint")
	}
	changed := func(name string, c HNSWConfig, its []Item) {
		if Fingerprint(c, itemSeq(its)) == base {
			t.Errorf("%s did not change the fingerprint", name)
		}
	}
	changed("M", HNSWConfig{Dim: 4, Seed: 1, M: 32}, items)
	changed("ef_construction", HNSWConfig{Dim: 4, Seed: 1, EfConstruction: 300}, items)
	changed("dim", HNSWConfig{Dim: 8, Seed: 1}, items)
	edited := slices.Clone(items)
	edited[7] = Item{ID: edited[7].ID, Vec: slices.Clone(edited[7].Vec)}
	edited[7].Vec[2] += 0.001
	changed("a vector component", cfg, edited)
	renamed := slices.Clone(items)
	renamed[3].ID += "x"
	changed("an ID", cfg, renamed)
	changed("a dropped embedding", cfg, items[1:])
}
