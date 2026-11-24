package badger

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strings"
	"unicode/utf8"

	badgerdb "github.com/dgraph-io/badger/v4"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// indexed holds the node fields the secondary indexes key on, decoded without the rest of the record.
type indexed struct {
	Kind      model.NodeKind `json:"kind"`
	SourceURI string         `json:"source_uri"`
}

// indexedOf returns n's indexed fields as its JSON record will store them.
func indexedOf(n model.Node) indexed {
	return indexed{Kind: model.NodeKind(storedString(string(n.Kind))), SourceURI: storedString(n.SourceURI)}
}

// storedString returns s as it reads back from JSON: encoding/json turns each invalid UTF-8 byte into U+FFFD.
func storedString(s string) string {
	if utf8.ValidString(s) {
		return s
	}
	var b strings.Builder
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError && size == 1 {
			b.WriteRune(utf8.RuneError)
		} else {
			b.WriteString(s[i : i+size])
		}
		i += size
	}
	return b.String()
}

// uriPrefix is the index prefix for a source_uri. Hashing keeps keys short and free of the \x00 separator.
func uriPrefix(uri string) string {
	sum := sha256.Sum256([]byte(uri))
	return pfxURI + hex.EncodeToString(sum[:16]) + "\x00"
}

func uriKey(uri, id string) []byte { return []byte(uriPrefix(uri) + id) }

// kindPrefix is the index prefix for a node kind, kept raw so CountByKind can read it back.
func kindPrefix(k model.NodeKind) string { return pfxKind + string(k) + "\x00" }

func kindKey(k model.NodeKind, id string) []byte { return []byte(kindPrefix(k) + id) }

// setIndex writes node id's index keys.
func setIndex(wb *badgerdb.WriteBatch, id string, x indexed) error {
	if err := wb.Set(uriKey(x.SourceURI, id), nil); err != nil {
		return err
	}
	return wb.Set(kindKey(x.Kind, id), nil)
}

// dropStaleIndex deletes the index keys prev gave node id that cur no longer shares.
func dropStaleIndex(wb *badgerdb.WriteBatch, id string, prev, cur indexed) error {
	if prev.SourceURI != cur.SourceURI {
		if err := wb.Delete(uriKey(prev.SourceURI, id)); err != nil {
			return err
		}
	}
	if prev.Kind != cur.Kind {
		return wb.Delete(kindKey(prev.Kind, id))
	}
	return nil
}

// eachKey calls fn, in key order, with the suffix of every key under pfx, reading keys only. The suffix is
// valid only during the call.
func eachKey(txn *badgerdb.Txn, pfx string, fn func(k []byte) error) error {
	it := prefixIterator(txn, pfx)
	defer it.Close()
	for it.Rewind(); it.Valid(); it.Next() {
		if err := fn(it.Item().Key()[len(pfx):]); err != nil {
			return err
		}
	}
	return nil
}

// keySuffixes returns the suffix of every key under pfx, in key order.
func keySuffixes(txn *badgerdb.Txn, pfx string) ([]string, error) {
	var out []string
	err := eachKey(txn, pfx, func(k []byte) error {
		out = append(out, string(k))
		return nil
	})
	return out, err
}

// candidateIDs returns, sorted, the node IDs the index admits for f's source_uri or else its kinds. ok is
// false when f constrains neither, which leaves a scan.
func candidateIDs(txn *badgerdb.Txn, f store.NodeFilter) (ids []string, ok bool, err error) {
	var prefixes []string
	switch {
	case f.SourceURI != "":
		prefixes = []string{uriPrefix(f.SourceURI)}
	case len(f.Kinds) > 0:
		for _, k := range f.Kinds {
			prefixes = append(prefixes, kindPrefix(k))
		}
	default:
		return nil, false, nil
	}
	for _, p := range prefixes {
		got, err := keySuffixes(txn, p)
		if err != nil {
			return nil, false, err
		}
		ids = append(ids, got...)
	}
	slices.Sort(ids)
	return slices.Compact(ids), true, nil
}

// reindex rebuilds both indexes from the node records. It is the v1→v2 migration and the repair for a store
// whose last writer didn't seal it.
func reindex(db *badgerdb.DB) error {
	return writeBatch(db, func(wb *badgerdb.WriteBatch) error {
		return db.View(func(txn *badgerdb.Txn) error {
			for _, pfx := range []string{pfxURI, pfxKind} {
				if err := eachKey(txn, pfx, func(k []byte) error {
					return wb.Delete(append([]byte(pfx), k...))
				}); err != nil {
					return err
				}
			}
			return scan(txn, pfxNode, func(k, v []byte) error {
				var x indexed
				if err := json.Unmarshal(v, &x); err != nil {
					return err
				}
				return setIndex(wb, string(k), x)
			})
		})
	})
}
