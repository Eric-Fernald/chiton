package badger

import (
	"crypto/rand"
	"errors"
	"fmt"
	"strconv"

	badgerdb "github.com/dgraph-io/badger/v4"
)

// schemaVersion is the key layout this chiton reads and writes. Bump it together with a migrations entry. v2
// added the source_uri and kind indexes and the seal.
const schemaVersion = 2

// schemaFresh is storedSchema's answer for a store with no nodes, which needs no migration.
const schemaFresh = 0

// Metadata keys live under 'm', which no data prefix uses.
const (
	pfxMeta    = "m:"
	metaSchema = pfxMeta + "schema" // decimal layout version
	metaSeal   = pfxMeta + "seal"   // Badger version at the last clean close (see sealed)
	metaEpoch  = pfxMeta + "epoch"  // random store identity (see Store.Epoch)
)

// ErrSchemaTooNew reports a store written by a newer chiton, which this binary must not read or write.
var ErrSchemaTooNew = errors.New("graph store was written by a newer chiton")

// migrations[v] upgrades the layout from v to v+1. A step must be idempotent: an interrupted upgrade reruns
// the first step whose version stamp is missing.
var migrations = map[int]func(*badgerdb.DB) error{
	1: reindex,
}

// prepareSchema checks the layout version on open and that the indexes are intact. A writable store is
// stamped, migrated, or reindexed as needed; a read-only one reports errNeedsWriterPass instead.
func prepareSchema(db *badgerdb.DB, readOnly bool) error {
	v, err := storedSchema(db)
	if err != nil {
		return err
	}
	switch {
	case v > schemaVersion:
		return fmt.Errorf("%w (layout v%d; this chiton reads up to v%d): upgrade chiton", ErrSchemaTooNew, v, schemaVersion)
	case v == schemaFresh:
		if readOnly {
			return nil
		}
		return setMeta(db, metaSchema, strconv.Itoa(schemaVersion))
	case v < schemaVersion && readOnly:
		return fmt.Errorf("%w (layout v%d predates v%d)", errNeedsWriterPass, v, schemaVersion)
	case v < schemaVersion:
		return migrate(db, v, schemaVersion, migrations) // v1→v2 builds the indexes afresh
	}
	ok, err := sealed(db)
	switch {
	case err != nil || ok:
		return err
	case readOnly:
		return fmt.Errorf("%w (the last writer did not close cleanly)", errNeedsWriterPass)
	default:
		return reindex(db)
	}
}

// seal records the store's current Badger version. The seal's own commit is the next version, which sealed
// checks to prove nothing wrote after it without index upkeep (a crash, or an older chiton).
func seal(db *badgerdb.DB) error {
	return setMeta(db, metaSeal, strconv.FormatUint(db.MaxVersion(), 10))
}

// sealed reports whether the seal is still the store's newest commit.
func sealed(db *badgerdb.DB) (bool, error) {
	var raw string
	if err := db.View(func(txn *badgerdb.Txn) (err error) {
		raw, err = getMeta(txn, metaSeal)
		return err
	}); err != nil || raw == "" {
		return false, err
	}
	v, err := strconv.ParseUint(raw, 10, 64)
	return err == nil && db.MaxVersion() == v+1, nil
}

// storedSchema returns the layout stamped in m:schema, 1 for a store with nodes but no stamp (written before
// versioning), or schemaFresh for one with no nodes.
func storedSchema(db *badgerdb.DB) (int, error) {
	v := schemaFresh
	err := db.View(func(txn *badgerdb.Txn) error {
		raw, err := getMeta(txn, metaSchema)
		if err != nil {
			return err
		}
		if raw != "" {
			if v, err = strconv.Atoi(raw); err != nil {
				return fmt.Errorf("bad store layout stamp %q: %w", raw, err)
			}
			return nil
		}
		it := prefixIterator(txn, pfxNode)
		defer it.Close()
		if it.Rewind(); it.Valid() {
			v = 1
		}
		return nil
	})
	return v, err
}

// migrate upgrades db's layout from version from to version to, stamping each step as it completes.
func migrate(db *badgerdb.DB, from, to int, steps map[int]func(*badgerdb.DB) error) error {
	for v := from; v < to; v++ {
		step, ok := steps[v]
		if !ok {
			return fmt.Errorf("no migration from store layout v%d", v)
		}
		if err := step(db); err != nil {
			return fmt.Errorf("migrate store layout v%d to v%d: %w", v, v+1, err)
		}
		if err := setMeta(db, metaSchema, strconv.Itoa(v+1)); err != nil {
			return err
		}
	}
	return nil
}

// loadEpoch returns the store's random identity, minting one on a writable store that has none.
func loadEpoch(db *badgerdb.DB, readOnly bool) (string, error) {
	var epoch string
	if err := db.View(func(txn *badgerdb.Txn) (err error) {
		epoch, err = getMeta(txn, metaEpoch)
		return err
	}); err != nil || epoch != "" || readOnly {
		return epoch, err
	}
	epoch = rand.Text()
	return epoch, setMeta(db, metaEpoch, epoch)
}

// getMeta returns the metadata value under key, or "" when it is absent.
func getMeta(txn *badgerdb.Txn, key string) (string, error) {
	item, err := txn.Get([]byte(key))
	if errors.Is(err, badgerdb.ErrKeyNotFound) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	v, err := item.ValueCopy(nil)
	return string(v), err
}

// setMeta stores a metadata value in its own transaction.
func setMeta(db *badgerdb.DB, key, val string) error {
	return db.Update(func(txn *badgerdb.Txn) error { return txn.Set([]byte(key), []byte(val)) })
}
