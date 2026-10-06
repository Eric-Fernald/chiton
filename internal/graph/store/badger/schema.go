package badger

import (
	"errors"
	"fmt"
	"strconv"

	badgerdb "github.com/dgraph-io/badger/v4"
)

// schemaVersion is the key layout this chiton reads and writes. Bump it together with a migrations entry.
const schemaVersion = 1

// schemaFresh is storedSchema's answer for a store with no nodes, which needs no migration.
const schemaFresh = 0

// Metadata keys live under 'm', which no data prefix uses.
const (
	pfxMeta    = "m:"
	metaSchema = pfxMeta + "schema" // decimal layout version
)

// ErrSchemaTooNew reports a store written by a newer chiton, which this binary must not read or write.
var ErrSchemaTooNew = errors.New("graph store was written by a newer chiton")

// migrations[v] upgrades the layout from v to v+1. A step must be idempotent: an interrupted upgrade reruns
// the first step whose version stamp is missing.
var migrations = map[int]func(*badgerdb.DB) error{}

// prepareSchema checks the layout version on open. A writable store is stamped when fresh and migrated when
// old; an old read-only store reports errNeedsWriterPass.
func prepareSchema(db *badgerdb.DB, readOnly bool) error {
	v, err := storedSchema(db)
	if err != nil {
		return err
	}
	switch {
	case v > schemaVersion:
		return fmt.Errorf("%w (layout v%d; this chiton reads up to v%d): upgrade chiton", ErrSchemaTooNew, v, schemaVersion)
	case v == schemaVersion, v == schemaFresh && readOnly:
		return nil
	case readOnly:
		return fmt.Errorf("%w (layout v%d predates v%d)", errNeedsWriterPass, v, schemaVersion)
	case v == schemaFresh:
		return setMeta(db, metaSchema, strconv.Itoa(schemaVersion))
	default:
		return migrate(db, v, schemaVersion, migrations)
	}
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
