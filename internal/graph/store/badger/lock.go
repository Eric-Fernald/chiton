package badger

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ErrStoreBusy reports that another chiton process holds the store in a mode that conflicts with this open.
var ErrStoreBusy = errors.New("graph store is in use by another chiton process")

// errLocked is lockFile's error when another handle holds a conflicting lock.
var errLocked = errors.New("store lock held")

// lockName is the advisory lock file in the store directory: writers lock it exclusive, readers shared.
const lockName = "chiton.lock"

// storeLock is a held lock on a store's chiton.lock. The OS drops it if the process dies.
type storeLock struct{ f *os.File }

// acquireLock locks dir's chiton.lock without blocking. A reader that can't create the file (a missing or
// read-only directory) proceeds unlocked and leaves the error to Badger.
func acquireLock(dir string, exclusive bool) (*storeLock, error) {
	f, err := os.OpenFile(filepath.Join(dir, lockName), os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		if !exclusive {
			return nil, nil
		}
		return nil, fmt.Errorf("open store lock: %w", err)
	}
	if err := lockFile(f, exclusive); err != nil {
		f.Close()
		if errors.Is(err, errLocked) {
			return nil, busyError(dir, exclusive)
		}
		return nil, fmt.Errorf("lock store: %w", err)
	}
	return &storeLock{f: f}, nil
}

// busyError names the kind of process that is in the way.
func busyError(dir string, writer bool) error {
	if writer {
		return fmt.Errorf("%w: %s is open in another process (a running 'chiton serve', 'sync watch', or other command?); stop it and retry",
			ErrStoreBusy, dir)
	}
	return fmt.Errorf("%w: %s is being written ('chiton sync watch', 'build', or 'serve --writable'?); retry when it finishes",
		ErrStoreBusy, dir)
}

// release unlocks and closes the lock file. It is nil-safe for readers that proceeded unlocked.
func (l *storeLock) release() error {
	if l == nil || l.f == nil {
		return nil
	}
	err := errors.Join(unlockFile(l.f), l.f.Close())
	l.f = nil
	return err
}

// hasWriteLog reports whether dir holds a Badger memtable log. A clean close deletes them, so one found under
// the lock means a writer crashed, and a read-only open can't replay it.
func hasWriteLog(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".mem") {
			return true
		}
	}
	return false
}
