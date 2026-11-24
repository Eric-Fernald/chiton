//go:build !windows && !plan9 && !js && !wasip1 && !aix

package badger

import (
	"errors"
	"os"

	"golang.org/x/sys/unix"
)

// holdReaderLock is false: a Unix reader keeps its mapped snapshot while a writer opens, so it holds the
// shared lock only during its own open, which keeps a writer from starting mid-open.
const holdReaderLock = false

func lockFile(f *os.File, exclusive bool) error {
	how := unix.LOCK_SH | unix.LOCK_NB
	if exclusive {
		how = unix.LOCK_EX | unix.LOCK_NB
	}
	err := unix.Flock(int(f.Fd()), how)
	if errors.Is(err, unix.EWOULDBLOCK) {
		return errLocked
	}
	return err
}

func unlockFile(f *os.File) error { return unix.Flock(int(f.Fd()), unix.LOCK_UN) }
