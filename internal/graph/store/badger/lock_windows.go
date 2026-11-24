//go:build windows

package badger

import (
	"errors"
	"os"

	"golang.org/x/sys/windows"
)

// holdReaderLock keeps a reader's shared lock until Close: Windows can't truncate or delete a file another
// process has mapped, so a writer opening beside a reader fails midway and strands a 128 MiB memtable log.
const holdReaderLock = true

func lockFile(f *os.File, exclusive bool) error {
	flags := uint32(windows.LOCKFILE_FAIL_IMMEDIATELY)
	if exclusive {
		flags |= windows.LOCKFILE_EXCLUSIVE_LOCK
	}
	err := windows.LockFileEx(windows.Handle(f.Fd()), flags, 0, 1, 0, new(windows.Overlapped))
	if errors.Is(err, windows.ERROR_LOCK_VIOLATION) {
		return errLocked
	}
	return err
}

func unlockFile(f *os.File) error {
	return windows.UnlockFileEx(windows.Handle(f.Fd()), 0, 1, 0, new(windows.Overlapped))
}
