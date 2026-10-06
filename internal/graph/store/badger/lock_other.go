//go:build plan9 || js || wasip1 || aix

package badger

import "os"

// holdReaderLock is false and locking is a no-op where chiton has no advisory file lock.
const holdReaderLock = false

func lockFile(*os.File, bool) error { return nil }

func unlockFile(*os.File) error { return nil }
