// Package fsutil holds filesystem helpers shared across chiton packages.
package fsutil

import (
	"io/fs"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
)

// WriteFileAtomic is os.WriteFile via a unique temp file renamed over path, so a crash or concurrent writer never
// leaves path half-written; the file gets perm minus the umask. path's directory must exist.
func WriteFileAtomic(path string, data []byte, perm os.FileMode) error {
	tmp, err := createTemp(filepath.Dir(path), perm)
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op once the rename below succeeds
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// createTemp is os.CreateTemp(dir, ".chiton-*.tmp") but creates the file with perm (umask applied), not 0600.
func createTemp(dir string, perm os.FileMode) (*os.File, error) {
	for range 100 {
		name := filepath.Join(dir, ".chiton-"+strconv.FormatUint(rand.Uint64(), 36)+".tmp")
		f, err := os.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_EXCL, perm)
		if !os.IsExist(err) {
			return f, err
		}
	}
	return nil, &fs.PathError{Op: "createtemp", Path: filepath.Join(dir, ".chiton-*.tmp"), Err: fs.ErrExist}
}
