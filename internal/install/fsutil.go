package install

import (
	"os"
	"path/filepath"
)

// writeFileAtomic writes data to path atomically: it writes a temp file in the
// same directory, closes it, then renames it over path. A crash, disk-full, or
// partial write can therefore never leave a user's config truncated or
// half-written — the rename either fully replaces the file or doesn't happen.
// os.Rename replaces an existing destination on both POSIX and Windows.
//
// The temp file is created in path's directory (not a global temp dir) so the
// rename stays on one filesystem; callers must ensure that directory exists
// (every install helper MkdirAll's it first).
func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".chiton-*.tmp")
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
	if err := os.Chmod(tmpName, perm); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}
