// Package fsutil holds filesystem helpers shared across chiton packages.
package fsutil

import (
	"fmt"
	"io/fs"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// ResolveWithin returns p (joined onto root when relative) as a clean absolute path, or an error when its
// symlink-free form lies outside root's. It keeps p's spelling, so file URIs match those a build over root records.
func ResolveWithin(root, p string) (string, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	realRoot, err := filepath.EvalSymlinks(absRoot)
	if err != nil {
		return "", err
	}
	abs := p
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(absRoot, p)
	}
	abs = filepath.Clean(abs)
	real, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(realRoot, real)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("%s is outside the project root %s", p, absRoot)
	}
	return abs, nil
}

// WriteFileAtomic is os.WriteFile via a unique temp file, fsynced then renamed over path, so no crash or
// concurrent writer leaves path half-written. The file gets perm minus the umask. path's directory must exist.
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
	// Without the sync, a power loss after the rename can leave path pointing at an empty file.
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return rename(tmpName, path)
}

// rename is os.Rename, retried for about a second on Windows, where replacing a file that another process
// (a reader, an antivirus scan) briefly holds open fails with "Access is denied".
func rename(from, to string) error {
	err := os.Rename(from, to)
	for i := 1; err != nil && runtime.GOOS == "windows" && i <= 10; i++ {
		time.Sleep(time.Duration(i) * 20 * time.Millisecond)
		err = os.Rename(from, to)
	}
	return err
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
