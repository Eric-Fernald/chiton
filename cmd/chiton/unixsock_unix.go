//go:build unix

package main

import (
	"net"
	"os"
	"path/filepath"
)

// maxSockPath is a conservative sun_path limit (macOS allows 104 bytes including the NUL).
const maxSockPath = 103

// listenUnixPrivate binds an owner-only socket in a fresh 0700 dir, chmods it 0600 and renames it into place, with
// no umask change. The caller removes path on shutdown, since Close would unlink only the old temp name.
func listenUnixPrivate(path string) (net.Listener, error) {
	dir, err := os.MkdirTemp(filepath.Dir(path), ".chiton-sock-*")
	if err != nil {
		return listenUnixChmod(path)
	}
	defer os.RemoveAll(dir)
	tmp := filepath.Join(dir, "s")
	if len(tmp) > maxSockPath {
		// The temp path would overflow sun_path, so bind at path directly and chmod at once.
		return listenUnixChmod(path)
	}
	ln, err := net.Listen("unix", tmp)
	if err != nil {
		return nil, err
	}
	if err := os.Chmod(tmp, 0o600); err != nil {
		ln.Close()
		return nil, err
	}
	if err := os.Rename(tmp, path); err != nil {
		ln.Close()
		return nil, err
	}
	return ln, nil
}

// listenUnixChmod is the fallback: bind at path, then chmod. A brief window follows the umask.
func listenUnixChmod(path string) (net.Listener, error) {
	ln, err := net.Listen("unix", path)
	if err != nil {
		return nil, err
	}
	if err := os.Chmod(path, 0o600); err != nil {
		ln.Close()
		return nil, err
	}
	return ln, nil
}
