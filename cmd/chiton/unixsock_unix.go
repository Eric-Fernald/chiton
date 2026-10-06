//go:build unix

package main

import (
	"net"
	"os"
	"syscall"
)

// listenUnixPrivate binds a Unix socket only its owner can connect to. The umask covers the moment before the chmod,
// and narrowing it can only make files other goroutines create in that moment more private.
func listenUnixPrivate(path string) (net.Listener, error) {
	old := syscall.Umask(0o177)
	ln, err := net.Listen("unix", path)
	syscall.Umask(old)
	if err != nil {
		return nil, err
	}
	if err := os.Chmod(path, 0o600); err != nil {
		ln.Close()
		return nil, err
	}
	return ln, nil
}
