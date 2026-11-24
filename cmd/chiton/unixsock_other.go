//go:build !unix

package main

import "net"

// listenUnixPrivate is only reached on unix, since serveUnix rejects Windows before calling it.
func listenUnixPrivate(path string) (net.Listener, error) { return net.Listen("unix", path) }
