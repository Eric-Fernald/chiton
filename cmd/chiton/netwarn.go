package main

import (
	"fmt"
	"io"
	"net"
)

// warnIfNonLoopback warns without blocking when addr is a literal wildcard bind. Named hosts aren't resolved.
// what is a noun phrase completing "…listens on ALL network interfaces, exposing <what>".
func warnIfNonLoopback(w io.Writer, addr, what string) {
	if !isWildcardAddr(addr) {
		return
	}
	fmt.Fprintf(w, "WARNING: %s listens on ALL network interfaces, exposing %s. "+
		"Bind to loopback instead, e.g. --addr 127.0.0.1%s.\n", addr, what, portSuffix(addr))
}

// isWildcardAddr reports whether a listen address binds every interface: an
// empty host (":8444"), or the unspecified IPv4/IPv6 addresses.
func isWildcardAddr(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		// Not host:port at all (e.g. a bare port or a Unix path); nothing to say.
		return false
	}
	if host == "" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsUnspecified()
}

// portSuffix returns ":<port>" for a host:port address, or "" when the port
// cannot be determined — used only to make the suggested fix concrete.
func portSuffix(addr string) string {
	_, port, err := net.SplitHostPort(addr)
	if err != nil || port == "" {
		return ""
	}
	return ":" + port
}
