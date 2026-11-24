package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestWarnIfNonLoopback(t *testing.T) {
	cases := []struct {
		addr string
		warn bool
	}{
		{":8444", true},
		{"0.0.0.0:7474", true},
		{"[::]:8445", true},
		{"127.0.0.1:7474", false},
		{"localhost:7474", false},
		{"[::1]:7474", false},
		{"192.168.1.10:8080", false}, // deliberate LAN bind; not a wildcard
		{"/tmp/chiton.sock", false},  // not host:port at all
	}

	for _, tc := range cases {
		var buf bytes.Buffer
		warnIfNonLoopback(&buf, tc.addr, "the test listener")
		got := buf.Len() > 0
		if got != tc.warn {
			t.Errorf("warnIfNonLoopback(%q): warned=%v, want %v (output: %q)", tc.addr, got, tc.warn, buf.String())
		}
		if got && !strings.Contains(buf.String(), "ALL network interfaces") {
			t.Errorf("warnIfNonLoopback(%q): warning text is not recognisable: %q", tc.addr, buf.String())
		}
	}
}

func TestWarnIfNonLoopbackSuggestsThePort(t *testing.T) {
	var buf bytes.Buffer
	warnIfNonLoopback(&buf, ":8444", "the Teams webhook receiver")
	if !strings.Contains(buf.String(), "127.0.0.1:8444") {
		t.Errorf("warning should suggest the same port on loopback, got: %q", buf.String())
	}
}
