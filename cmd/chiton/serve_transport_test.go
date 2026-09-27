package main

import (
	"context"
	"errors"
	"net"
	"net/http"
	"sync/atomic"
	"testing"
	"time"
)

// The caller's deferred store Close runs as soon as serveUntilDone returns, so it must not return
// on Ctrl-C until in-flight handlers (which may be writing to the store) have finished.
func TestServeUntilDoneWaitsForInFlightHandlers(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	started := make(chan struct{})
	var finished atomic.Bool
	srv := &http.Server{Handler: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		close(started)
		time.Sleep(200 * time.Millisecond)
		finished.Store(true)
	})}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- serveUntilDone(ctx, func() error { return srv.Serve(ln) }, srv.Shutdown) }()
	go func() {
		if resp, err := http.Get("http://" + ln.Addr().String()); err == nil {
			resp.Body.Close()
		}
	}()

	<-started
	cancel()
	if err := <-done; err != nil {
		t.Fatalf("serveUntilDone: %v", err)
	}
	if !finished.Load() {
		t.Fatal("serveUntilDone returned before the in-flight handler finished")
	}
}

// A handler that outlives the grace period must still finish before serveUntilDone returns.
func TestServeUntilDoneWaitsPastGracePeriod(t *testing.T) {
	old := shutdownGrace
	shutdownGrace = 20 * time.Millisecond
	defer func() { shutdownGrace = old }()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	started := make(chan struct{})
	var finished atomic.Bool
	srv := &http.Server{Handler: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		close(started)
		time.Sleep(300 * time.Millisecond)
		finished.Store(true)
	})}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- serveUntilDone(ctx, func() error { return srv.Serve(ln) }, srv.Shutdown) }()
	go func() {
		if resp, err := http.Get("http://" + ln.Addr().String()); err == nil {
			resp.Body.Close()
		}
	}()

	<-started
	cancel()
	if err := <-done; err != nil {
		t.Fatalf("serveUntilDone: %v", err)
	}
	if !finished.Load() {
		t.Fatal("serveUntilDone returned before a handler slower than the grace period finished")
	}
}

func TestServeUntilDoneReturnsServeErrors(t *testing.T) {
	boom := errors.New("boom")
	noShutdown := func(context.Context) error { return nil }
	if err := serveUntilDone(context.Background(), func() error { return boom }, noShutdown); !errors.Is(err, boom) {
		t.Errorf("serve error: got %v, want %v", err, boom)
	}
	if err := serveUntilDone(context.Background(), func() error { return http.ErrServerClosed }, noShutdown); err != nil {
		t.Errorf("ErrServerClosed: got %v, want nil", err)
	}
}
