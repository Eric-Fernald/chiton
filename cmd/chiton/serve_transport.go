package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"runtime"
	"time"

	mcpsdk "github.com/mark3labs/mcp-go/server"
	"github.com/spf13/cobra"
)

// serveUnix serves the MCP SSE transport over a Unix domain socket, a low-latency shared-daemon
// transport where filesystem permissions gate access. A stale socket is removed before binding.
func serveUnix(cmd *cobra.Command, sse *mcpsdk.SSEServer, path string) error {
	// AF_UNIX and socket-file cleanup are unreliable on Windows; fail fast instead of an opaque listen error.
	if runtime.GOOS == "windows" {
		return fmt.Errorf("the --unix transport is supported on Linux/macOS; on Windows use --http or --stdio")
	}

	// Remove a stale socket only if it IS a socket: --unix is user-supplied and a typo must not delete
	// a real file. Lstat so a symlink is inspected, not its target.
	if info, err := os.Lstat(path); err == nil {
		if info.Mode()&os.ModeSocket == 0 {
			return fmt.Errorf("--unix path %s already exists and is not a socket; refusing to overwrite it", path)
		}
		if err := os.Remove(path); err != nil {
			return fmt.Errorf("remove stale socket %s: %w", path, err)
		}
	}

	ln, err := net.Listen("unix", path)
	if err != nil {
		return fmt.Errorf("listen on unix socket %s: %w", path, err)
	}
	// os.Remove on shutdown; the listener also unlinks on Close for "unix".
	defer os.Remove(path)

	srv := &http.Server{Handler: sse}
	fmt.Fprintf(cmd.OutOrStdout(), "serve: listening on unix:%s (SSE)\n", path)
	return serveUntilDone(cmd.Context(), func() error { return srv.Serve(ln) }, func(ctx context.Context) error {
		sse.CloseSessions() // as SSEServer.Shutdown does, so open SSE streams don't stall the drain
		return srv.Shutdown(ctx)
	})
}

// baseContext derives every request context from ctx, so Ctrl-C cancels in-flight handler work (e.g. a
// network re-ingest) instead of letting it outlive the shutdown.
func baseContext(ctx context.Context) func(net.Listener) context.Context {
	return func(net.Listener) context.Context { return ctx }
}

// shutdownGrace is how long serveUntilDone drains gracefully before waiting on handlers without a deadline.
var shutdownGrace = 5 * time.Second

// serveUntilDone runs serve until it fails or ctx is cancelled (Ctrl-C). On cancel it returns only once every
// in-flight handler has finished, so the caller's deferred store Close never runs under one.
func serveUntilDone(ctx context.Context, serve func() error, shutdown func(context.Context) error) error {
	errCh := make(chan error, 1)
	go func() { errCh <- serve() }()
	select {
	case <-ctx.Done():
		graceCtx, cancel := context.WithTimeout(context.Background(), shutdownGrace)
		defer cancel()
		if shutdown(graceCtx) != nil {
			// Past the grace period: keep waiting, since returning would close the store under the handlers.
			_ = shutdown(context.Background())
		}
		return nil
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	}
}
