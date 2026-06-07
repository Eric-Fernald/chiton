package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"time"

	"github.com/spf13/cobra"
)

// serveUnix serves the MCP HTTP handler (SSE transport) over a Unix domain
// socket. This is the lowest-latency shared-daemon transport for Linux/macOS
// power users — no TCP port to manage, and filesystem permissions gate access.
//
// A stale socket from a previous unclean shutdown is removed before binding;
// Ctrl-C triggers a graceful HTTP shutdown and removes the socket file.
func serveUnix(cmd *cobra.Command, handler http.Handler, path string) error {
	// AF_UNIX support and socket-file cleanup semantics are unreliable across
	// Windows versions; the flag is documented as Linux/macOS. Fail fast with a
	// clear message instead of an opaque listen error.
	if runtime.GOOS == "windows" {
		return fmt.Errorf("the --unix transport is supported on Linux/macOS; on Windows use --http or --stdio")
	}

	// Remove a stale socket left by a hard kill so the bind doesn't fail with
	// "address already in use" — but ONLY if it is actually a socket. The path
	// is user-supplied via --unix; a typo pointing at a real file must not cause
	// us to delete that file. Lstat so a symlink is inspected, not its target.
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

	srv := &http.Server{Handler: handler}

	ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()

	fmt.Fprintf(cmd.OutOrStdout(), "serve: listening on unix:%s (SSE)\n", path)
	if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
