package main

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"fmt"
	"mime"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	mcpsdk "github.com/mark3labs/mcp-go/server"
	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/fsutil"
)

// mcpTokenEnv pins the HTTP transport's bearer token, e.g. for a long-lived shared-daemon client config.
const mcpTokenEnv = "CHITON_MCP_TOKEN"

// noCORSOrigin is a reserved .invalid origin no browser can send. Listing it keeps mcp-go in explicit-CORS mode,
// which never emits "Access-Control-Allow-Origin: *", while allowing no real origin.
const noCORSOrigin = "https://chiton.invalid"

// maxMessageBytes caps one JSON-RPC POST body, which mcp-go decodes without a limit.
const maxMessageBytes = 8 << 20

// serveTokenPath is where the HTTP transport writes its per-start bearer token, beside the graph store.
func serveTokenPath(graphPath string) string {
	return filepath.Join(filepath.Dir(graphPath), "mcp-token")
}

// serveToken returns $CHITON_MCP_TOKEN, or a fresh random token it writes 0600 to path (written reports which).
func serveToken(path string) (token string, written bool, err error) {
	if t := strings.TrimSpace(os.Getenv(mcpTokenEnv)); t != "" {
		if len(t) < 16 {
			return "", false, fmt.Errorf("%s must be at least 16 characters", mcpTokenEnv)
		}
		return t, false, nil
	}
	token = rand.Text()
	if err := fsutil.WriteFileAtomic(path, []byte(token+"\n"), 0o600); err != nil {
		return "", false, fmt.Errorf("write MCP token %s: %w", path, err)
	}
	return token, true, nil
}

// guardMCPHTTP admits only requests with the bearer token, from no browser origin or an allowlisted one, whose
// POST bodies are declared JSON. A text/plain POST needs no CORS preflight, so the guard must refuse it itself.
func guardMCPHTTP(next http.Handler, token string, allowedOrigins []string) http.Handler {
	want := []byte("Bearer " + token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" && !slices.Contains(allowedOrigins, origin) {
			http.Error(w, "forbidden: origin not allowed", http.StatusForbidden)
			return
		}
		// A preflight never carries credentials, so an allowlisted one goes straight to mcp-go's CORS handling.
		if r.Method == http.MethodOptions && origin != "" {
			next.ServeHTTP(w, r)
			return
		}
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), want) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="chiton"`)
			http.Error(w, "unauthorized: missing or wrong bearer token", http.StatusUnauthorized)
			return
		}
		if r.Method == http.MethodPost {
			if mt, _, err := mime.ParseMediaType(r.Header.Get("Content-Type")); err != nil || mt != "application/json" {
				http.Error(w, "unsupported media type: want application/json", http.StatusUnsupportedMediaType)
				return
			}
			r.Body = http.MaxBytesReader(w, r.Body, maxMessageBytes)
		}
		next.ServeHTTP(w, r)
	})
}

// newMCPHTTPServer builds the guarded SSE transport. mcp-go's CORS allows exactly allowedOrigins (none by
// default), and its DNS-rebinding Host check stays on.
func newMCPHTTPServer(ctx context.Context, s *mcpsdk.MCPServer, addr, token string, allowedOrigins []string) (*http.Server, *mcpsdk.SSEServer) {
	cors := allowedOrigins
	if len(cors) == 0 {
		cors = []string{noCORSOrigin}
	}
	sse := mcpsdk.NewSSEServer(s, mcpsdk.WithSSECORS(mcpsdk.WithCORSAllowedOrigins(cors...)))
	return &http.Server{
		Addr:              addr,
		Handler:           guardMCPHTTP(sse, token, allowedOrigins),
		ReadHeaderTimeout: 10 * time.Second,
		BaseContext:       baseContext(ctx),
	}, sse
}

// serveHTTP serves MCP over HTTP+SSE on addr until the command's context is cancelled.
func serveHTTP(cmd *cobra.Command, s *mcpsdk.MCPServer, addr, tokenPath string, allowedOrigins []string) error {
	if slices.Contains(allowedOrigins, "*") {
		return fmt.Errorf("--allow-origin '*' is not supported; list each origin")
	}
	// Bind before touching the token file, so a failed bind never clobbers a running server's token.
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("listen on %s: %w", addr, err)
	}
	token, written, err := serveToken(tokenPath)
	if err != nil {
		ln.Close()
		return err
	}
	if written {
		defer os.Remove(tokenPath)
		fmt.Fprintf(cmd.ErrOrStderr(), "serve: bearer token written to %s — clients send 'Authorization: Bearer <token>'\n", tokenPath)
	}
	srv, sse := newMCPHTTPServer(cmd.Context(), s, addr, token, allowedOrigins)
	fmt.Fprintf(cmd.OutOrStdout(), "serve: listening on %s (SSE, bearer-token auth)\n", addr)
	return serveUntilDone(cmd.Context(), func() error { return srv.Serve(ln) }, func(ctx context.Context) error {
		sse.CloseSessions() // as SSEServer.Shutdown does, so open SSE streams don't block shutdown
		return srv.Shutdown(ctx)
	})
}

// transportFor picks serve's transport: --stdio wins, then --unix, then an explicitly set --http. The default is
// stdio, so a bare 'chiton serve' opens no socket.
func transportFor(stdio bool, unixPath string, httpSet bool) string {
	switch {
	case stdio:
		return "stdio"
	case unixPath != "":
		return "unix"
	case httpSet:
		return "http"
	default:
		return "stdio"
	}
}
