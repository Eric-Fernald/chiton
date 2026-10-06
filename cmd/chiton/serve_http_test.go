package main

import (
	"bufio"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	mcpsdk "github.com/mark3labs/mcp-go/server"
)

const testMCPToken = "test-token-0123456789"

// newGuardedTestServer serves the real guarded SSE transport on a loopback httptest listener.
func newGuardedTestServer(t *testing.T, origins ...string) *httptest.Server {
	t.Helper()
	srv, sse := newMCPHTTPServer(context.Background(), mcpsdk.NewMCPServer("t", "0"), "", testMCPToken, origins)
	if srv.ReadHeaderTimeout <= 0 {
		t.Fatal("HTTP transport has no ReadHeaderTimeout")
	}
	ts := httptest.NewServer(srv.Handler)
	t.Cleanup(func() {
		sse.CloseSessions()
		ts.Close()
	})
	return ts
}

// send issues one request with hdr set. The response body is closed when the test ends.
func send(t *testing.T, method, url, body string, hdr map[string]string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(method, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

// openSession opens an SSE stream and returns it with the message endpoint path from its first event.
func openSession(t *testing.T, ts *httptest.Server, hdr map[string]string) (*http.Response, string) {
	t.Helper()
	resp := send(t, http.MethodGet, ts.URL+"/sse", "", hdr)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /sse = %d, want 200", resp.StatusCode)
	}
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		if ep, ok := strings.CutPrefix(sc.Text(), "data: "); ok {
			return resp, strings.TrimSpace(ep)
		}
	}
	t.Fatal("no endpoint event on /sse")
	return nil, ""
}

const initializeMsg = `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}`

func TestMCPHTTPRequiresBearerToken(t *testing.T) {
	ts := newGuardedTestServer(t)
	for _, hdr := range []map[string]string{nil, {"Authorization": "Bearer wrong-token-0123456789"}} {
		if resp := send(t, http.MethodGet, ts.URL+"/sse", "", hdr); resp.StatusCode != http.StatusUnauthorized {
			t.Errorf("GET /sse with %v = %d, want 401", hdr, resp.StatusCode)
		}
	}
}

// A hostile page can POST text/plain to 127.0.0.1 without a CORS preflight. The guard must refuse that by
// Origin and by Content-Type, while a real JSON client still gets through.
func TestMCPHTTPRefusesBrowserStylePosts(t *testing.T) {
	ts := newGuardedTestServer(t)
	_, endpoint := openSession(t, ts, map[string]string{"Authorization": "Bearer " + testMCPToken})
	for _, tc := range []struct {
		origin, contentType string
		want                int
	}{
		{"https://evil.example", "text/plain", http.StatusForbidden},
		{"", "text/plain", http.StatusUnsupportedMediaType},
		{"", "application/json", http.StatusAccepted},
	} {
		hdr := map[string]string{"Authorization": "Bearer " + testMCPToken, "Content-Type": tc.contentType}
		if tc.origin != "" {
			hdr["Origin"] = tc.origin
		}
		if got := send(t, http.MethodPost, ts.URL+endpoint, initializeMsg, hdr).StatusCode; got != tc.want {
			t.Errorf("POST origin=%q type=%q = %d, want %d", tc.origin, tc.contentType, got, tc.want)
		}
	}
}

func TestMCPHTTPNeverSendsWildcardCORS(t *testing.T) {
	resp, _ := openSession(t, newGuardedTestServer(t), map[string]string{"Authorization": "Bearer " + testMCPToken})
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "" {
		t.Errorf("Access-Control-Allow-Origin = %q with no allowlist, want none", got)
	}
	const dev = "http://localhost:5173"
	hdr := map[string]string{"Authorization": "Bearer " + testMCPToken, "Origin": dev}
	resp, _ = openSession(t, newGuardedTestServer(t, dev), hdr)
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != dev {
		t.Errorf("Access-Control-Allow-Origin = %q for an allowlisted origin, want %q", got, dev)
	}
}

func TestServeTokenWritesFileOrUsesEnv(t *testing.T) {
	t.Setenv(mcpTokenEnv, "")
	path := filepath.Join(t.TempDir(), "mcp-token")
	token, written, err := serveToken(path)
	if err != nil || !written || len(token) < 16 {
		t.Fatalf("serveToken = %q, %v, %v", token, written, err)
	}
	if b, err := os.ReadFile(path); err != nil || strings.TrimSpace(string(b)) != token {
		t.Fatalf("token file = %q, %v; want %q", b, err, token)
	}
	t.Setenv(mcpTokenEnv, "short")
	if _, _, err := serveToken(path); err == nil {
		t.Error("a CHITON_MCP_TOKEN under 16 characters was accepted")
	}
}
