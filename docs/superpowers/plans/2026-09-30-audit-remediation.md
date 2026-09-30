# Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the verified performance, reliability and security findings from the 2026-09-29 audit of chiton. The CI workflow change is out of scope.

**Architecture:** 45 tasks in six sections, grouped by code area: A MCP server and network, B Badger store, C ingest/build/sync, D analyzer/clustering/hooks, E config/crypto/first-run, F installer/sources/embeddings/reports. Each task is one test-first, reviewable commit. Sections are listed by area, but they must be *executed* in the cross-section order below, because several tasks share files (`internal/ingest/engine.go`, `internal/graph/store/badger/store.go`, `cmd/chiton/main.go`).

**Tech Stack:** Go (`go 1.25.5`, toolchain `go1.26.4`), BadgerDB v4.9.6, mcp-go v0.58.0, cobra/viper, fsnotify v1.10, modernc.org/sqlite.

**Spec:** No separate spec document. Each task's **Finding** line is its requirement, with `path:line` evidence re-verified against commit `3e0680d` on 2026-09-29/30. Sections A and D were applied end to end to scratch copies of the repo, where build, vet, gofmt and the full test suite pass and each new test fails on the unmodified code. Sections B, C, E and F were written against the code and partly probed, but not applied end to end, so expect small compile fixes there.

## Global Constraints

- Go language line `go 1.25.5`, toolchain `go1.26.4`. The default build stays pure Go (CGO off). The `msgraph` and `whisper_cgo` tags stay optional.
- No new Go module dependencies. Task F3 vendors a JavaScript asset, which is not a Go dependency.
- Before every commit, run `gofmt -l .` (the output must be empty), `go vet ./...` and `go test ./... -count=1` (all 45 packages must pass).
- Commands are written for PowerShell on Windows, the maintainer's machine, where the Bash tool returns no output.
- Comments are one or two lines, as set by the September 2026 cleanup sweep.
- Reuse the existing shared helpers instead of copying them:
  - `store.FindNode`, `PickByFile`, `NodesMatching`, `EmbeddingIDs`
  - `model.FileURI`, `ContentSHA`, `AttrString`
  - `fsutil.WriteFileAtomic`, `analyze.Lines`/`Cached`, `cluster.Full`
- Line numbers are as of `3e0680d`. Earlier tasks shift them, so find code by function name and use the numbers only as hints.
- `CLAUDE.md` is untracked and gitignored, so edit it locally but don't try to commit it. Canonical skill text lives in repo-root `skills/`.
- Ingested content (docs, transcripts, Slack, email, web pages) is untrusted data, and no task may treat it as instructions.
- Don't touch `.github/workflows/*` (the maintainer is keeping CI as it is).

## Decisions to confirm before starting

Each task implements the **recommended** option. If you choose the alternative, adjust only that task.

| Task | Decision | Recommended (planned) | Alternative |
|---|---|---|---|
| A1 | HTTP transport token | Required by default, generated each start, written 0600 | Opt-in via flag |
| A9 | Default `serve` transport | stdio; HTTP only with `--http <addr>` | Keep HTTP/SSE as default |
| A3 | `add_url` and HTTP(S)_PROXY | Ignore proxy env so the dial guard sees the real IP | Honour proxy with a weaker guard |
| A4 | PII in generic graph tools | Redacted by default; `--include-pii` opts out | Off unless asked; also decide whether `--include-pii` caps the meeting tools' per-call `include_pii` |
| A7 | Concept conflicts in `submit_extraction` | First writer wins; later submissions only link | Later submissions may raise confidence |
| B1 | Windows writer while a reader is open | Fail fast with a clear "in use" error | Follow-up: `serve` releases the store when idle |
| B2 | Read-only open of an older layout | Performs the one-time upgrade/repair itself | Refuse and tell the user to run `chiton sync once` |
| B6 | Pre-epoch `cache.json` | Discard (one full re-extract after upgrade) | Adopt once |
| C1 | Default ignore list | Includes `vendor`, `target`, `dist` | Leave those to `.chitonignore` |
| C2 | `build` and re-clustering | `build` skips re-cluster (run `chiton cluster`) | Re-cluster when `cluster.json` exists |
| C8 | Symlinks when indexing | Skip every symlink | Follow symlinks whose target is inside the repo |
| D2 | Pre-commit gate scope | Staged files only, FileScope rules, in memory | Whole-tree scan (brings back GraphScope rules) |
| D2 | Hook timeout | Fail open | Fail closed on timeout only |
| D6 | Region re-cluster limits | Stubs keep community/degree; unreachable new nodes wait for the next full `chiton cluster` | Full re-cluster on every batch |
| E5 | OS keychain for the passphrase | Defer (not planned here) | Implement now |
| E6 | Plaintext stores | One-line notice (`CHITON_PLAINTEXT_OK=1` silences it) | `build` refuses plaintext without an explicit opt-out marker |
| E8 | Old `--signing-secret` / `--client-state` flags | Fail with guidance pointing to `--*-file` / env | Keep working via `MarkDeprecated` |
| F1 | Mode for new MCP config files | 0600 everywhere, project scope included | 0600 only under `$HOME` |
| F2 | Browser URL redaction | On by default, plus an automatic one-time purge of old query-bearing nodes | Purge only on an explicit flag |
| F3 | vis-network delivery | Vendor pinned 9.1.9 via `go:embed` | Pinned CDN URL with SRI |
| F4 | Vectors from another model | `serve` warns and excludes them; legacy vectors re-embedded once | Disable `semantic_search`; `--adopt-legacy` flag |
| F5 | int8 vector quantization | Defer | Implement now |

## Execution order

This order satisfies every task's **Depends on** line and closes the worst exposure first.

| Phase | Tasks, in order | Why |
|---|---|---|
| 1. Close the holes | A1 → A9 → A2 → D1 → C1 → D2 → D3 | Browser-to-localhost graph access, file reads through MCP, commits blocked by store errors, quadratic sync analysis, collapsed clustering |
| 2. Store foundation | B1 → B2 → C3 → B3 → B5 | Windows wedge fix, schema versioning, edge-preserving re-sync, then the source_uri/kind indexes built on C3's write seams |
| 3. Ingest pipeline | C6 → C2 → C8 → C7 → C4 → C5 | Build runs through the engine (parallel, batched, prunes stale symbols), then symlinks, dedupe, path case, overflow |
| 4. Remaining security | A3 → A8 → A7 → A4 → E1 → E3 → E2 → E4 → E5 → E7 → E8 → F1 → F2 → F3 | SSRF, ref injection, extraction binding, PII, config trust, passphrase UX, file modes, URL secrets, CDN |
| 5. Remaining reliability and performance | F4 → F5 → F6 → A5 → A6 → B4 → B6 → B7 → D4 → D5 → D6 → D7 → E6 | Embedding staleness, webhooks, bounded MCP output, read-path speedups, cache epoch, vlog cap, determinism, region re-cluster, parallel betweenness, sealed sidecar |

## Cross-section integration notes

- **C3 before B3.** C3 routes all Badger node/edge writes, deletes and source_uri lookups through `putNodes`, `putEdges`, `deleteNode`, `deleteEdge` and `sourceIDs`. B3 puts its `u:`/`k:` index maintenance inside those functions (the note is repeated in B3).
- **`internal/ingest/engine.go` is edited by D1, C3, C6, C2, E2, B6, D6 and E6**, in that execution order. Each task shows code against today's layout, so merge into the current function instead of pasting over it:
  - D1 adds `ReadPriorFindings` before the delete in `SyncFile`. C3 must keep that call immediately before its `store.ReplaceSources`.
  - C2 adds `ingest.Config.NoRecluster`.
  - D6 replaces the whole-graph re-cluster with `cluster.BuildRegion`/`ApplyRegion`.
  - E6 adds `Config.SidecarKey`.
- **Hooks.** D2 makes the pre-commit gate store-free and fail-open, and removes the store parameter from `hooks.Run`. That supersedes the pre-commit concern in B1, so land D2 (phase 1) before B1 (phase 2).
- **Ignore policy.** C1 creates the leaf package `internal/walk` (`LoadPolicy`, `(*Policy).Skip`, `Files`). D2's `ignoredDir` delegates to it. It can't live in `internal/ingest`, because `ingest` imports `analyze`.
- **Cluster sidecar.** D6 owns its encoding, through `marshalSidecar` / `unmarshalSidecar`. E6 adds a `key []byte` parameter to `WriteSidecar`/`LoadSidecar` and seals and unseals around that seam.
- **Embeddings.** F4 changes only the `v:` value format: a tagged header, with legacy values still decoding. No B2/B3 migration may rewrite `v:` values. C3 already deletes the vector of a node whose content changed, and F4's text-hash check covers model changes and anything C3 misses. The two are compatible.
- **`KindCounter`.** B3 exports `store.KindCounter` and the helper `store.CountByKind(ctx, gs)`, which falls back to a full scan. A6's `graph_summary` consumes it.
- **Serve command.** A1/A9 own the HTTP/SSE path and the default transport (new `cmd/chiton/serve_http.go`). E7 owns the Unix-socket mode in `serve_transport.go`. E6 touches the serve sidecar load.
- **New exported helpers others may reuse:**
  - `fsutil.ResolveWithin` (A2), `retrieve.TrimToBudget` (A5), `git.ValidateRef` (A8)
  - `store.DeleteNodes` (F4), `model.IsExternal` (D4), `config.StateDir` (E2)

## Out of scope

- `.github/workflows/ci.yml` changes: run CI on PRs, add a Windows job, add `govulncheck`. You're keeping CI as it is.
- Pinning `release.yml`'s third-party action to a commit SHA. This is CI-adjacent and was left out with the above.
- Rule suppression (`chiton rules disable`, inline ignores): a new feature, not a fix.
- OS keychain for the passphrase (E5), int8 vector quantization (F5), and releasing the store from `serve` when idle (B1) are all listed as follow-ups.

## Task index

| ID | Title | Pri | Effort | Depends on |
|---|---|---|---|---|
| A1 | Authenticate the HTTP/SSE transport and refuse browser origins | P1 | M | — |
| A2 | Keep MCP tools from reading files outside the project | P1 | M | — |
| A3 | Vet the dialled address in add_url's SSRF guard | P2 | S | A2 |
| A4 | Redact people's identities in the generic graph tools | P2 | M | — |
| A5 | Bound get_node_neighborhood and emit compact JSON | P3 | S | A4 |
| A6 | Stop decoding the whole graph per graph_summary / meeting call | P3 | M | A4, B3 |
| A7 | Bind submit_extraction to an extraction task; stop overwriting concepts | P3 | S | — |
| A8 | Guard GitCommit's ref against option injection | P3 | S | — |
| A9 | Make stdio the default serve transport | P1 | S | A1 |
| B1 | Coordinate readers and writers with `chiton.lock`; self-repair a crashed writer's log | P1 | M | — |
| B2 | Store layout version and migration runner | P1 | M | B1 |
| B3 | source_uri and kind indexes, seal, `store.KindCounter` | P1 | L | B2, C3 |
| B4 | Label index for read-only `SearchNodes` | P3 | S | B2 |
| B5 | ShortestPath BFS over adjacency keys | P3 | S | — |
| B6 | Store epoch in `cache.json`; sync the store before saving the cache | P3 | M | B2, B4, C2 |
| B7 | Bound value-log preallocation; bounded value-log GC on close | P3 | S | B1, B3 |
| C1 | One walk policy for every indexer, plus `.chitonignore` | P2 | M | — |
| C2 | `chiton build` runs the sync engine as a parallel, batched pipeline | P1 | L | C1, C6, C3 |
| C3 | Diff-based source replacement that keeps other sources' edges | P1 | L | — |
| C4 | Canonical Windows path case in `file://` URIs | P2 | M | C2 |
| C5 | Rescan after an fsnotify event overflow | P2 | S | C1, C4 |
| C6 | Surface unreadable entries and honour cancellation in sync | P3 | S | C1 |
| C7 | Stop writing duplicate nodes and edges | P2 | S | C2 |
| C8 | Never read a symlink when indexing | P2 | S | C1, C2 |
| D1 | Prune live-sync findings from the file's own edges, not a store scan | P1 | M | — |
| D2 | Git hooks gate in memory, honour the timeout, and fail open | P1 | L | D1, C1 |
| D3 | Keep intra-community weight when Leiden aggregates | P1 | M | — |
| D4 | Let circular-import see cycles through relative imports | P3 | M | — |
| D5 | Emit graph findings and report members in a fixed order | P3 | S | D4 |
| D6 | Re-cluster only the touched region, read through the store | P3 | L | D1, D4 |
| D7 | Parallelise Brandes betweenness across pivots | P3 | S | — |
| E1 | Fail on a malformed `.chitonrc` instead of ignoring it | P2 | S | — |
| E2 | Resolve `.chiton/` state against `CHITON_PROJECT_DIR` | P2 | M | — |
| E3 | Refuse a remote embed endpoint chosen by the repository's `.chitonrc` | P2 | M | E1 |
| E4 | Store a passphrase verifier in the keyfile | P3 | M | E2 |
| E5 | Make the missing-passphrase failure actionable for assistant-launched servers | P2 | S | E4 |
| E6 | Close encryption gaps (plaintext notice, sealed sidecar, `chiton-out/.gitignore`) | P2 | L | E2, E4, E5, D6 |
| E7 | Create the MCP Unix socket owner-only | P3 | S | — |
| E8 | Take webhook secrets from a file, not a flag value | P3 | S | — |
| F1 | Keep config file modes and write through symlinks in the installer | P2 | S | — |
| F2 | Strip secret-bearing URL parts from browser history | P2 | M | — |
| F3 | Vendor a pinned vis-network into the HTML reports | P3 | S | — |
| F4 | Record each embedding's model and text hash; re-embed stale vectors | P2 | M | — |
| F5 | Stop embedding external stubs; store vectors in one slab | P3 | S | F4 |
| F6 | Ack Slack and Teams webhooks before ingesting; handle 429s | P2 | M | — |

---


---

## Section A: MCP server and network hardening

This section closes the ways a web page, a prompt-injected assistant, or an oversized query can misuse `chiton serve`. It adds bearer-token auth, an Origin allowlist and JSON-only POSTs to the HTTP transport (A1), then makes stdio the default (A9). It stops MCP tools reading files outside the project (A2), hardens the SSRF guard (A3), redacts people's identities (A4), bounds and compacts output (A5), stops whole-graph decodes (A6), binds `submit_extraction` to real tasks (A7), and guards a git ref (A8). Order: A1 → A9. A4 → A5 → A6, because A5 and A6 edit handlers whose signatures A4 changes; A6 also needs Section B's `store.KindCounter`. A2, A3, A7 and A8 are independent, but do A2 before A3 because both edit `internal/ingest/url`. A1, A4 and A9 edit `serveCmd` in `cmd/chiton/main.go`, and Section E's Unix-socket task edits the same function, so land them one at a time. All code below was applied to a scratch copy of the repo: `go build ./...`, `go vet`, `gofmt -l .` and `go test ./... -count=1` all pass. Every new test was run against the unmodified code, and each one fails there.

### Task A1: Authenticate the HTTP/SSE transport and refuse browser origins

**Finding:** With no transport flag, `chiton serve` falls through to HTTP/SSE on 127.0.0.1:7474 with no auth (`cmd/chiton/main.go:909-922`, flag default at `:927`). mcp-go v0.58.0 sets `Access-Control-Allow-Origin: *` on `/sse` whenever no CORS allowlist is configured (`server/sse.go:500-506`, `http_cors.go:105-107`). Its `/message` handler decodes any POST body without checking Origin or Content-Type (`sse.go:641-670`). A `text/plain` POST needs no CORS preflight, so any web page the user visits can call tools. mcp-go's DNS-rebinding check (`http_localhost.go:46-56`) does not help, because the page targets `127.0.0.1` directly. The server also comes from `sse.Start` (`sse.go:439-457`), so it has no `ReadHeaderTimeout`.
**Priority:** P1   **Effort:** M   **Depends on:** none
**Behaviour change:** Every HTTP-transport request must now carry `Authorization: Bearer <token>`. The token is generated at each start, written to `<graph dir>/mcp-token` (mode 0600, removed on exit), or pinned with `CHITON_MCP_TOKEN` (at least 16 characters). Requests with an `Origin` header are refused unless that origin is listed with the new `--allow-origin` flag (default: none). POSTs must be `application/json` and at most 8 MiB. **DECISION NEEDED:** token required by default (recommended). Alternative: make the token opt-in behind a `--require-token` flag and keep only the Origin and Content-Type checks by default.

**Files:**
- Create: `cmd/chiton/serve_http.go`
- Modify: `cmd/chiton/main.go:860-932` (`serveCmd`: new flag and the HTTP branch)
- Test: `cmd/chiton/serve_http_test.go`

**Interfaces:**
- Consumes: `fsutil.WriteFileAtomic(path string, data []byte, perm os.FileMode) error`; `baseContext`, `serveUntilDone` (`cmd/chiton/serve_transport.go:53,62`); mcp-go `WithSSECORS`, `WithCORSAllowedOrigins`, `SSEServer.CloseSessions`.
- Produces: `serveHTTP(cmd *cobra.Command, s *mcpsdk.MCPServer, addr, tokenPath string, allowedOrigins []string) error`, `serveTokenPath(graphPath string) string`, `newMCPHTTPServer(ctx context.Context, s *mcpsdk.MCPServer, addr, token string, allowedOrigins []string) (*http.Server, *mcpsdk.SSEServer)`, the `--allow-origin` flag, and the `CHITON_MCP_TOKEN` env var.

- [ ] **Step 1: Write the failing test**

Create `cmd/chiton/serve_http_test.go`:

```go
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
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./cmd/chiton/ -run 'TestMCPHTTP|TestServeToken' -count=1`
Expected: FAIL — build error: `undefined: newMCPHTTPServer`, `undefined: serveToken`, `undefined: mcpTokenEnv`.

- [ ] **Step 3: Implement**

Create `cmd/chiton/serve_http.go`:

```go
package main

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"fmt"
	"mime"
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
	token, written, err := serveToken(tokenPath)
	if err != nil {
		return err
	}
	if written {
		defer os.Remove(tokenPath)
		fmt.Fprintf(cmd.ErrOrStderr(), "serve: bearer token written to %s — clients send 'Authorization: Bearer <token>'\n", tokenPath)
	}
	srv, sse := newMCPHTTPServer(cmd.Context(), s, addr, token, allowedOrigins)
	fmt.Fprintf(cmd.OutOrStdout(), "serve: listening on %s (SSE, bearer-token auth)\n", addr)
	return serveUntilDone(cmd.Context(), srv.ListenAndServe, func(ctx context.Context) error {
		sse.CloseSessions() // as SSEServer.Shutdown does, so open SSE streams don't block shutdown
		return srv.Shutdown(ctx)
	})
}
```

In `cmd/chiton/main.go` `serveCmd`, add `var allowOrigins []string` after `var graphPath string` (line 865). Replace the `--http` lines of `Long` (lines 872-873) with:

```text
  --http <addr>  HTTP + SSE — a shared daemon across sessions, or for debugging
                 (default 127.0.0.1:7474; non-loopback binds print a warning).
                 Clients send 'Authorization: Bearer <token>'; the token is
                 written to <graph dir>/mcp-token at start (or pinned with
                 CHITON_MCP_TOKEN). Browser origins are refused unless listed
                 with --allow-origin.
```

Then replace the transport switch (lines 909-922) and add the flag after line 930:

```go
			// All three transports expose the same tool set.
			switch {
			case stdio:
				return mcpsdk.ServeStdio(s)
			case unixPath != "":
				return serveUnix(cmd, mcpsdk.NewSSEServer(s), unixPath)
			default:
				// No TLS, so warn without blocking on a wildcard bind.
				warnIfNonLoopback(cmd.ErrOrStderr(), addr,
					"the MCP server over plain HTTP — its bearer token and the entire graph (code, meetings, emails, PII) would cross the network unencrypted")
				return serveHTTP(cmd, s, addr, serveTokenPath(cfg.GraphPath), allowOrigins)
			}
```

```go
	cmd.Flags().StringSliceVar(&allowOrigins, "allow-origin", nil, "browser Origin allowed to call the HTTP transport, e.g. http://localhost:5173 (repeatable; default none)")
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./cmd/chiton/... -count=1; go vet ./cmd/chiton/...`
Expected: PASS

- [ ] **Step 5: Docs**
- `docs/MCP.md:27-33` flags table: change the `--http` row to: "HTTP/SSE listen address. Every request must send `Authorization: Bearer <token>`. The token is generated at each start and written to `<graph dir>/mcp-token` (mode 0600, removed on exit), or pinned with `CHITON_MCP_TOKEN` (at least 16 characters). Requests with a browser `Origin` are refused unless the origin is listed with `--allow-origin`. POSTs must be `Content-Type: application/json`. There is no TLS, and a non-loopback bind prints a warning." Add an `--allow-origin` row: default none, "Browser origin allowed to call the HTTP transport (repeatable). `*` is rejected." Below the table, add this client config snippet for Claude Code SSE:
  ```json
  { "mcpServers": { "chiton": { "type": "sse", "url": "http://127.0.0.1:7474/sse",
      "headers": { "Authorization": "Bearer ${CHITON_MCP_TOKEN}" } } } }
  ```
- `docs/SECURITY.md:410-416` table: set the HTTP row's Auth cell to "Per-start bearer token + Origin allowlist + JSON-only POST (no TLS)". In `:425-431` (§5.1), replace "no authentication, no TLS, and no origin check" with a description of the four controls: token, Origin allowlist (default none), `application/json` with an 8 MiB cap, and no `Access-Control-Allow-Origin: *`. Keep the "no TLS" warning. Change the threat-model row `:496` to **Shipped** (same four controls, no TLS), and update the out-of-scope bullet `:506-507`.
- `docs/INSTALL.md:232`: HTTP row now says it needs the bearer token. In the env table at `:267-273`, add a `CHITON_MCP_TOKEN` row: "`chiton serve --http`. Pins the bearer token (at least 16 characters) instead of a fresh one per start written to `<graph dir>/mcp-token`."
- `README.md:219`: append "HTTP requires the bearer token from `<graph dir>/mcp-token`" to the `chiton serve` row.
- `CONTRIBUTING.md:324`: replace "The MCP server has no authentication." with "The HTTP transport requires a per-start bearer token and refuses browser origins that are not allowlisted."
- `CHANGELOG.md` `### Security`: "- `chiton serve --http` now requires a bearer token (per start, written to `<graph dir>/mcp-token`, or `CHITON_MCP_TOKEN`), refuses browser `Origin`s not listed with `--allow-origin`, requires `application/json` POSTs, and no longer sends `Access-Control-Allow-Origin: *`. Before this, any web page could call tools on the loopback server with a `text/plain` POST."

- [ ] **Step 6: Commit**

```powershell
git add cmd/chiton/serve_http.go cmd/chiton/serve_http_test.go cmd/chiton/main.go docs/MCP.md docs/SECURITY.md docs/INSTALL.md README.md CONTRIBUTING.md CHANGELOG.md
git commit -m "fix(serve): require a bearer token, an origin allowlist and JSON POSTs on the HTTP transport"
```

### Task A2: Keep MCP tools from reading files outside the project

**Finding:** `add_url` treats scheme-less input and `file://` as local paths (`internal/ingest/url/fetch.go:44-56`), and its tool description invites this (`internal/mcp/meetings.go:168,171`). A prompt-injected Slack message or email can therefore steer the agent to call `add_url("file:///home/me/.aws/credentials")`, which puts the file into the graph. The agent can then send it out with a second `add_url` call to an attacker URL. Against the current code, `file://<path>` was ingested (2 nodes). `pending_analyses` also passes its MCP `path` straight to `filepath.WalkDir` (`internal/mcp/analyze.go:107` → `internal/analyze/llm.go:160`). It returns up to 60 KB of each file's contents (`llm.go:25,188,194`) from any directory, with no cap on `limit`, even on a read-only server. The walk also reads through symlinked files (`llm.go:188`).
**Priority:** P1   **Effort:** M   **Depends on:** none
**Behaviour change:** Over MCP, `add_url` accepts only `http`/`https` URLs, and the `chiton add-url` CLI still ingests local files. The `pending_analyses` `path` now resolves against the project root (`CHITON_PROJECT_DIR`, else the working directory). A path that resolves outside it, including through a symlink, is refused. `limit` is clamped to 1-200. The LLM-rule walk skips symlinked files, and the task `path` field is now absolute.

**Files:**
- Modify: `internal/ingest/url/ingest.go:53-61` (`AddOptions`), `:76-88` (`Add`)
- Modify: `internal/ingest/url/fetch.go` (add `requireRemote` after `hostAllowed`, `:201-228`)
- Modify: `cmd/chiton/meeting.go:248-252` (CLI sets `AllowLocal`)
- Modify: `internal/mcp/meetings.go:166-183` (`add_url` description)
- Modify: `internal/fsutil/fsutil.go` (add `ResolveWithin`)
- Modify: `internal/mcp/analyze.go:83-111` (`pending_analyses`)
- Modify: `internal/mcp/server.go:25-72` (`Deps.ProjectRoot`, `New` wiring)
- Modify: `cmd/chiton/main.go:946` (`buildSemanticDeps`)
- Modify: `internal/analyze/llm.go:160-171` (skip symlinks)
- Test: `internal/ingest/url/ingest_test.go`, `internal/fsutil/fsutil_test.go`, `internal/mcp/meetings_test.go`, `internal/mcp/analyze_test.go`, `internal/analyze/llm_test.go`

**Interfaces:**
- Consumes: `config.ProjectDir() string`; `urlingest.New`; `analyze.NewLLMService`.
- Produces: `fsutil.ResolveWithin(root, p string) (string, error)`; `urlingest.AddOptions.AllowLocal bool`; `mcpsrv.Deps.ProjectRoot string`; `pendingAnalysesHandler(svc *analyze.LLMService, root string) mcpsdk.ToolHandlerFunc`.

- [ ] **Step 1: Write the failing test**

Append to `internal/ingest/url/ingest_test.go`:

```go
// recordingFetcher records whether fetch ran, so a test can prove a refused input never reached it.
type recordingFetcher struct{ called bool }

func (f *recordingFetcher) fetch(context.Context, string) (*fetchResult, error) {
	f.called = true
	return &fetchResult{contentType: "text/plain", body: []byte("aws_secret_access_key = hunter2")}, nil
}

func TestAddRefusesLocalInputsUnlessAllowed(t *testing.T) {
	ctx := context.Background()
	for _, raw := range []string{
		"/etc/passwd", `C:\Users\me\.aws\credentials`, "~/.aws/credentials",
		"file:///etc/passwd", "FILE:///etc/passwd", "ftp://example.com/x", "https:///no-host",
	} {
		ing, _ := newIngester(t, nil)
		rf := &recordingFetcher{}
		ing.fetcher = rf
		if _, err := ing.Add(ctx, raw, AddOptions{}); err == nil {
			t.Errorf("Add(%q) succeeded; want it refused without AllowLocal", raw)
		}
		if rf.called {
			t.Errorf("Add(%q) reached the fetcher", raw)
		}
	}

	ing, _ := newIngester(t, nil)
	rf := &recordingFetcher{}
	ing.fetcher = rf
	if _, err := ing.Add(ctx, "/tmp/notes.txt", AddOptions{AllowLocal: true}); err != nil || !rf.called {
		t.Fatalf("Add with AllowLocal = %v (fetched=%v); want the CLI path to keep reading local files", err, rf.called)
	}
}
```

Append to `internal/fsutil/fsutil_test.go`:

```go
func TestResolveWithin(t *testing.T) {
	root, outside := t.TempDir(), t.TempDir()
	pkg := filepath.Join(root, "pkg")
	if err := os.Mkdir(pkg, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		in, want string // want "" means p must be refused
	}{
		{".", root},
		{"pkg", pkg},
		{pkg, pkg},
		{filepath.Join("pkg", ".."), root},
		{"..", ""},
		{filepath.Join("pkg", "..", ".."), ""},
		{outside, ""},
		{"missing", ""},
	} {
		got, err := ResolveWithin(root, tc.in)
		if tc.want == "" {
			if err == nil {
				t.Errorf("ResolveWithin(%q) = %q, want an error", tc.in, got)
			}
			continue
		}
		if err != nil || got != tc.want {
			t.Errorf("ResolveWithin(%q) = %q, %v; want %q", tc.in, got, err, tc.want)
		}
	}

	if err := os.Symlink(outside, filepath.Join(root, "escape")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if got, err := ResolveWithin(root, "escape"); err == nil {
		t.Errorf("ResolveWithin followed a symlink out of root to %q", got)
	}
}
```

Append to `internal/analyze/llm_test.go`:

```go
// A symlinked file inside the walked tree can point anywhere, so Pending must not read through it.
func TestLLMPendingSkipsSymlinks(t *testing.T) {
	ctx := context.Background()
	tmp, outside := t.TempDir(), t.TempDir()
	secret := filepath.Join(outside, "secret.go")
	if err := os.WriteFile(secret, []byte("package s\n\n// hunter2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(secret, filepath.Join(tmp, "link.go")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	reg := NewRegistry()
	reg.Register(fakeLLMRule{id: "llm/test", cat: CategoryBugs, sev: SeverityHigh, langs: []string{"go"}})
	specs, err := NewLLMService(nil, reg, 0.5, filepath.Join(tmp, ".chiton")).Pending(ctx, tmp, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, sp := range specs {
		if strings.Contains(sp.Input, "hunter2") {
			t.Fatalf("Pending read through symlink %s", sp.Path)
		}
	}
}
```

In `internal/mcp/meetings_test.go`, add `"os"`, `"path/filepath"` and `urlingest "github.com/Eric-Fernald/chiton/internal/ingest/url"` to the imports, then add:

```go
// A prompt-injected add_url("~/.aws/credentials") must not put the file into the graph.
func TestAddURLHandlerRefusesLocalFiles(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	secret := filepath.Join(t.TempDir(), "credentials")
	if err := os.WriteFile(secret, []byte("aws_secret_access_key = hunter2"), 0o600); err != nil {
		t.Fatal(err)
	}
	h := addURLHandler(urlingest.New(gs))
	for _, raw := range []string{secret, model.FileURI(secret)} {
		res, err := h(ctx, toolReq(map[string]any{"url": raw}))
		if err != nil {
			t.Fatal(err)
		}
		if !res.IsError {
			t.Errorf("add_url(%q) = %s; want a refusal", raw, resultText(t, res))
		}
	}
	if nodes, _ := gs.AllNodes(ctx); len(nodes) != 0 {
		t.Fatalf("add_url ingested a local file: %d nodes", len(nodes))
	}
}
```

In `internal/mcp/analyze_test.go`, add `"os"` to the imports and change `pendingAnalysesHandler(nil)` (line 150) to `pendingAnalysesHandler(nil, ".")`. Then add:

```go
// Task specs carry file contents, so pending_analyses must not walk outside the project root.
func TestPendingAnalysesStaysInsideProjectRoot(t *testing.T) {
	ctx := context.Background()
	root, outside := t.TempDir(), t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "in.go"), []byte("package in\n\nfunc F() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outside, "secret.go"), []byte("package s\n\nconst key = \"hunter2\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	h := pendingAnalysesHandler(analyze.NewLLMService(openStore(t), nil, 0.5, t.TempDir()), root)

	for _, p := range []string{outside, "..", filepath.Join("..", filepath.Base(outside))} {
		res, err := h(ctx, toolReq(map[string]any{"path": p}))
		if err != nil {
			t.Fatal(err)
		}
		if !res.IsError || strings.Contains(resultText(t, res), "hunter2") {
			t.Errorf("pending_analyses(path=%q) was not refused: %s", p, resultText(t, res))
		}
	}
	res, err := h(ctx, toolReq(map[string]any{"limit": 1}))
	if err != nil {
		t.Fatal(err)
	}
	if res.IsError || !strings.Contains(resultText(t, res), "in.go") {
		t.Errorf("the default path should walk the project root: %s", resultText(t, res))
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/ingest/url/ ./internal/fsutil/ ./internal/mcp/ ./internal/analyze/ -run 'RefusesLocal|ResolveWithin|StaysInside|SkipsSymlinks' -count=1`
Expected: FAIL. Build errors in `url` (`unknown field AllowLocal`), in `fsutil` (`undefined: ResolveWithin`) and in `mcpsrv` (`too many arguments in call to pendingAnalysesHandler`). `TestLLMPendingSkipsSymlinks` fails with "Pending read through symlink" on hosts that allow symlinks; it skips on Windows without the privilege. With only the `mcpsrv` build fixed, `TestAddURLHandlerRefusesLocalFiles` fails with "add_url ingested a local file: 2 nodes".

- [ ] **Step 3: Implement**

`internal/ingest/url/ingest.go`: add a field to `AddOptions` and a check in `Add` right after the read-only check (line 83):

```go
	MaxTextBytes int // per-section cap; 0 = default
	// AllowLocal permits bare paths and file:// URLs. The CLI sets it; the MCP add_url tool never does.
	AllowLocal bool
}
```

```go
	if store.IsReadOnly(ing.store) {
		return Report{}, fmt.Errorf("add-url: store is read-only — restart 'chiton serve --writable' or use the 'chiton add-url' CLI")
	}
	if !opts.AllowLocal {
		if err := requireRemote(rawURL); err != nil {
			return Report{}, err
		}
	}
```

`internal/ingest/url/fetch.go` (add after `isBlockedIP`; this file already imports `net/url` and `strings`):

```go
// requireRemote rejects anything but an absolute http(s) URL, so an assistant-driven add_url can't read local files.
func requireRemote(rawURL string) error {
	u, err := url.Parse(rawURL)
	if err != nil || u.Host == "" || (!strings.EqualFold(u.Scheme, "http") && !strings.EqualFold(u.Scheme, "https")) {
		return fmt.Errorf("add-url: only http/https URLs are accepted here (got %q); use the 'chiton add-url' CLI for local files", rawURL)
	}
	return nil
}
```

`cmd/chiton/meeting.go:248-252`:

```go
			rep, err := urlingest.New(gs).Add(cmd.Context(), args[0], urlingest.AddOptions{
				Kind:        kind,
				Author:      author,
				Contributor: contributor,
				AllowLocal:  true, // the CLI user names the file; only MCP callers are held to http(s)
			})
```

`internal/mcp/meetings.go:168-172` (the `add_url` description and `url` parameter):

```go
		mcplib.WithDescription("Fetch an http(s) URL (article, paper PDF, or YouTube transcript) and ingest its readable text as a url node with doc_section children, so its concepts are extracted by the normal pending_extractions flow. Local paths and file:// URLs are refused here; the 'chiton add-url' CLI ingests local files. Idempotent: re-adding a URL replaces it. Requires a writable store (chiton serve --writable)."),
		mcplib.WithString("url",
			mcplib.Required(),
			mcplib.Description("The http or https URL to ingest."),
		),
```

`internal/fsutil/fsutil.go`: add `"fmt"` and `"strings"` to the imports, and add:

```go
// ResolveWithin returns p (joined onto root when relative) as a clean absolute path, or an error when its
// symlink-free form lies outside root's. It keeps p's spelling, so file URIs match those a build over root records.
func ResolveWithin(root, p string) (string, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	realRoot, err := filepath.EvalSymlinks(absRoot)
	if err != nil {
		return "", err
	}
	abs := p
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(absRoot, p)
	}
	abs = filepath.Clean(abs)
	real, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(realRoot, real)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("%s is outside the project root %s", p, absRoot)
	}
	return abs, nil
}
```

`internal/mcp/analyze.go`: add `"github.com/Eric-Fernald/chiton/internal/fsutil"` to the imports. Change the `path` description (lines 86-88) and replace `pendingAnalysesHandler` (lines 102-111):

```go
		mcplib.WithString("path",
			mcplib.Description("Directory to analyse, relative to the project root (default '.'). Paths outside the project root are refused."),
		),
```

```go
// pendingAnalysesHandler serves pending_analyses. Task specs carry file contents, so the walk is confined to
// root: a caller-supplied path that resolves (symlinks included) outside it is refused.
func pendingAnalysesHandler(svc *analyze.LLMService, root string) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		if svc == nil {
			return mcplib.NewToolResultText("LLM analysis unavailable — the server was started without a graph store."), nil
		}
		dir, err := fsutil.ResolveWithin(root, req.GetString("path", "."))
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("pending analyses refused", err), nil
		}
		specs, err := svc.Pending(ctx, dir, req.GetString("rule", ""), clampInt(req.GetInt("limit", 20), 1, 200))
		return pendingResult(specs, err, "pending_analyses",
			"no pending analyses — every file is already analysed for the selected LLM rule(s).")
	}
}
```

`internal/mcp/server.go`: add a field to `Deps` (after `BaselineDir`, line 40) and wire it in `New` (replace line 63):

```go
	// ProjectRoot confines pending_analyses' path argument (default "."; serve passes config.ProjectDir()).
	ProjectRoot string
```

```go
	s.AddTool(pendingAnalysesTool(), pendingAnalysesHandler(deps.LLM, cmp.Or(deps.ProjectRoot, ".")))
```

`cmd/chiton/main.go` `buildSemanticDeps`, after `deps.URLIngester = urlingest.New(gs)` (line 946):

```go
	deps.ProjectRoot = config.ProjectDir()
```

`internal/analyze/llm.go:170` (inside the `WalkDir` callback, replacing `if extast.ForFile(path) == nil {`):

```go
		// A symlinked file can point anywhere, and its content would ship verbatim in the task input.
		if d.Type()&fs.ModeSymlink != 0 || extast.ForFile(path) == nil {
			return nil
		}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/ingest/url/... ./internal/fsutil/... ./internal/mcp/... ./internal/analyze/ ./cmd/chiton/... -count=1; go vet ./internal/ingest/url/... ./internal/fsutil/... ./internal/mcp/... ./internal/analyze/ ./cmd/chiton/...`
Expected: PASS

- [ ] **Step 5: Docs**
- `docs/MCP.md:898-934` (`add_url`): describe http(s) only, and add "Local paths and `file://` URLs are refused over MCP; `chiton add-url <path>` still ingests local files." Change the `url` parameter row to "The `http`/`https` URL to ingest."
- `docs/MCP.md:702-706` (`pending_analyses` table): change the `path` row to "Directory to analyse, relative to the project root (`CHITON_PROJECT_DIR`, else the server's working directory). Paths that resolve outside it (`..`, other absolute paths, symlinks) are refused." Add to the `limit` row: "clamped to 1-200".
- `docs/SECURITY.md:448-451,472-473` (§5.3): remove the claim that `add_url` reads local files. Replace the residual bullet "Local file reads are not path-restricted" with "Over MCP `add_url` accepts only http/https; local reads are CLI-only (`chiton add-url`)." Add one line saying `pending_analyses` is confined to the project root and skips symlinked files.
- `CHANGELOG.md` `### Security`: "- MCP `add_url` accepts only http/https URLs (the CLI still reads local files), and `pending_analyses` refuses paths outside the project root and no longer reads through symlinks, so an injected instruction can't make an MCP tool read `~/.aws/credentials`."

- [ ] **Step 6: Commit**

```powershell
git add internal/ingest/url internal/fsutil internal/mcp/analyze.go internal/mcp/analyze_test.go internal/mcp/meetings.go internal/mcp/meetings_test.go internal/mcp/server.go internal/analyze/llm.go internal/analyze/llm_test.go cmd/chiton/meeting.go cmd/chiton/main.go docs/MCP.md docs/SECURITY.md CHANGELOG.md
git commit -m "fix(mcp): keep add_url and pending_analyses from reading files outside the project"
```

### Task A3: Vet the dialled address in add_url's SSRF guard

**Finding:** `isBlockedIP` (`internal/ingest/url/fetch.go:225-228`) covers only loopback, RFC 1918, unspecified and link-local. It lets through 100.64.0.0/10, 198.18.0.0/15, 0.0.0.0/8 (non-zero hosts such as `0.1.2.3`), 240.0.0.0/4 (including 255.255.255.255), 192.0.0.0/24, NAT64 `64:ff9b::/96`, 6to4 `2002::/16` and Teredo. All of these were confirmed passing `hostAllowed` against the current code. `hostAllowed` resolves the name (`:213`), and the HTTP client then resolves it again to dial, so DNS rebinding can race the check. The code says so in its own comment (`:201-202`).
**Priority:** P2   **Effort:** S   **Depends on:** A2 (same package; its test file must compile)
**Behaviour change:** `add_url` refuses the ranges listed above. It now connects directly and ignores `HTTP_PROXY`/`HTTPS_PROXY`, because through a proxy the dial guard would vet the proxy's address instead of the target's. **DECISION NEEDED:** no proxy for `add_url` (recommended). Alternative: keep `http.ProxyFromEnvironment` and accept that behind a proxy only the literal-IP pre-check applies.

**Files:**
- Modify: `internal/ingest/url/fetch.go:3-16` (imports), `:201-228` (`hostAllowed`, `isBlockedIP`)
- Modify: `internal/ingest/url/ingest.go:5-15` (imports), `:34-51` (`New`)
- Test: `internal/ingest/url/ingest_test.go`

**Interfaces:**
- Consumes: none
- Produces: none exported (`isBlockedAddr`, `dialControl`, `blockedPrefixes` are package-private).

- [ ] **Step 1: Write the failing test**

Add `"fmt"`, `"net"`, `"net/http"` and `"net/http/httptest"` to the imports of `internal/ingest/url/ingest_test.go`, then append:

```go
func TestHostAllowedBlocksSpecialPurposeRanges(t *testing.T) {
	for _, h := range []string{
		"100.64.0.1", "100.127.255.254", // carrier-grade NAT
		"198.18.0.1", "198.19.255.254", // benchmarking
		"0.1.2.3", "240.0.0.1", "255.255.255.255", "192.0.0.170", "224.0.0.251",
		"::ffff:127.0.0.1", "::ffff:169.254.169.254", // IPv4-mapped
		"64:ff9b::a9fe:a9fe", "2002:a9fe:a9fe::1", "2001:0:4136:e378:8000:63bf:3fff:fdd2", // embed IPv4
		"fd00::1", "fe80::1%eth0", "ff02::1",
	} {
		if err := hostAllowed(h); err == nil {
			t.Errorf("hostAllowed(%q) = nil; want blocked", h)
		}
	}
	for _, h := range []string{"93.184.216.34", "1.1.1.1", "100.128.0.1", "198.20.0.1", "2606:4700:4700::1111"} {
		if err := hostAllowed(h); err != nil {
			t.Errorf("hostAllowed(%q) = %v; want allowed", h, err)
		}
	}
}

// A name resolving to loopback passes the literal-IP pre-check, so only the dial-time guard can stop it. The
// same guard sees every redirect hop and any DNS answer that changes between lookup and connect.
func TestFetchRefusesNameResolvingToLoopback(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, "internal admin page")
	}))
	defer ts.Close()
	_, port, err := net.SplitHostPort(ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	_, err = New(nil).fetcher.fetch(context.Background(), "http://localhost:"+port+"/")
	if err == nil || !strings.Contains(err.Error(), "refusing to connect") {
		t.Fatalf("fetch via localhost = %v; want the dial guard's refusal", err)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/ingest/url/ -run 'SpecialPurpose|NameResolving' -count=1`
Expected: FAIL. You should see `hostAllowed("100.64.0.1") = nil; want blocked` (and ten more), and `fetch via localhost = refusing to fetch localhost — it resolves to private/loopback ::1; want the dial guard's refusal`.

- [ ] **Step 3: Implement**

In `internal/ingest/url/fetch.go`, replace the `"net"` import with `"net/netip"` and add `"syscall"`. Then replace `hostAllowed` and `isBlockedIP` (lines 201-228):

```go
// blockedPrefixes are the non-public ranges add_url must never connect to (SSRF guard).
var blockedPrefixes = []netip.Prefix{
	netip.MustParsePrefix("0.0.0.0/8"),      // "this network"
	netip.MustParsePrefix("10.0.0.0/8"),     // private
	netip.MustParsePrefix("100.64.0.0/10"),  // carrier-grade NAT
	netip.MustParsePrefix("127.0.0.0/8"),    // loopback
	netip.MustParsePrefix("169.254.0.0/16"), // link-local, incl. cloud metadata
	netip.MustParsePrefix("172.16.0.0/12"),  // private
	netip.MustParsePrefix("192.0.0.0/24"),   // IETF protocol assignments
	netip.MustParsePrefix("192.168.0.0/16"), // private
	netip.MustParsePrefix("198.18.0.0/15"),  // benchmarking
	netip.MustParsePrefix("224.0.0.0/4"),    // multicast
	netip.MustParsePrefix("240.0.0.0/4"),    // reserved, incl. broadcast
	netip.MustParsePrefix("::/127"),         // unspecified and loopback
	netip.MustParsePrefix("64:ff9b::/96"),   // NAT64, embeds an IPv4 address
	netip.MustParsePrefix("64:ff9b:1::/48"), // local-use NAT64
	netip.MustParsePrefix("100::/64"),       // discard-only
	netip.MustParsePrefix("2001::/32"),      // Teredo, embeds an IPv4 address
	netip.MustParsePrefix("2002::/16"),      // 6to4, embeds an IPv4 address
	netip.MustParsePrefix("fc00::/7"),       // unique local
	netip.MustParsePrefix("fe80::/10"),      // link-local
	netip.MustParsePrefix("ff00::/8"),       // multicast
}

// isBlockedAddr reports whether a is non-public. It unmaps IPv4-mapped IPv6 and drops any zone first, since
// netip.Prefix.Contains matches neither.
func isBlockedAddr(a netip.Addr) bool {
	a = a.Unmap().WithZone("")
	for _, p := range blockedPrefixes {
		if p.Contains(a) {
			return true
		}
	}
	return !a.IsValid()
}

// hostAllowed fails fast on a literal non-public IP. A named host is vetted by dialControl, which sees the
// address actually dialled, so DNS rebinding and redirects can't slip between a lookup and the connect.
func hostAllowed(host string) error {
	if a, err := netip.ParseAddr(host); err == nil && isBlockedAddr(a) {
		return fmt.Errorf("refusing to fetch private/loopback address %s", host)
	}
	return nil
}

// dialControl is the add_url dialer's Control hook: it refuses a connection to a non-public address after DNS
// resolution, for the first request and every redirect alike.
func dialControl(_, address string, _ syscall.RawConn) error {
	ap, err := netip.ParseAddrPort(address)
	if err != nil {
		return fmt.Errorf("refusing to dial unparseable address %q", address)
	}
	if isBlockedAddr(ap.Addr()) {
		return fmt.Errorf("refusing to connect to private/loopback address %s", ap.Addr())
	}
	return nil
}
```

In `internal/ingest/url/ingest.go`, add `"net"` to the imports and replace the client construction in `New` (lines 35-45):

```go
func New(s store.GraphStore) *Ingester {
	dialer := &net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second, Control: dialControl}
	client := &http.Client{
		Timeout: 30 * time.Second,
		// No Proxy: through a proxy, dialControl would vet the proxy's address instead of the target's.
		Transport: &http.Transport{
			DialContext:           dialer.DialContext,
			ForceAttemptHTTP2:     true,
			TLSHandshakeTimeout:   10 * time.Second,
			ResponseHeaderTimeout: 20 * time.Second,
			IdleConnTimeout:       90 * time.Second,
		},
		// Fail fast on a redirect to a literal private IP; dialControl still vets every named host at connect time.
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return fmt.Errorf("stopped after 10 redirects")
			}
			return hostAllowed(req.URL.Hostname())
		},
	}
	return &Ingester{
		store:   s,
		fetcher: &httpFetcher{client: client},
		now:     time.Now,
	}
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/ingest/url/... -count=1; go vet ./internal/ingest/url/...`
Expected: PASS. The existing `TestHostAllowedBlocksPrivate` still passes.

- [ ] **Step 5: Docs**
- `docs/SECURITY.md:457-461` (§5.3 mitigations): replace the `hostAllowed` bullet with a new one. It should say a dial-time guard (`net.Dialer.Control`) checks the address actually connected to, on the first request and on every redirect. It refuses loopback, RFC 1918, link-local, CGNAT 100.64/10, 198.18/15, 0/8, 240/4, 192.0.0/24, multicast, ULA, and IPv4-embedding IPv6 (NAT64, 6to4, Teredo), and treats IPv4-mapped IPv6 as IPv4. Delete the "DNS-rebinding TOCTOU" residual bullet (`:466-469`) and add "`add_url` connects directly and ignores `HTTP(S)_PROXY`." Change the threat-model row `:495` from "with known gaps" to note only the yt-dlp gap.
- `CHANGELOG.md` `### Security`: "- `add_url`'s SSRF guard now checks the IP it actually dials (closing the DNS-rebinding race) and blocks CGNAT, benchmarking, reserved, multicast, NAT64/6to4/Teredo and IPv4-mapped ranges; `add_url` no longer goes through `HTTP(S)_PROXY`."

- [ ] **Step 6: Commit**

```powershell
git add internal/ingest/url docs/SECURITY.md CHANGELOG.md
git commit -m "fix(url): vet the dialled address and block all special-purpose ranges in add_url"
```

### Task A4: Redact people's identities in the generic graph tools

**Finding:** PII gating exists only in `search_meetings`/`decisions_about` (`internal/mcp/meetings.go:327-364`). `toLeanNode` (`internal/mcp/server.go:327-337`) returns every attr. So `get_node`, `query_graph`, `get_node_neighborhood`, `shortest_path` and `semantic_search` expose `person` labels and `attrs.email` (`internal/ingest/email/email.go:273-276`), email `from`/`from_name`/`to`/`cc` (`email.go:190-201`), meeting `participants` and `Name: ` transcript prefixes (`internal/extract/meeting/graph.go:115-118`, `transcript.go:35-46`), turn `speaker` and speaker-named turn labels (`graph.go:97-99`, `:178-187`), and action-item assignees. Edge endpoints (`toLeanEdge`, `server.go:389-396`) and the cluster labels in `god_nodes`/`graph_summary` (`server.go:446,549,558`) leak person names too. PII is represented as `attrs["pii"] == true` on `person` nodes, and each person has a `person://` source URI (`meeting.PersonURIScheme`).
**Priority:** P2   **Effort:** M   **Depends on:** none
**Behaviour change:** By default the graph tools show people as `[redacted person]` (source URI `person://[redacted]`). They drop the attrs listed above (meetings gain `participant_count`), replace speaker prefixes and turn labels with `[redacted]`, and strip action-item assignees. Label lookups (`get_node`, `get_node_neighborhood`, `query_graph`, `shortest_path`) and `semantic_search` no longer return a node whose label names a person. The new `chiton serve --include-pii` flag restores the old output. The meeting tools keep their per-call `include_pii`. Free text (email bodies, decision summaries) is still not scanned. **DECISION NEEDED:** redaction on by default (recommended). Alternative: off by default with `--redact-pii`. A second option: make `--include-pii` the ceiling for the meeting tools' per-call `include_pii` too.

**Files:**
- Create: `internal/mcp/redact.go`
- Modify: `internal/mcp/server.go:25-72` (`Deps.IncludePII`, `New`), `:160-306` (lookup and graph handlers), `:327-363` (delete `toLeanNode`, `toLeanNodes`, `nodeLabels`), `:414-453` (`godNodesHandler`), `:471` and `:549,558` (`graphSummaryHandler`)
- Modify: `internal/mcp/semantic.go:3-11,54-95` (`semanticSearchHandler`)
- Modify: `cmd/chiton/main.go:860-932` (`--include-pii` flag)
- Test: `internal/mcp/redact_test.go`; update `internal/mcp/server_test.go:34`

**Interfaces:**
- Consumes: `meeting.PersonURIScheme`, `meeting.PersonID`, `meeting.PersonURI`, `model.AttrStrings`
- Produces: `mcpsrv.Deps.IncludePII bool`; the `--include-pii` serve flag; package-private `type redactor struct{ includePII bool }` with methods `label`, `hidden`, `node`, `nodes`, `labels`, `overlayLabel` (A5 and A6 use it); test helper `call(t *testing.T, h mcpsdk.ToolHandlerFunc, args map[string]any) string`.

- [ ] **Step 1: Write the failing test**

Create `internal/mcp/redact_test.go`:

```go
package mcpsrv

import (
	"context"
	"strings"
	"testing"

	mcpsdk "github.com/mark3labs/mcp-go/server"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// call runs one tool handler and returns its text output.
func call(t *testing.T, h mcpsdk.ToolHandlerFunc, args map[string]any) string {
	t.Helper()
	res, err := h(context.Background(), toolReq(args))
	if err != nil {
		t.Fatal(err)
	}
	return resultText(t, res)
}

func TestGraphToolsRedactPeopleByDefault(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	seedLinkedMeeting(t, gs) // Alice and Bob speak in "Retry Policy Sync"
	alice := model.Node{
		ID: meeting.PersonID("alice@example.com"), Kind: model.KindPerson, Label: "Alice Smith",
		SourceURI:  meeting.PersonURI("alice@example.com"),
		Attrs:      map[string]any{"pii": true, "email": "alice@example.com"},
		Provenance: model.ProvExtracted, Confidence: 1,
	}
	mail := model.Node{
		ID: "mail1", Kind: model.KindEmail, Label: "Q3 plan", SourceURI: "email://q3",
		Attrs:      map[string]any{"subject": "Q3 plan", "from": "alice@example.com", "to": []any{"bob@example.com"}},
		Provenance: model.ProvExtracted, Confidence: 1,
	}
	if err := gs.UpsertNodes(ctx, []model.Node{alice, mail}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{{
		ID: model.NewEdgeID(mail.ID, alice.ID, model.EdgeMentioned), Src: mail.ID, Dst: alice.ID,
		Kind: model.EdgeMentioned, Provenance: model.ProvExtracted, Confidence: 1,
	}}); err != nil {
		t.Fatal(err)
	}
	overlay := &cluster.Results{Nodes: map[string]cluster.NodeResult{
		alice.ID: {Label: alice.Label, SourceURI: alice.SourceURI, Betweenness: 1},
	}}

	r := redactor{}
	outputs := map[string]string{
		"get_node meeting":     call(t, getNodeHandler(gs, r), map[string]any{"label": "Retry Policy Sync"}),
		"neighborhood meeting": call(t, neighborhoodHandler(gs, r), map[string]any{"label": "Retry Policy Sync", "depth": 1}),
		"neighborhood email":   call(t, neighborhoodHandler(gs, r), map[string]any{"label": "Q3 plan", "depth": 1}),
		"god_nodes":            call(t, godNodesHandler(overlay, r), map[string]any{}),
	}
	for tool, out := range outputs {
		for _, pii := range []string{"Alice", "Bob", "alice@example.com", "bob@example.com"} {
			if strings.Contains(out, pii) {
				t.Errorf("%s leaked %q: %s", tool, pii, out)
			}
		}
	}
	// A label search must not confirm that a person by that name exists.
	if out := call(t, queryHandler(gs, r), map[string]any{"query": "alice"}); !strings.Contains(out, "no nodes found") {
		t.Errorf("query_graph matched a person by name: %s", out)
	}

	// serve --include-pii restores the ungated view.
	out := call(t, getNodeHandler(gs, redactor{includePII: true}), map[string]any{"label": "Alice Smith"})
	if !strings.Contains(out, "alice@example.com") {
		t.Errorf("include-pii get_node = %s, want the address", out)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/mcp/ -run TestGraphToolsRedactPeopleByDefault -count=1`
Expected: FAIL — build error: `undefined: redactor` and `too many arguments in call to getNodeHandler`.

- [ ] **Step 3: Implement**

Create `internal/mcp/redact.go`:

```go
package mcpsrv

import (
	"maps"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

const (
	redactedName   = "[redacted]"
	redactedPerson = "[redacted person]"
)

// personAttrKeys are attrs that name or address people: meeting participants and speakers, email
// correspondents, and a person's own address.
var personAttrKeys = []string{"participants", "speaker", "person_id", "from", "from_name", "to", "cc", "email"}

// redactor projects nodes for tool output. Unless includePII (serve --include-pii), it hides who people are
// while keeping what was said and decided.
type redactor struct{ includePII bool }

// isPerson reports whether n identifies a person: a person node, or any node tagged pii.
func isPerson(n model.Node) bool {
	pii, _ := n.Attrs["pii"].(bool)
	return n.Kind == model.KindPerson || pii
}

// label is n's label with any person's name removed.
func (r redactor) label(n model.Node) string {
	if r.includePII {
		return n.Label
	}
	if isPerson(n) {
		return redactedPerson
	}
	if sp := model.AttrString(n.Attrs, "speaker"); sp != "" && n.Kind == model.KindMeetingTurn {
		if rest, ok := strings.CutPrefix(n.Label, sp); ok {
			return redactedName + rest
		}
	}
	return n.Label
}

// hidden reports whether a label lookup must skip n, because matching its label would itself reveal a name.
func (r redactor) hidden(n model.Node) bool { return r.label(n) != n.Label }

// node projects n to its lean shape, redacted.
func (r redactor) node(n model.Node) leanNode {
	ln := leanNode{
		Kind: n.Kind, Label: r.label(n), SourceURI: n.SourceURI, Span: n.Span,
		Attrs: n.Attrs, Provenance: n.Provenance, Confidence: n.Confidence,
	}
	switch {
	case r.includePII:
	case isPerson(n):
		ln.SourceURI = meeting.PersonURIScheme + redactedName
		ln.Attrs = map[string]any{"pii": true}
	default:
		ln.Attrs = redactAttrs(n.Attrs)
	}
	return ln
}

// nodes projects nodes in order. The result is never nil, so JSON renders [], not null.
func (r redactor) nodes(nodes []model.Node) []leanNode {
	out := make([]leanNode, len(nodes))
	for i, n := range nodes {
		out[i] = r.node(n)
	}
	return out
}

// labels maps node ID to redacted label, for naming edge endpoints.
func (r redactor) labels(nodes []model.Node) map[string]string {
	m := make(map[string]string, len(nodes))
	for _, n := range nodes {
		m[n.ID] = r.label(n)
	}
	return m
}

// overlayLabel is a cluster result's label, hidden for a person (recognised by its person:// source URI).
func (r redactor) overlayLabel(nr cluster.NodeResult) string {
	if !r.includePII && strings.HasPrefix(nr.SourceURI, meeting.PersonURIScheme) {
		return redactedPerson
	}
	return nr.Label
}

// redactAttrs returns attrs without person identifiers, cloning only when there is something to drop.
func redactAttrs(attrs map[string]any) map[string]any {
	_, hasActions := attrs["action_items"]
	if !hasActions && !hasAnyKey(attrs, personAttrKeys) {
		return attrs
	}
	out := maps.Clone(attrs)
	for _, k := range personAttrKeys {
		delete(out, k)
	}
	if ps := model.AttrStrings(attrs, "participants"); len(ps) > 0 {
		out["participant_count"] = len(ps)
		if text, ok := out["text"].(string); ok {
			out["text"] = redactSpeakers(text, ps)
		}
	}
	if items, ok := out["action_items"].([]any); ok {
		out["action_items"] = actionTexts(items)
	}
	return out
}

func hasAnyKey(m map[string]any, keys []string) bool {
	for _, k := range keys {
		if _, ok := m[k]; ok {
			return true
		}
	}
	return false
}

// redactSpeakers replaces the "<participant>: " line prefixes that meeting.Transcript.Text writes.
func redactSpeakers(text string, participants []string) string {
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		for _, p := range participants {
			if rest, ok := strings.CutPrefix(line, p+": "); ok {
				lines[i] = redactedName + ": " + rest
				break
			}
		}
	}
	return strings.Join(lines, "\n")
}

// actionTexts copies decision action items keeping only their text, since assignees are people.
func actionTexts(items []any) []any {
	out := make([]any, 0, len(items))
	for _, it := range items {
		if m, ok := it.(map[string]any); ok {
			out = append(out, map[string]any{"text": m["text"]})
		}
	}
	return out
}
```

`internal/mcp/server.go`:
1. Add to `Deps` (after `BaselineDir`/`ProjectRoot`):
```go
	// IncludePII turns off redaction of people's names and addresses in the graph tools (serve --include-pii).
	IncludePII bool
```
2. In `New`, build the redactor once and pass it to the graph handlers (these lines replace `server.go:47-52,55`):
```go
	r := redactor{includePII: deps.IncludePII}
	s.AddTool(getNodeTool(), getNodeHandler(gs, r))
	s.AddTool(neighborhoodTool(), neighborhoodHandler(gs, r))
	s.AddTool(queryTool(), queryHandler(gs, r))
	s.AddTool(shortestPathTool(), shortestPathHandler(gs, r))
	s.AddTool(godNodesTool(), godNodesHandler(overlay, r))
	s.AddTool(graphSummaryTool(), graphSummaryHandler(gs, overlay, r))
```
```go
	s.AddTool(semanticSearchTool(), semanticSearchHandler(deps.Retriever, r))
```
3. Replace `lookupNode`, `getNodeHandler`, `neighborhoodHandler`, `queryHandler` and `shortestPathHandler` (lines 160-306):
```go
// lookupNode resolves label (plus an optional file hint) to one node. A non-nil result is the tool response
// to return instead: a search failure, or a no-match message, also given for a node whose label names a person.
func lookupNode(ctx context.Context, gs store.GraphStore, r redactor, label, file string) (model.Node, *mcplib.CallToolResult) {
	node, found, err := store.FindNode(ctx, gs, label, file)
	if err != nil {
		return model.Node{}, mcplib.NewToolResultErrorFromErr("search failed", err)
	}
	if !found || r.hidden(node) {
		return model.Node{}, mcplib.NewToolResultText(fmt.Sprintf("no node found matching %q", label))
	}
	return node, nil
}

func getNodeHandler(gs store.GraphStore, r redactor) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		label, err := req.RequireString("label")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		node, errRes := lookupNode(ctx, gs, r, label, req.GetString("file", ""))
		if errRes != nil {
			return errRes, nil
		}
		return jsonResult(r.node(node))
	}
}

func neighborhoodHandler(gs store.GraphStore, r redactor) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		label, err := req.RequireString("label")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		prefix := req.GetString("source_prefix", "")
		depth := clampInt(req.GetInt("depth", 2), 1, 5)

		seed, errRes := lookupNode(ctx, gs, r, label, req.GetString("file", ""))
		if errRes != nil {
			return errRes, nil
		}

		sub, err := gs.Neighbors(ctx, seed.ID, depth, parseEdgeKinds(req))
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("neighbors failed", err), nil
		}

		labels := r.labels(sub.Nodes)
		labels[seed.ID] = r.label(seed)
		leanNodes, leanEdges := filterByPrefix(labels[seed.ID],
			r.nodes(sub.Nodes), toLeanEdges(sub.Edges, labels), prefix)

		type result struct {
			Seed    leanNode   `json:"seed"`
			Nodes   []leanNode `json:"nodes"`
			Edges   []leanEdge `json:"edges"`
			Summary string     `json:"summary"`
		}
		return jsonResult(result{
			Seed:  r.node(seed),
			Nodes: leanNodes,
			Edges: leanEdges,
			Summary: fmt.Sprintf("%d nodes, %d edges within depth %d of %q",
				len(leanNodes), len(leanEdges), depth, labels[seed.ID]),
		})
	}
}

func queryHandler(gs store.GraphStore, r redactor) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		query, err := req.RequireString("query")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		limit := clampInt(req.GetInt("limit", 10), 1, 100)
		prefix := req.GetString("source_prefix", "")
		nodes, err := gs.SearchNodes(ctx, query, limit)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search failed", err), nil
		}
		lower := strings.ToLower(prefix)
		lean := make([]leanNode, 0, len(nodes))
		for _, n := range nodes {
			if !r.hidden(n) && (prefix == "" || strings.Contains(strings.ToLower(n.SourceURI), lower)) {
				lean = append(lean, r.node(n))
			}
		}
		switch {
		case len(lean) > 0:
			return jsonResult(lean)
		case prefix != "":
			return mcplib.NewToolResultText(fmt.Sprintf("no nodes found matching %q with source_prefix %q", query, prefix)), nil
		default:
			return mcplib.NewToolResultText(fmt.Sprintf("no nodes found matching %q", query)), nil
		}
	}
}

func shortestPathHandler(gs store.GraphStore, r redactor) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		fromLabel, err := req.RequireString("from")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		toLabel, err := req.RequireString("to")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}

		fromNodes, err := gs.SearchNodes(ctx, fromLabel, 1)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search 'from' failed", err), nil
		}
		if len(fromNodes) == 0 || r.hidden(fromNodes[0]) {
			return mcplib.NewToolResultText(fmt.Sprintf("no node found matching 'from' %q", fromLabel)), nil
		}

		toNodes, err := gs.SearchNodes(ctx, toLabel, 1)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search 'to' failed", err), nil
		}
		if len(toNodes) == 0 || r.hidden(toNodes[0]) {
			return mcplib.NewToolResultText(fmt.Sprintf("no node found matching 'to' %q", toLabel)), nil
		}

		path, err := gs.ShortestPath(ctx, fromNodes[0].ID, toNodes[0].ID)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("shortest path failed", err), nil
		}
		if len(path.Nodes) == 0 {
			return mcplib.NewToolResultText(
				fmt.Sprintf("no path found between %q and %q", fromNodes[0].Label, toNodes[0].Label),
			), nil
		}
		type leanPath struct {
			Nodes []leanNode `json:"nodes"`
			Edges []leanEdge `json:"edges"`
		}
		return jsonResult(leanPath{
			Nodes: r.nodes(path.Nodes),
			Edges: toLeanEdges(path.Edges, r.labels(path.Nodes)),
		})
	}
}
```
4. Delete `toLeanNode` (`:327-337`), `toLeanNodes` (`:339-346`) and `nodeLabels` (`:357-363`); `redactor.node`, `.nodes` and `.labels` replace them. Keep `leanNode`, `leanEdge`, `toLeanEdges`, `toLeanEdge` and `filterByPrefix`.
5. `godNodesHandler`: change the signature to `func godNodesHandler(overlay *cluster.Results, r redactor) mcpsdk.ToolHandlerFunc` and line 446 to `Label: r.overlayLabel(nr),`.
6. `graphSummaryHandler`: change the signature (line 471) to `func graphSummaryHandler(gs store.GraphStore, overlay *cluster.Results, r redactor) mcpsdk.ToolHandlerFunc`. Change line 549 to `c.TopNodes = append(c.TopNodes, r.overlayLabel(nr))` and line 558 to `Label: r.overlayLabel(nr),`.

`internal/mcp/semantic.go`: add `"slices"` and `"github.com/Eric-Fernald/chiton/internal/graph/model"` to the imports. Change the signature to `func semanticSearchHandler(eng *retrieve.Engine, r redactor) mcpsdk.ToolHandlerFunc`, and replace the lean-output block (lines 75-93) with:

```go
		// Lean output: strip embeddings and internal metadata, and drop nodes whose label names a person.
		type scoredLean struct {
			Score float32  `json:"score"`
			Hop   int      `json:"hop"`
			Node  leanNode `json:"node"`
		}
		labels := make(map[string]string, len(res.Nodes))
		nodes := make([]scoredLean, 0, len(res.Nodes))
		for _, s := range res.Nodes {
			if r.hidden(s.Node) {
				continue
			}
			labels[s.Node.ID] = s.Node.Label
			nodes = append(nodes, scoredLean{Score: s.Score, Hop: s.Hop, Node: r.node(s.Node)})
		}
		edges := slices.DeleteFunc(res.Edges, func(e model.Edge) bool {
			_, src := labels[e.Src]
			_, dst := labels[e.Dst]
			return !src || !dst
		})
		return jsonResult(struct {
			Nodes   []scoredLean `json:"nodes"`
			Edges   []leanEdge   `json:"edges"`
			Summary string       `json:"summary"`
		}{nodes, toLeanEdges(edges, labels), res.Summary})
```

`cmd/chiton/main.go` `serveCmd`: add `var includePII bool`. Set it on the deps right after `buildSemanticDeps` (line 906), and add the flag:

```go
			deps := buildSemanticDeps(cmd, cfg, gs, clusterResults)
			deps.IncludePII = includePII
			s := mcpsrv.New(gs, clusterResults, deps)
```

```go
	cmd.Flags().BoolVar(&includePII, "include-pii", false, "return people's names and email addresses from the generic graph tools (redacted by default)")
```

Update the existing test call at `internal/mcp/server_test.go:34` to `graphSummaryHandler(openStore(t), overlay, redactor{})`.

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/mcp/... ./cmd/chiton/... -count=1; go vet ./internal/mcp/... ./cmd/chiton/...`
Expected: PASS

- [ ] **Step 5: Docs**
- `docs/MCP.md:134-144`: rename the section "PII redaction". Say that by default the graph tools replace people with `[redacted person]` and drop the person-identifying attrs (participants, speaker, email from/to/cc, addresses, action-item assignees; meetings keep a `participant_count`). Transcript and turn speaker names become `[redacted]`. Label lookups never match a node whose label names a person. `chiton serve --include-pii` turns redaction off, and the meeting tools keep their per-call `include_pii`. Add an `--include-pii` row (default `false`) to the flags table at `:27-33`.
- `docs/SECURITY.md:326-349` (§3): rewrite "Where the gate applies" to describe server-wide redaction plus the two meeting tools' per-call flag. Keep the note that free-text PII (bodies, summaries) and browser-history `url` nodes are not redacted, and that `pending_meeting_links` task specs still carry transcripts verbatim. Change the threat-model row `:493` to "Shipped: redacted by default in every read tool; `--include-pii` opts out".
- `README.md:219`: add `--include-pii` to the `chiton serve` row.
- The local, gitignored `CLAUDE.md:96-99` says the generic tools return person nodes ungated. Update your working copy; it is not committed.
- `CHANGELOG.md` `### Security`: "- The graph tools (`get_node`, `query_graph`, `get_node_neighborhood`, `shortest_path`, `semantic_search`, `god_nodes`, `graph_summary`) now redact people's names, email addresses, speakers and assignees by default; `chiton serve --include-pii` restores them."

- [ ] **Step 6: Commit**

```powershell
git add internal/mcp/redact.go internal/mcp/redact_test.go internal/mcp/server.go internal/mcp/server_test.go internal/mcp/semantic.go cmd/chiton/main.go docs/MCP.md docs/SECURITY.md README.md CHANGELOG.md
git commit -m "feat(mcp): redact people's identities in the graph tools unless serve --include-pii"
```

### Task A5: Bound get_node_neighborhood and emit compact JSON

**Finding:** `get_node_neighborhood` returns everything reachable within `depth` (up to 5), with no node or token cap (`internal/mcp/server.go:192-230`). `jsonResult` pretty-prints with `json.MarshalIndent` (`:624-630`) and HTML-escapes `<`, `>` and `&` into `\u003c`-style sequences. The audit measured `errors.Is` at depth 2 as 943 nodes and 1,574 edges (about 1 MB), with indentation adding 33%. `budgetTrim` (`internal/graph/retrieve/retrieve.go:245-261`) already implements the "longest prefix within ~4 bytes/token" rule, but it only accepts `[]Scored`.
**Priority:** P3   **Effort:** S   **Depends on:** A4
**Behaviour change:** `get_node_neighborhood` gains `max_nodes` (default 100, max 500) and `max_tokens` (default 4000, 500-16000). It keeps the nearest hops first, and the response gains `"truncated": true` plus a hint in `summary` when nodes were cut. Every tool result becomes compact JSON with no HTML escaping. When `source_prefix` is set, the seed now always appears in `nodes`.

**Files:**
- Modify: `internal/graph/retrieve/retrieve.go:245-261`
- Modify: `internal/mcp/server.go:90-115` (`neighborhoodTool`), `neighborhoodHandler` (as rewritten in A4), `:365-387` (delete `filterByPrefix`), `:624-630` (`jsonResult`)
- Test: `internal/graph/retrieve/retrieve_test.go`, `internal/mcp/server_test.go`

**Interfaces:**
- Consumes: `redactor` (A4); `call` test helper (A4)
- Produces: `retrieve.TrimToBudget[T any](items []T, maxTokens int, cost func(T) int) []T`

- [ ] **Step 1: Write the failing test**

Append to `internal/graph/retrieve/retrieve_test.go`:

```go
func TestTrimToBudget(t *testing.T) {
	size := func(n int) int { return n }
	if got := TrimToBudget([]int{4, 4, 4}, 2, size); len(got) != 2 {
		t.Errorf("kept %d items in an 8-byte budget, want 2", len(got))
	}
	if got := TrimToBudget([]int{100, 1}, 1, size); len(got) != 1 {
		t.Errorf("kept %d items, want just the first even though it is over budget", len(got))
	}
	if got := TrimToBudget([]int{1, 1}, 10, size); len(got) != 2 {
		t.Errorf("kept %d items, want all when they fit", len(got))
	}
}
```

In `internal/mcp/server_test.go`, add `"strings"`, `"github.com/Eric-Fernald/chiton/internal/graph/model"` and `"github.com/Eric-Fernald/chiton/internal/graph/store"` to the imports, then add:

```go
func TestJSONResultIsCompactAndUnescaped(t *testing.T) {
	res, err := jsonResult(map[string]string{"sig": "func f(ch <-chan int) bool { return a && b }"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := resultText(t, res), `{"sig":"func f(ch <-chan int) bool { return a && b }"}`; got != want {
		t.Errorf("jsonResult = %s, want %s", got, want)
	}
}

// seedFanOut stores a hub symbol that CALLS n leaf symbols.
func seedFanOut(t *testing.T, gs store.GraphStore, n int) {
	t.Helper()
	ctx := context.Background()
	nodes := []model.Node{{ID: "hub", Kind: model.KindSymbol, Label: "Hub", SourceURI: "file://hub.go",
		Provenance: model.ProvExtracted, Confidence: 1}}
	var edges []model.Edge
	for i := range n {
		id := fmt.Sprintf("leaf%03d", i)
		nodes = append(nodes, model.Node{ID: id, Kind: model.KindSymbol, Label: id, SourceURI: "file://leaf.go",
			Provenance: model.ProvExtracted, Confidence: 1})
		edges = append(edges, model.Edge{ID: model.NewEdgeID("hub", id, model.EdgeCalls), Src: "hub", Dst: id,
			Kind: model.EdgeCalls, Provenance: model.ProvExtracted, Confidence: 1})
	}
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, edges); err != nil {
		t.Fatal(err)
	}
}

func TestNeighborhoodCapsLargeFanOut(t *testing.T) {
	gs := openStore(t)
	seedFanOut(t, gs, 300)
	h := neighborhoodHandler(gs, redactor{})

	var got struct {
		Nodes     []leanNode `json:"nodes"`
		Edges     []leanEdge `json:"edges"`
		Truncated bool       `json:"truncated"`
	}
	out := call(t, h, map[string]any{"label": "Hub", "depth": 1, "max_nodes": 50})
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out)
	}
	if len(got.Nodes) != 50 || len(got.Edges) != 49 || !got.Truncated {
		t.Errorf("max_nodes=50: %d nodes, %d edges, truncated=%v; want 50, 49, true", len(got.Nodes), len(got.Edges), got.Truncated)
	}

	out = call(t, h, map[string]any{"label": "Hub", "depth": 1, "max_nodes": 500, "max_tokens": 500})
	if len(out) > 3000 {
		t.Errorf("max_tokens=500 produced %d bytes, want roughly 2000 plus the seed and summary", len(out))
	}
	if strings.Contains(out, "\n") {
		t.Error("neighborhood output is indented; want compact JSON")
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/graph/retrieve/ ./internal/mcp/ -run 'TrimToBudget|JSONResultIsCompact|NeighborhoodCaps' -count=1`
Expected: FAIL. `retrieve` fails to build with `undefined: TrimToBudget`. In `mcpsrv`, `jsonResult` returns indented output with `\u003c-chan` and `\u0026\u0026`, and the uncapped call gives `max_nodes=50: 301 nodes, 300 edges, truncated=false`.

- [ ] **Step 3: Implement**

`internal/graph/retrieve/retrieve.go`: replace `budgetTrim` (lines 245-261):

```go
// TrimToBudget keeps the longest prefix of items whose summed cost (in bytes) fits maxTokens at ~4 bytes per
// token. It always keeps the first item.
func TrimToBudget[T any](items []T, maxTokens int, cost func(T) int) []T {
	budget := maxTokens * 4
	used := 0
	for i, it := range items {
		c := cost(it)
		if i > 0 && used+c > budget {
			return items[:i]
		}
		used += c
	}
	return items
}

// budgetTrim keeps the highest-ranked nodes whose cumulative serialized size fits within maxTokens.
func budgetTrim(scored []Scored, maxTokens int) []Scored {
	return TrimToBudget(scored, maxTokens, func(s Scored) int {
		b, _ := json.Marshal(s.Node)
		return len(b)
	})
}
```

`internal/mcp/server.go`: in `neighborhoodTool`, insert these two parameters before `mcplib.WithReadOnlyHintAnnotation(true)` (line 113):

```go
		mcplib.WithInteger("max_nodes",
			mcplib.DefaultNumber(100),
			mcplib.Description("Max nodes to return, nearest hops first (default 100, max 500). The result sets truncated when more were reachable."),
			mcplib.Min(1),
			mcplib.Max(500),
		),
		mcplib.WithInteger("max_tokens",
			mcplib.DefaultNumber(4000),
			mcplib.Description("Approximate token budget for the returned nodes and edges (default 4000, max 16000)."),
			mcplib.Min(500),
			mcplib.Max(16000),
		),
```

Replace `neighborhoodHandler` (A4's version) and `filterByPrefix` (lines 365-387) with:

```go
func neighborhoodHandler(gs store.GraphStore, r redactor) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		label, err := req.RequireString("label")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		depth := clampInt(req.GetInt("depth", 2), 1, 5)
		maxNodes := clampInt(req.GetInt("max_nodes", 100), 1, 500)
		maxTokens := clampInt(req.GetInt("max_tokens", 4000), 500, 16000)

		seed, errRes := lookupNode(ctx, gs, r, label, req.GetString("file", ""))
		if errRes != nil {
			return errRes, nil
		}
		sub, err := gs.Neighbors(ctx, seed.ID, depth, parseEdgeKinds(req))
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("neighbors failed", err), nil
		}
		reachable := keepPrefix(seed.ID, sub.Nodes, req.GetString("source_prefix", ""))
		nodes, edges := capNeighborhood(reachable, sub.Edges, maxNodes, maxTokens, r)

		type result struct {
			Seed      leanNode   `json:"seed"`
			Nodes     []leanNode `json:"nodes"`
			Edges     []leanEdge `json:"edges"`
			Truncated bool       `json:"truncated,omitempty"`
			Summary   string     `json:"summary"`
		}
		res := result{
			Seed:      r.node(seed),
			Nodes:     nodes,
			Edges:     edges,
			Truncated: len(nodes) < len(reachable),
			Summary:   fmt.Sprintf("%d nodes, %d edges within depth %d of %q", len(nodes), len(edges), depth, r.label(seed)),
		}
		if res.Truncated {
			res.Summary += fmt.Sprintf(" (truncated from %d reachable nodes; raise max_nodes/max_tokens or narrow edge_kinds/source_prefix)", len(reachable))
		}
		return jsonResult(res)
	}
}

// keepPrefix keeps the seed and the nodes whose source_uri contains prefix (case-insensitive).
func keepPrefix(seedID string, nodes []model.Node, prefix string) []model.Node {
	if prefix == "" {
		return nodes
	}
	lower := strings.ToLower(prefix)
	var out []model.Node
	for _, n := range nodes {
		if n.ID == seedID || strings.Contains(strings.ToLower(n.SourceURI), lower) {
			out = append(out, n)
		}
	}
	return out
}

// capNeighborhood keeps the longest BFS-order prefix of nodes within maxNodes and maxTokens, pricing each node
// as its lean JSON plus its edges back to earlier nodes, and returns that prefix with the edges among it.
func capNeighborhood(nodes []model.Node, edges []model.Edge, maxNodes, maxTokens int, r redactor) ([]leanNode, []leanEdge) {
	pos := make(map[string]int, len(nodes))
	for i, n := range nodes {
		pos[n.ID] = i
	}
	labels := r.labels(nodes)
	lean := r.nodes(nodes)
	cost := make([]int, len(nodes))
	for i := range lean {
		cost[i] = jsonLen(lean[i])
	}
	var within []model.Edge
	for _, e := range edges {
		i, okSrc := pos[e.Src]
		j, okDst := pos[e.Dst]
		if okSrc && okDst {
			within = append(within, e)
			cost[max(i, j)] += jsonLen(toLeanEdge(e, labels))
		}
	}
	idx := make([]int, min(len(nodes), maxNodes))
	for i := range idx {
		idx[i] = i
	}
	kept := len(retrieve.TrimToBudget(idx, maxTokens, func(i int) int { return cost[i] }))
	keptEdges := []leanEdge{}
	for _, e := range within {
		if pos[e.Src] < kept && pos[e.Dst] < kept {
			keptEdges = append(keptEdges, toLeanEdge(e, labels))
		}
	}
	return lean[:kept], keptEdges
}

// jsonLen is v's compact JSON size, the unit token budgets are priced in (~4 bytes per token).
func jsonLen(v any) int {
	b, _ := json.Marshal(v)
	return len(b)
}
```

Add `"bytes"` to the imports of `internal/mcp/server.go` and replace `jsonResult` (lines 624-630):

```go
// jsonResult renders v as compact JSON. HTML escaping is off, so code like "<-ch" or "a && b" stays readable
// and doesn't grow into \u003c / \u0026 sequences that cost tokens.
func jsonResult(v any) (*mcplib.CallToolResult, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return mcplib.NewToolResultErrorFromErr("marshal failed", err), nil
	}
	return mcplib.NewToolResultText(strings.TrimSuffix(buf.String(), "\n")), nil
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/graph/retrieve/... ./internal/mcp/... -count=1; go vet ./internal/graph/retrieve/... ./internal/mcp/...`
Expected: PASS. The existing `TestQueryTokenBudgetTrims` still passes on the refactored `budgetTrim`.

- [ ] **Step 5: Docs**
- `docs/MCP.md:87-112` (Response shape): add "Responses are compact JSON without HTML escaping; the examples here are pretty-printed for reading."
- `docs/MCP.md:194-200` (`get_node_neighborhood` table): add `max_nodes` (default `100`, min `1`, max `500`, "nearest hops first") and `max_tokens` (default `4000`, min `500`, max `16000`). Below the response example, add: "`truncated: true` (and a hint in `summary`) means more nodes were reachable than returned. When `source_prefix` is set, the seed is always included in `nodes`."
- `CHANGELOG.md` `### Changed`: "- `get_node_neighborhood` takes `max_nodes`/`max_tokens` (defaults 100 / 4000) and reports `truncated`; all MCP tool results are compact JSON (no indentation or HTML escaping)."

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/retrieve/retrieve.go internal/graph/retrieve/retrieve_test.go internal/mcp/server.go internal/mcp/server_test.go docs/MCP.md CHANGELOG.md
git commit -m "perf(mcp): cap get_node_neighborhood by nodes and tokens, emit compact JSON"
```

### Task A6: Stop decoding the whole graph per graph_summary / meeting call

**Finding:** `graph_summary` loads every node and edge just to count kinds (`internal/mcp/server.go:476-510`). `search_meetings` and `decisions_about` decode all edges on every call to find DECIDED_IN (`internal/mcp/meetings.go:388-429`) and lowercase every transcript on every call (`meetings.go:459`). The store has no public edge-adjacency API except `Neighbors` (`internal/graph/store/store.go:86`; the Badger version reads the `s:`/`d:` adjacency keys, `badger/store.go:200-246,351-407`). A read-only store is a snapshot fixed at open (`OpenReadOnlyWith`, `badger/store.go:83-91`).
**Priority:** P3   **Effort:** M   **Depends on:** A4 (handler signatures, `call`), Section B's KindCounter task
**Behaviour change:** none in output. On a read-only server, meeting results are computed once and reflect the store as of startup, the same as embeddings and cluster data.

**Files:**
- Modify: `internal/mcp/server.go:471-510` (`graphSummaryHandler` counting), `:583-591` (summary literal)
- Modify: `internal/mcp/meetings.go:3-17` (imports), `:43-50` (`searchMeetingsHandler`), `:100-111` (`decisionsAboutHandler`), `:381-429` (`buildMeetingViews`)
- Test: `internal/mcp/server_test.go`, `internal/mcp/meetings_test.go`

**Interfaces:**
- Consumes: `store.KindCounter` from Section B: `CountByKind(ctx context.Context) (nodes map[model.NodeKind]int, edges map[model.EdgeKind]int, err error)`. If B changes the signature, change `kindCounts` to match. `GraphStore.Neighbors`; `store.IsReadOnly`.
- Produces: none exported.

- [ ] **Step 1: Write the failing test**

Append to `internal/mcp/server_test.go` (its imports already include `errors`, `model` and `store` after A5):

```go
// kindCountingStore answers counts from an index and fails any full scan.
type kindCountingStore struct{ store.GraphStore }

func (kindCountingStore) CountByKind(context.Context) (map[model.NodeKind]int, map[model.EdgeKind]int, error) {
	return map[model.NodeKind]int{model.KindFile: 1, model.KindSymbol: 3}, map[model.EdgeKind]int{model.EdgeCalls: 2}, nil
}
func (kindCountingStore) AllNodes(context.Context) ([]model.Node, error) {
	return nil, errors.New("AllNodes called")
}
func (kindCountingStore) AllEdges(context.Context) ([]model.Edge, error) {
	return nil, errors.New("AllEdges called")
}

func TestGraphSummaryUsesKindCounter(t *testing.T) {
	out := call(t, graphSummaryHandler(kindCountingStore{openStore(t)}, nil, redactor{}), map[string]any{})
	var got struct {
		NodeCount int `json:"node_count"`
		EdgeCount int `json:"edge_count"`
	}
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out)
	}
	if got.NodeCount != 4 || got.EdgeCount != 2 {
		t.Errorf("graph_summary counts = %d nodes, %d edges; want 4, 2 from CountByKind", got.NodeCount, got.EdgeCount)
	}
}
```

In `internal/mcp/meetings_test.go`, add `"errors"` to the imports and add:

```go
// countingStore counts full node scans, reports a settable read-only flag, and fails AllEdges.
type countingStore struct {
	store.GraphStore
	readOnly bool
	scans    int
}

func (c *countingStore) AllNodes(ctx context.Context) ([]model.Node, error) {
	c.scans++
	return c.GraphStore.AllNodes(ctx)
}
func (c *countingStore) AllEdges(context.Context) ([]model.Edge, error) {
	return nil, errors.New("AllEdges called")
}
func (c *countingStore) ReadOnly() bool { return c.readOnly }

func TestMeetingViewsSkipEdgeScanAndCacheReadOnly(t *testing.T) {
	gs := openStore(t)
	seedLinkedMeeting(t, gs)
	cs := &countingStore{GraphStore: gs}
	args := map[string]any{"topic": "fetchAll"}

	if out := call(t, decisionsAboutHandler(cs), args); !strings.Contains(out, "Cap retries") {
		t.Fatalf("decisions_about without a full edge scan = %s", out)
	}

	cs.readOnly = true
	h := decisionsAboutHandler(cs)
	before := cs.scans
	for range 3 {
		call(t, h, args)
	}
	if got := cs.scans - before; got != 1 {
		t.Errorf("3 calls on a read-only store scanned nodes %d times, want 1", got)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/mcp/ -run 'TestGraphSummaryUsesKindCounter|TestMeetingViewsSkipEdgeScan' -count=1`
Expected: FAIL. `graph_summary` fails with `unmarshal: invalid character 'f'` over the text `failed to load nodes: AllNodes called`, and `decisions_about without a full edge scan = load graph failed: AllEdges called`.

- [ ] **Step 3: Implement**

`internal/mcp/server.go`: add these two helpers above `graphSummaryHandler`:

```go
// kindCounts counts nodes and edges by kind within scope (a source_uri substring, "" = whole graph), returning
// the in-scope node IDs when scoped. An unscoped count uses the store's kind index when it has one.
func kindCounts(ctx context.Context, gs store.GraphStore, scope string) (map[model.NodeKind]int, map[model.EdgeKind]int, map[string]bool, error) {
	if kc, ok := gs.(store.KindCounter); ok && scope == "" {
		byNode, byEdge, err := kc.CountByKind(ctx)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("count by kind: %w", err)
		}
		if byNode == nil {
			byNode = map[model.NodeKind]int{}
		}
		if byEdge == nil {
			byEdge = map[model.EdgeKind]int{}
		}
		return byNode, byEdge, nil, nil
	}
	allNodes, err := gs.AllNodes(ctx)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load nodes: %w", err)
	}
	allEdges, err := gs.AllEdges(ctx)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load edges: %w", err)
	}
	scopeLower := strings.ToLower(scope)
	var inScope map[string]bool // built only when scoped, since every node counts when unscoped
	if scope != "" {
		inScope = make(map[string]bool)
	}
	byNode := make(map[model.NodeKind]int)
	for _, n := range allNodes {
		if scope != "" {
			if !strings.Contains(strings.ToLower(n.SourceURI), scopeLower) {
				continue
			}
			inScope[n.ID] = true
		}
		byNode[n.Kind]++
	}
	byEdge := make(map[model.EdgeKind]int)
	for _, e := range allEdges {
		if scope == "" || (inScope[e.Src] && inScope[e.Dst]) {
			byEdge[e.Kind]++
		}
	}
	return byNode, byEdge, inScope, nil
}

// sumCounts totals a by-kind count map.
func sumCounts[K comparable](m map[K]int) int {
	total := 0
	for _, c := range m {
		total += c
	}
	return total
}
```

In `graphSummaryHandler`, replace lines 476-510 (from `allNodes, err := gs.AllNodes(ctx)` through the end of the edge-counting loop) with:

```go
		byNodeKind, byEdgeKind, inScope, err := kindCounts(ctx, gs, scope)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("graph summary failed", err), nil
		}
```

and in the returned `summary` literal (lines 583-591), set:

```go
			NodeCount:   sumCounts(byNodeKind),
			EdgeCount:   sumCounts(byEdgeKind),
```

`internal/mcp/meetings.go`: add `"sync"` to the imports. In `searchMeetingsHandler` (line 43) and `decisionsAboutHandler` (line 100), create a cache in the constructor and use it in place of `buildMeetingViews`. Shown for `searchMeetingsHandler`; `decisionsAboutHandler` gets the same two lines:

```go
func searchMeetingsHandler(gs store.GraphStore) mcpsdk.ToolHandlerFunc {
	cache := &meetingViewCache{gs: gs}
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		query := strings.ToLower(strings.TrimSpace(req.GetString("query", "")))
		since, sinceOK := parseSince(req.GetString("since", ""))
		limit := clampInt(req.GetInt("limit", 10), 1, 100)
		includePII := req.GetBool("include_pii", false)

		views, err := cache.get(ctx)
		// ... rest unchanged
```

Replace the head of `buildMeetingViews` (lines 381-429: the doc comment through the end of the DECIDED_IN index loop). The code from `// Group decisions by meeting.` (line 431) to the end of the function is unchanged:

```go
// meetingViewCache memoises buildMeetingViews on a read-only store, whose snapshot is fixed for the server's
// life. A writable store is rebuilt per call, since submit_meeting_links changes it.
type meetingViewCache struct {
	gs     store.GraphStore
	mu     sync.Mutex
	views  []meetingView
	loaded bool
}

func (c *meetingViewCache) get(ctx context.Context) ([]meetingView, error) {
	if !store.IsReadOnly(c.gs) {
		return buildMeetingViews(ctx, c.gs)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.loaded {
		views, err := buildMeetingViews(ctx, c.gs)
		if err != nil {
			return nil, err
		}
		c.views, c.loaded = views, true
	}
	return c.views, nil
}

// buildMeetingViews assembles one view per meeting node, newest first, with its decisions and the symbols they
// govern. It reads each decision's DECIDED_IN edges from the adjacency index instead of decoding every edge.
func buildMeetingViews(ctx context.Context, gs store.GraphStore) ([]meetingView, error) {
	nodes, err := store.NodesMatching(ctx, gs, store.NodeFilter{Kinds: []model.NodeKind{model.KindMeeting, model.KindDecision}})
	if err != nil {
		return nil, err
	}

	// Index decision → (governed symbol labels, meeting id).
	decGoverns := map[string][]string{}
	decMeeting := map[string]string{}
	for _, d := range nodes {
		if d.Kind != model.KindDecision {
			continue
		}
		sub, err := gs.Neighbors(ctx, d.ID, 1, []model.EdgeKind{model.EdgeDecidedIn})
		if err != nil {
			return nil, err
		}
		byID := make(map[string]model.Node, len(sub.Nodes))
		for _, n := range sub.Nodes {
			byID[n.ID] = n
		}
		for _, e := range sub.Edges {
			dst, ok := byID[e.Dst]
			if e.Src != d.ID || !ok {
				continue
			}
			if dst.Kind == model.KindMeeting {
				decMeeting[d.ID] = dst.ID
			} else {
				decGoverns[d.ID] = append(decGoverns[d.ID], dst.Label)
			}
		}
	}

	// Group decisions by meeting.
	// ... unchanged from line 431 to the end of the function
```

On a writable server, `Neighbors` still loads each decision's meeting node, so each meeting's transcript is decoded once per decision it holds. That costs far less than decoding every edge in a code graph. Removing it would need an edge-keys-only adjacency capability, which is Section B's territory.

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/mcp/... -count=1; go vet ./internal/mcp/...`
Expected: PASS. The existing `TestSearchMeetingsHandler`, `TestDecisionsAboutHandler` and `TestGraphSummaryTopNodesByBetweenness` still pass.

- [ ] **Step 5: Docs**
- `docs/MCP.md:558-563` (`search_meetings` intro): add "On a read-only server, meeting results are computed once and reflect the store as of startup; restart `chiton serve` after ingesting meetings, as for embeddings and cluster data."
- `CHANGELOG.md` `### Changed`: "- `graph_summary` counts kinds from the store's kind index, and `search_meetings`/`decisions_about` read DECIDED_IN edges per decision (memoised on a read-only server) instead of decoding the whole graph on every call."

- [ ] **Step 6: Commit**

```powershell
git add internal/mcp/server.go internal/mcp/server_test.go internal/mcp/meetings.go internal/mcp/meetings_test.go docs/MCP.md CHANGELOG.md
git commit -m "perf(mcp): count kinds from the index and read DECIDED_IN per decision"
```

### Task A7: Bind submit_extraction to an extraction task and stop it overwriting concepts

**Finding:** `Submit` (`internal/extract/subagent/service.go:130-197`) accepts any existing node ID as `task_id` (`:138-144`) and stamps `extraction_status=done` on it (`:188-193`). On a meeting node, that hides the meeting from `pending_meeting_links`, because the two flows share the attr (`internal/extract/meeting/linkage.go:18-20,104`). The concept nodes it builds have global IDs (`concept://<slug>`, `:161-162`), so upserting them rewrites another task's concept. Against the current code, a second task replaced "JSON Web Token" / INFERRED 0.9 / `extracted_from=d1` with "attacker text" / AMBIGUOUS 0.1 / `d2`. `verifyTaskBinding` (`internal/analyze/llm.go:348-362`) re-derives a content hash, but it doesn't fit here: an extraction task ID *is* the source node ID. Meeting linkage already checks the kind (`linkage.go:170-172`).
**Priority:** P3   **Effort:** S   **Depends on:** none
**Behaviour change:** `submit_extraction` and `chiton extract submit` reject a `task_id` that is not a doc_section with text or an image. Concepts that already exist are linked to the new source but never rewritten: the first writer's description, provenance and confidence stand, and `concepts_added` counts only new concepts. Re-submitting the same task is still a no-op. **DECISION NEEDED:** first writer wins (recommended). Alternative: let a later submission raise an existing concept's confidence and provenance, but never lower them or change the description.

**Files:**
- Modify: `internal/extract/subagent/tasks.go:1-3` (add `taskFor`)
- Modify: `internal/extract/subagent/service.go:5-11` (imports), `:60-95` (`Pending`), `:130-180` (`Submit`); add `dropExisting`
- Modify: `internal/mcp/semantic.go:131-134` (`task_id` description)
- Test: `internal/extract/subagent/service_test.go`

**Interfaces:**
- Consumes: `store.GetNodes(ctx, gs, ids []string) ([]model.Node, error)`
- Produces: none exported.

- [ ] **Step 1: Write the failing test**

Append to `internal/extract/subagent/service_test.go`:

```go
// A task_id naming a symbol or meeting must not mark it extracted (which would also hide the meeting from
// pending_meeting_links, since both flows share extraction_status) or hang concepts off it.
func TestSubmitRejectsNodesThatAreNotExtractionTasks(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{
		{ID: "sym1", Kind: model.KindSymbol, Label: "fetchAll", SourceURI: "file://a.go"},
		{ID: "m1", Kind: model.KindMeeting, Label: "Sync", SourceURI: "meeting://sync", Attrs: map[string]any{"text": "x"}},
		docSection("empty", "Empty", ""),
	})
	svc := New(gs, nil, 0.5)
	for _, id := range []string{"sym1", "m1", "empty"} {
		if _, err := svc.Submit(ctx, SubmitResult{TaskID: id, Concepts: []ConceptResult{{Label: "X", Confidence: 0.9}}}); err == nil {
			t.Errorf("Submit(task_id=%s) succeeded; want it rejected", id)
		}
		if n, _ := gs.GetNode(ctx, id); model.AttrString(n.Attrs, "extraction_status") != "" {
			t.Errorf("%s was marked extracted", id)
		}
	}
}

// Concepts are global, so a later task must not rewrite one an earlier task created.
func TestSubmitKeepsExistingConcept(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{docSection("d1", "Auth", "we use JWTs"), docSection("d2", "Notes", "tokens")})
	svc := New(gs, nil, 0.5)
	if _, err := svc.Submit(ctx, SubmitResult{TaskID: "d1",
		Concepts: []ConceptResult{{Label: "JWT", Description: "JSON Web Token", Confidence: 0.9}}}); err != nil {
		t.Fatal(err)
	}
	rep, err := svc.Submit(ctx, SubmitResult{TaskID: "d2",
		Concepts: []ConceptResult{{Label: "JWT", Description: "attacker text", Confidence: 0.1}}})
	if err != nil {
		t.Fatal(err)
	}
	if rep.ConceptsAdded != 0 {
		t.Errorf("ConceptsAdded = %d for an existing concept, want 0", rep.ConceptsAdded)
	}
	id := model.NewNodeID("concept://jwt", "JWT")
	n, _ := gs.GetNode(ctx, id)
	if n == nil || model.AttrString(n.Attrs, "description") != "JSON Web Token" ||
		model.AttrString(n.Attrs, "extracted_from") != "d1" || n.Provenance != model.ProvInferred {
		t.Fatalf("existing concept was rewritten: %+v", n)
	}
	edges, _ := gs.AllEdges(ctx)
	if !slices.ContainsFunc(edges, func(e model.Edge) bool { return e.Src == "d2" && e.Dst == id }) {
		t.Error("d2 was not linked to the existing concept")
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/extract/subagent/ -run 'NotExtractionTasks|KeepsExistingConcept' -count=1`
Expected: FAIL with `Submit(task_id=sym1) succeeded; want it rejected` and `sym1 was marked extracted` (the same for `m1` and `empty`), then `ConceptsAdded = 1 for an existing concept, want 0` and `existing concept was rewritten: ... description:attacker text extracted_from:d2] Provenance:AMBIGUOUS`.

- [ ] **Step 3: Implement**

`internal/extract/subagent/tasks.go`: replace the single-line import with a block and add `taskFor`:

```go
import (
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// taskFor returns n's extraction task when n is a doc_section with text or an image matching kind ("" = any).
// Pending and Submit share it, so a submit can only name a node Pending would issue a task for.
func taskFor(n model.Node, kind string) (TaskSpec, bool) {
	switch n.Kind {
	case model.KindDocSection:
		text := model.AttrString(n.Attrs, "text")
		if (kind != "" && kind != KindDocConcepts) || strings.TrimSpace(text) == "" {
			return TaskSpec{}, false
		}
		return docTask(n, text), true
	case model.KindImage:
		if kind != "" && kind != KindImageDescription {
			return TaskSpec{}, false
		}
		return imageTask(n), true
	}
	return TaskSpec{}, false
}
```

`internal/extract/subagent/service.go`: add `"slices"` to the imports. In `Pending`, replace the loop body's `switch` (lines 70-88) so the loop reads:

```go
	var specs []TaskSpec
	for _, n := range nodes {
		if model.AttrString(n.Attrs, "extraction_status") == extractionStatusDone {
			continue
		}
		spec, ok := taskFor(n, kind)
		if !ok {
			continue
		}
		specs = append(specs, spec)
		if limit > 0 && len(specs) >= limit {
			break
		}
	}
	return specs, nil
```

In `Submit`, replace lines 142-180 (from the `src == nil` check through the end of the concept loop) with the code below. The code that follows, starting at the embedding comment on line 182, is unchanged:

```go
	if src == nil {
		return SubmitReport{}, fmt.Errorf("submit: no node with id %q (call pending_extractions first)", r.TaskID)
	}
	// Bind task_id to a node Pending issues tasks for, so a submit can't mark or link arbitrary nodes.
	if _, ok := taskFor(*src, ""); !ok {
		return SubmitReport{}, fmt.Errorf("submit: node %q is a %s, not an extraction task (want a doc_section with text, or an image)", r.TaskID, src.Kind)
	}

	now := s.now()
	rep := SubmitReport{TaskID: r.TaskID}

	// Build concept nodes, keyed by label for edge resolution.
	conceptID := map[string]string{}
	var nodes []model.Node
	for _, c := range r.Concepts {
		label := strings.TrimSpace(c.Label)
		if label == "" {
			continue
		}
		prov, conf := model.ResolveProvenance(c.Confidence, s.threshold)
		uri := "concept://" + model.Slug(label, "concept")
		id := model.NewNodeID(uri, label)
		conceptID[label] = id
		nodes = append(nodes, model.Node{
			ID:        id,
			Kind:      model.KindConcept,
			Label:     label,
			SourceURI: uri,
			Attrs: map[string]any{
				"description":    strings.TrimSpace(c.Description),
				"extracted_from": r.TaskID,
			},
			Provenance: prov,
			Confidence: conf,
			CreatedAt:  now,
			UpdatedAt:  now,
		})
	}
	// conceptID keeps the dropped IDs, so this task's edges still link to concepts other tasks created.
	if nodes, err = s.dropExisting(ctx, nodes); err != nil {
		return SubmitReport{}, err
	}
	rep.ConceptsAdded = len(nodes)
	for _, n := range nodes {
		if n.Provenance == model.ProvAmbiguous {
			rep.Ambiguous++
		}
	}
```

Add after `Submit`:

```go
// dropExisting removes concepts already in the store. Concept IDs are global (concept://<slug>), so a task may
// link to another task's concept but never rewrite its description, provenance, or confidence.
func (s *Service) dropExisting(ctx context.Context, concepts []model.Node) ([]model.Node, error) {
	if len(concepts) == 0 {
		return concepts, nil
	}
	ids := make([]string, len(concepts))
	for i, c := range concepts {
		ids[i] = c.ID
	}
	existing, err := store.GetNodes(ctx, s.store, ids)
	if err != nil {
		return nil, fmt.Errorf("load existing concepts: %w", err)
	}
	have := make(map[string]bool, len(existing))
	for _, n := range existing {
		have[n.ID] = true
	}
	return slices.DeleteFunc(concepts, func(n model.Node) bool { return have[n.ID] }), nil
}
```

`internal/mcp/semantic.go:133`: change the `task_id` description to `"The task spec's id: a doc_section or image node ID from pending_extractions. Concepts that already exist are linked, never overwritten."`

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/extract/subagent/... ./internal/mcp/... -count=1; go vet ./internal/extract/subagent/... ./internal/mcp/...`
Expected: PASS. The existing `TestSubmitIdempotent`, `TestSubmitWritesConceptsAndMarksDone` and `TestSubmitDescribesWhenAllEdgesUnresolved` still pass.

- [ ] **Step 5: Docs**
- `docs/MCP.md:783-813` (`submit_extraction`): change the `task_id` row to "a doc_section or image node ID from `pending_extractions`; other nodes are rejected". Add: "A concept that already exists (concept IDs are global by label) is linked to this source but not rewritten; `concepts_added` counts only new concepts."
- `CHANGELOG.md` `### Security`: "- `submit_extraction` rejects a `task_id` that isn't a doc_section or image and no longer overwrites concepts other tasks created, so a forged submit can't mark meetings as linked or rewrite concept descriptions."

- [ ] **Step 6: Commit**

```powershell
git add internal/extract/subagent internal/mcp/semantic.go docs/MCP.md CHANGELOG.md
git commit -m "fix(subagent): bind submit_extraction to extraction tasks and keep existing concepts"
```

### Task A8: Guard GitCommit's ref against option injection

**Finding:** `analyze.GitCommit` runs `git rev-parse <ref>` (`internal/analyze/diff.go:176-184`). `diff_findings` passes MCP-supplied `baseline`/`target` into it (`internal/mcp/analyze.go:31-39,43-44`). Against the current code, `GitCommit(root, "--show-toplevel")` returned the repository's absolute path, and `diff_findings` echoes an unresolved target back ("no baseline found for target %q"). `internal/ingest/git` already has `validateRef` (`git.go:100-110`) plus the `--end-of-options` pattern (`:81,93`).
**Priority:** P3   **Effort:** S   **Depends on:** none
**Behaviour change:** `GitCommit` returns "" for option-like refs and for refs that don't name a commit. It needs git 2.30 or later for `rev-parse --end-of-options`; on older git it returns "", and callers already fall back to the raw ref or "working".

**Files:**
- Modify: `internal/ingest/git/git.go:75,78,90,100-110` (export `ValidateRef`)
- Modify: `internal/analyze/diff.go:3-18` (imports), `:176-184` (`GitCommit`)
- Test: `internal/analyze/diff_test.go`

**Interfaces:**
- Consumes: none
- Produces: `git.ValidateRef(ref string) error` (renamed from `validateRef`; package `github.com/Eric-Fernald/chiton/internal/ingest/git`)

- [ ] **Step 1: Write the failing test**

In `internal/analyze/diff_test.go`, add `"os"` and `"os/exec"` to the imports, then add:

```go
// diff_findings resolves MCP-supplied refs through GitCommit, and an unguarded `git rev-parse --show-toplevel`
// prints the repo path, which the tool then echoes back.
func TestGitCommitRefusesOptionLikeRefs(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	for _, args := range [][]string{{"init", "-q"}, {"commit", "--allow-empty", "-q", "-m", "c"}} {
		cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	for _, ref := range []string{"--show-toplevel", "--git-dir", "--output=x"} {
		if got := GitCommit(root, ref); got != "" {
			t.Errorf("GitCommit(%q) = %q, want \"\"", ref, got)
		}
	}
	if sha := GitCommit(root, "HEAD"); len(sha) != 40 {
		t.Errorf("GitCommit(HEAD) = %q, want a 40-hex SHA", sha)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/analyze/ -run TestGitCommitRefusesOptionLikeRefs -count=1`
Expected: FAIL with `GitCommit("--show-toplevel") = "C:/.../001", want ""`, `GitCommit("--git-dir") = ".git"` and `GitCommit("--output=x") = "--output=x"`.

- [ ] **Step 3: Implement**

`internal/ingest/git/git.go`: rename `validateRef` to `ValidateRef` in its definition (line 102), its comment (line 100: `// ValidateRef rejects option-like refs, ...`), and its callers at lines 75, 78 and 90.

`internal/analyze/diff.go`: add `gitref "github.com/Eric-Fernald/chiton/internal/ingest/git"` to the imports and replace `GitCommit`:

```go
// GitCommit resolves ref to a full commit SHA in repoDir, or "" when git is unavailable or the ref doesn't
// resolve. ref can come from an MCP caller, so an option-like value must never reach git as a flag.
func GitCommit(repoDir, ref string) string {
	if gitref.ValidateRef(ref) != nil {
		return ""
	}
	out, err := exec.Command("git", "-C", repoDir, "rev-parse", "--verify", "--quiet", "--end-of-options", ref+"^{commit}").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/analyze/ ./internal/ingest/git/... ./internal/mcp/... -count=1; go vet ./internal/analyze/ ./internal/ingest/git/...`
Expected: PASS. The existing `TestRejectsOptionLikeRefs` in `internal/ingest/git` still passes.

- [ ] **Step 5: Docs**
- `CHANGELOG.md` `### Security`: "- `diff_findings` baseline/target refs are validated and passed to `git rev-parse --verify --end-of-options`, so an option-like ref such as `--show-toplevel` can no longer leak the repository path."

- [ ] **Step 6: Commit**

```powershell
git add internal/ingest/git/git.go internal/analyze/diff.go internal/analyze/diff_test.go CHANGELOG.md
git commit -m "fix(analyze): validate and --end-of-options the ref GitCommit passes to git"
```

### Task A9: Make stdio the default serve transport (A1b)

**Finding:** A bare `chiton serve` still opens a TCP listener (`cmd/chiton/main.go:909-922`). Every installer and the plugin pass `--stdio` explicitly (`internal/install/mcp_config.go:19`, `.mcp.json:5`), so the HTTP default only helps someone starting a daemon by hand. A socket nobody asked for is exposure even with A1's auth in place.
**Priority:** P1   **Effort:** S   **Depends on:** A1
**Behaviour change:** `chiton serve` with no transport flag speaks MCP over stdio and prints a one-line stderr hint. HTTP/SSE starts only when `--http <addr>` is given, and the precedence is `--stdio` > `--unix` > `--http`. `--stdio` is still accepted, so existing configs are unaffected. Anyone who ran a bare `chiton serve` as a daemon must add `--http 127.0.0.1:7474`. **DECISION NEEDED:** stdio default (recommended). Alternative: keep HTTP as the default and rely on A1's auth.

**Files:**
- Modify: `cmd/chiton/serve_http.go` (add `transportFor`)
- Modify: `cmd/chiton/main.go:860-932` (`serveCmd`: `Long`, switch, flag help)
- Test: `cmd/chiton/serve_http_test.go`

**Interfaces:**
- Consumes: `serveHTTP`, `serveTokenPath` (A1)
- Produces: `transportFor(stdio bool, unixPath string, httpSet bool) string`

- [ ] **Step 1: Write the failing test**

Append to `cmd/chiton/serve_http_test.go`:

```go
func TestTransportForDefaultsToStdio(t *testing.T) {
	for _, tc := range []struct {
		stdio   bool
		unix    string
		httpSet bool
		want    string
	}{
		{false, "", false, "stdio"},
		{true, "/tmp/c.sock", true, "stdio"},
		{false, "/tmp/c.sock", true, "unix"},
		{false, "", true, "http"},
	} {
		if got := transportFor(tc.stdio, tc.unix, tc.httpSet); got != tc.want {
			t.Errorf("transportFor(%v, %q, %v) = %q, want %q", tc.stdio, tc.unix, tc.httpSet, got, tc.want)
		}
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./cmd/chiton/ -run TestTransportForDefaultsToStdio -count=1`
Expected: FAIL — build error: `undefined: transportFor`.

- [ ] **Step 3: Implement**

Append to `cmd/chiton/serve_http.go`:

```go
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
```

In `cmd/chiton/main.go` `serveCmd`, replace A1's switch with:

```go
			// All three transports expose the same tool set.
			switch transportFor(stdio, unixPath, cmd.Flags().Changed("http")) {
			case "unix":
				return serveUnix(cmd, mcpsdk.NewSSEServer(s), unixPath)
			case "http":
				// No TLS, so warn without blocking on a wildcard bind.
				warnIfNonLoopback(cmd.ErrOrStderr(), addr,
					"the MCP server over plain HTTP — its bearer token and the entire graph (code, meetings, emails, PII) would cross the network unencrypted")
				return serveHTTP(cmd, s, addr, serveTokenPath(cfg.GraphPath), allowOrigins)
			default:
				if !stdio {
					fmt.Fprintln(cmd.ErrOrStderr(), "serve: speaking MCP over stdio (the default); pass --http 127.0.0.1:7474 to run the HTTP daemon")
				}
				return mcpsdk.ServeStdio(s)
			}
```

Change the first three lines of `Long` to:

```text
Start the MCP server. stdio is the default; --http or --unix runs a shared
daemon instead (identical tool surface):
  --stdio        assistant-launched (the default; kept for existing configs)
```

In A1's `--http` line of `Long`, change "(default 127.0.0.1:7474; ...)" to "(e.g. 127.0.0.1:7474; ...)". Update the flag help strings:

```go
	cmd.Flags().BoolVar(&stdio, "stdio", false, "serve over stdio (the default; kept for existing configs)")
	cmd.Flags().StringVar(&addr, "http", "127.0.0.1:7474", "serve HTTP/SSE on this address instead of stdio, e.g. 127.0.0.1:7474 (shared daemon / debug)")
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./cmd/chiton/... ./internal/install/... -count=1; go vet ./cmd/chiton/...`
Expected: PASS

- [ ] **Step 5: Docs**
- `docs/MCP.md:20-25` examples: `chiton serve` becomes "# stdio (default)", and `chiton serve --http 127.0.0.1:7474` becomes "# HTTP/SSE daemon". In the flags table, the `--stdio` row reads "default transport" and the `--http` default reads "unset (e.g. `127.0.0.1:7474`)". Lines `:35-36`: "`--stdio` wins; otherwise `--unix` if set; otherwise HTTP/SSE when `--http` is given; otherwise stdio."
- `docs/INSTALL.md:231-232`: the stdio row is the default, and the HTTP row reads "only when `--http <addr>` is given".
- `docs/SECURITY.md:412-423`: change the first table row to `chiton serve --http <addr>`, and replace §5.1's first paragraph with "A bare `chiton serve` speaks stdio and opens no socket; HTTP/SSE starts only with an explicit `--http <addr>`."
- `README.md:219,261-264`: say stdio is the default and HTTP needs `--http <addr>`.
- `CHANGELOG.md` `### Changed`: "- `chiton serve` with no transport flag now speaks stdio instead of starting the HTTP/SSE daemon on 127.0.0.1:7474. Pass `--http 127.0.0.1:7474` for the daemon; `--stdio` is still accepted."

- [ ] **Step 6: Commit**

```powershell
git add cmd/chiton/serve_http.go cmd/chiton/serve_http_test.go cmd/chiton/main.go docs/MCP.md docs/INSTALL.md docs/SECURITY.md README.md CHANGELOG.md
git commit -m "feat(serve): make stdio the default transport; HTTP needs an explicit --http"
```

---

## Section B: Store layer (BadgerDB) reliability and indexing

This section makes the Badger store safe to share between chiton processes, gives it a layout version, and replaces its full scans with key-only indexes. B1 comes first (P1). It adds `chiton.lock` coordination and read-only self-repair, and introduces `badgerOptions`/`openReadOnly`, which later tasks edit. B2 adds the `m:` metadata prefix and the migration runner. B3 builds on B2: the source_uri and kind indexes, the v1→v2 migration, a clean-close seal that keeps the indexes exact, and the exported `store.KindCounter`. B4 and B5 are independent read-path speedups (B4 edits the `prepare` hook from B2). B6 reuses B2's metadata for a store epoch and syncs the store before `cache.json` is saved. B7 sits last because its `Close` edit assumes B3's seal. Apply in the order B1, B2, B3, B4, B5, B6, B7.

### Task B1: Coordinate readers and writers with `chiton.lock`; self-repair a crashed writer's log

**Finding:** On Windows a read-write open beside a read-only `serve` can fail halfway. I reproduced this across two processes. The first writer open succeeded. The second failed in Badger's `vlog.open` (`badger@v4.9.6/value.go:616-621`) while deleting an empty value-log file the reader had mapped ("user-mapped section open"). By then `newMemTable` had already created a 128 MiB `00001.mem`, and from that point every read-only open (`internal/graph/store/badger/store.go:83-91`) failed with "Log truncate required" until a clean read-write open ran. In the other direction, a live writer's preallocated memtable makes every read-only open fail on every platform, which contradicts the help text at `cmd/chiton/main.go:795-797` and `876-880`. Nothing handles `ErrTruncateNeeded`, and `errors.Is` can't: outside debug builds `y.Wrap` formats with `%+v` (`badger@v4.9.6/y/error.go:55-63`). A failed read-only Badger open also leaks its memtable mapping (`db.go:344` shadows `err`, so cleanup is skipped). As a result, "retry read-write in the same process" fails again on Windows, and the log has to be detected before Badger opens.
**Priority:** P1   **Effort:** M   **Depends on:** none
**Behaviour change:** Both directions now fail fast with `graph store is in use by another chiton process` instead of colliding. A reader can't open while a writer holds the store (any platform). On Windows only, a writer can't open while any reader (for example an assistant's `chiton serve`) holds it. Linux/macOS keep today's "writer beside a read-only serve" concurrency. A read-only open now replays a crashed writer's log by itself when no other chiton process holds the store. **DECISION NEEDED:** recommended: fail fast now. Alternative (follow-up): make `serve` release the store when idle so Windows writers can run beside it. Until that exists, on Windows the CLI submit path and `build`/`embed` need the server stopped. The pre-commit gate (`chiton hooks run`, `cmd/chiton/main.go:1183`) opens the store as a writer today; Task D2 removes that dependency (the gate analyses staged content in memory and fails open when the store is busy), so land D2 before B1 or accept that commits block on Windows while `serve` runs until D2 lands.

**Files:**
- Create: `internal/graph/store/badger/lock.go`
- Create: `internal/graph/store/badger/lock_windows.go`
- Create: `internal/graph/store/badger/lock_unix.go`
- Create: `internal/graph/store/badger/lock_other.go`
- Modify: `internal/graph/store/badger/store.go:5-23` (imports), `35-39` (Store), `63-91` (Open*), `104` (Close)
- Modify: `cmd/chiton/main.go:793-797`, `806`, `876-880`, `929`
- Modify: `go.mod` (`golang.org/x/sys` becomes a direct requirement)
- Test: `internal/graph/store/badger/lock_test.go`

**Interfaces:**
- Consumes: none
- Produces: `badger.ErrStoreBusy error` (exported sentinel; wrap-checked with `errors.Is`); package-internal `acquireLock(dir string, exclusive bool) (*storeLock, error)`, `errNeedsWriterPass`, `openReadOnly(path string, opts Options) (*Store, error)`, `badgerOptions(path string, opts Options, readOnly bool) badgerdb.Options`, `isTruncateNeeded(err error) bool`

I chose an OS advisory lock over the pidfile registry the audit suggested. The kernel drops it when the process dies, so there is no stale-pid or pid-reuse handling. By contrast, `os.FindProcess` on Windows succeeds for an exited process whose handle is still referenced, and Unix would need `Signal(0)`. A probe confirmed three things: `LockFileEx` shared and exclusive locks conflict between handles in the same process, Badger ignores a locked `chiton.lock` inside its directory, and a read-write open replays and deletes a zero-filled `.mem`.

- [ ] **Step 1: Write the failing test**

```go
// internal/graph/store/badger/lock_test.go
package badger

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	badgerdb "github.com/dgraph-io/badger/v4"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// seededStore creates a store holding node "a", closes it, and returns its directory.
func seededStore(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "graph.badger")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertNodes(context.Background(), []model.Node{{ID: "a", Label: "a"}}); err != nil {
		t.Fatal(err)
	}
	if err := gs.Close(); err != nil {
		t.Fatal(err)
	}
	return dir
}

// memLogs lists dir's Badger memtable logs. A cleanly closed store has none.
func memLogs(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var logs []string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".mem") {
			logs = append(logs, e.Name())
		}
	}
	return logs
}

func TestReaderRefusedWhileWriterOpen(t *testing.T) {
	dir := seededStore(t)
	rw, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer rw.Close()
	ro, err := OpenReadOnly(dir)
	if !errors.Is(err, ErrStoreBusy) {
		if ro != nil {
			ro.Close()
		}
		t.Fatalf("OpenReadOnly beside a writer = %v, want ErrStoreBusy", err)
	}
}

func TestWriterRefusedWhileReaderOpenOnWindows(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("only Windows refuses a writer beside a reader")
	}
	dir := seededStore(t)
	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	rw, err := Open(dir)
	if !errors.Is(err, ErrStoreBusy) {
		if rw != nil {
			rw.Close()
		}
		ro.Close()
		t.Fatalf("Open beside a reader = %v, want ErrStoreBusy", err)
	}
	if logs := memLogs(t, dir); len(logs) != 0 {
		t.Errorf("refused writer left memtable logs %v", logs)
	}
	if err := ro.Close(); err != nil {
		t.Fatal(err)
	}
	rw, err = Open(dir)
	if err != nil {
		t.Fatalf("Open after the reader closed: %v", err)
	}
	rw.Close()
}

func TestWriterOpensBesideReaderOffWindows(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows refuses a writer beside a reader")
	}
	dir := seededStore(t)
	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ro.Close()
	rw, err := Open(dir)
	if err != nil {
		t.Fatalf("Open beside a reader = %v; a Unix reader keeps its snapshot, so a writer may open", err)
	}
	rw.Close()
}

func TestReadOnlyOpenReplaysCrashedWriterLog(t *testing.T) {
	dir := seededStore(t)
	// A crashed writer leaves a preallocated, mostly zero memtable log that a read-only Badger open rejects.
	if err := os.WriteFile(filepath.Join(dir, "00099.mem"), make([]byte, 1<<20), 0o600); err != nil {
		t.Fatal(err)
	}
	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatalf("OpenReadOnly after a crashed writer: %v", err)
	}
	defer ro.Close()
	if n, err := ro.GetNode(context.Background(), "a"); err != nil || n == nil {
		t.Fatalf("GetNode(a) = %v, %v; want the node", n, err)
	}
	if logs := memLogs(t, dir); len(logs) != 0 {
		t.Errorf("memtable logs %v survived the repair", logs)
	}
}

func TestIsTruncateNeeded(t *testing.T) {
	wrapped := fmt.Errorf("while opening memtables err: %+v", badgerdb.ErrTruncateNeeded)
	if !isTruncateNeeded(wrapped) {
		t.Errorf("isTruncateNeeded(%v) = false", wrapped)
	}
	if isTruncateNeeded(errors.New("other")) || isTruncateNeeded(nil) {
		t.Error("isTruncateNeeded matched an unrelated error")
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/graph/store/badger/ -run "TestReaderRefused|TestWriterRefused|TestWriterOpensBeside|TestReadOnlyOpenReplays|TestIsTruncateNeeded" -count=1`
Expected: FAIL. The build fails with `undefined: ErrStoreBusy` and `undefined: isTruncateNeeded`. With those names stubbed, the behavioural failures are: `OpenReadOnly beside a writer = ... Log truncate required ..., want ErrStoreBusy`; on Windows, `Open beside a reader = <nil>, want ErrStoreBusy`; and `OpenReadOnly after a crashed writer: ... Log truncate required`.

- [ ] **Step 3: Implement**

Create `internal/graph/store/badger/lock.go`:

```go
package badger

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ErrStoreBusy reports that another chiton process holds the store in a mode that conflicts with this open.
var ErrStoreBusy = errors.New("graph store is in use by another chiton process")

// errLocked is lockFile's error when another handle holds a conflicting lock.
var errLocked = errors.New("store lock held")

// lockName is the advisory lock file in the store directory: writers lock it exclusive, readers shared.
const lockName = "chiton.lock"

// storeLock is a held lock on a store's chiton.lock. The OS drops it if the process dies.
type storeLock struct{ f *os.File }

// acquireLock locks dir's chiton.lock without blocking. A reader that can't create the file (a missing or
// read-only directory) proceeds unlocked and leaves the error to Badger.
func acquireLock(dir string, exclusive bool) (*storeLock, error) {
	f, err := os.OpenFile(filepath.Join(dir, lockName), os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		if !exclusive {
			return nil, nil
		}
		return nil, fmt.Errorf("open store lock: %w", err)
	}
	if err := lockFile(f, exclusive); err != nil {
		f.Close()
		if errors.Is(err, errLocked) {
			return nil, busyError(dir, exclusive)
		}
		return nil, fmt.Errorf("lock store: %w", err)
	}
	return &storeLock{f: f}, nil
}

// busyError names the kind of process that is in the way.
func busyError(dir string, writer bool) error {
	if writer {
		return fmt.Errorf("%w: %s is open in another process (a running 'chiton serve', 'sync watch', or other command?); stop it and retry",
			ErrStoreBusy, dir)
	}
	return fmt.Errorf("%w: %s is being written ('chiton sync watch', 'build', or 'serve --writable'?); retry when it finishes",
		ErrStoreBusy, dir)
}

// release unlocks and closes the lock file. It is nil-safe for readers that proceeded unlocked.
func (l *storeLock) release() error {
	if l == nil || l.f == nil {
		return nil
	}
	err := errors.Join(unlockFile(l.f), l.f.Close())
	l.f = nil
	return err
}

// hasWriteLog reports whether dir holds a Badger memtable log. A clean close deletes them, so one found under
// the lock means a writer crashed, and a read-only open can't replay it.
func hasWriteLog(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".mem") {
			return true
		}
	}
	return false
}
```

Create `internal/graph/store/badger/lock_windows.go`:

```go
//go:build windows

package badger

import (
	"errors"
	"os"

	"golang.org/x/sys/windows"
)

// holdReaderLock keeps a reader's shared lock until Close: Windows can't truncate or delete a file another
// process has mapped, so a writer opening beside a reader fails midway and strands a 128 MiB memtable log.
const holdReaderLock = true

func lockFile(f *os.File, exclusive bool) error {
	flags := uint32(windows.LOCKFILE_FAIL_IMMEDIATELY)
	if exclusive {
		flags |= windows.LOCKFILE_EXCLUSIVE_LOCK
	}
	err := windows.LockFileEx(windows.Handle(f.Fd()), flags, 0, 1, 0, new(windows.Overlapped))
	if errors.Is(err, windows.ERROR_LOCK_VIOLATION) {
		return errLocked
	}
	return err
}

func unlockFile(f *os.File) error {
	return windows.UnlockFileEx(windows.Handle(f.Fd()), 0, 1, 0, new(windows.Overlapped))
}
```

Create `internal/graph/store/badger/lock_unix.go` (the same constraint Badger's `dir_unix.go` uses for `unix.Flock`):

```go
//go:build !windows && !plan9 && !js && !wasip1 && !aix

package badger

import (
	"errors"
	"os"

	"golang.org/x/sys/unix"
)

// holdReaderLock is false: a Unix reader keeps its mapped snapshot while a writer opens, so it holds the
// shared lock only during its own open, which keeps a writer from starting mid-open.
const holdReaderLock = false

func lockFile(f *os.File, exclusive bool) error {
	how := unix.LOCK_SH | unix.LOCK_NB
	if exclusive {
		how = unix.LOCK_EX | unix.LOCK_NB
	}
	err := unix.Flock(int(f.Fd()), how)
	if errors.Is(err, unix.EWOULDBLOCK) {
		return errLocked
	}
	return err
}

func unlockFile(f *os.File) error { return unix.Flock(int(f.Fd()), unix.LOCK_UN) }
```

Create `internal/graph/store/badger/lock_other.go`:

```go
//go:build plan9 || js || wasip1 || aix

package badger

import "os"

// holdReaderLock is false and locking is a no-op where chiton has no advisory file lock.
const holdReaderLock = false

func lockFile(*os.File, bool) error { return nil }

func unlockFile(*os.File) error { return nil }
```

In `internal/graph/store/badger/store.go`, add `"os"` to the imports. Then replace the `Store` struct (35-39), everything from `// Open opens or creates` through `OpenReadOnlyWith` (63-91), and `Close` (104) with:

```go
// Store is a BadgerDB-backed GraphStore.
type Store struct {
	db       *badgerdb.DB
	readOnly bool
	lock     *storeLock // chiton.lock; nil for a reader once holdReaderLock lets it go
}
```

```go
// Open opens or creates a BadgerDB store at path with read-write access.
func Open(path string) (*Store, error) { return OpenWith(path, Options{}) }

// OpenReadOnly opens an existing store read-only. Readers never block each other; see OpenReadOnlyWith.
func OpenReadOnly(path string) (*Store, error) { return OpenReadOnlyWith(path, Options{}) }

// OpenWith opens (or creates) the store with explicit options, enabling Badger encryption when
// opts.EncryptionKey is set. It returns ErrStoreBusy while another chiton process holds the store.
func OpenWith(path string, opts Options) (*Store, error) {
	if err := os.MkdirAll(path, 0o700); err != nil {
		return nil, fmt.Errorf("open badger at %s: %w", path, err)
	}
	lock, err := acquireLock(path, true)
	if err != nil {
		return nil, err
	}
	db, err := badgerdb.Open(badgerOptions(path, opts, false))
	if err != nil {
		lock.release()
		return nil, fmt.Errorf("open badger at %s: %w", path, err)
	}
	return &Store{db: db, lock: lock}, nil
}

// OpenReadOnlyWith is the read-only variant of OpenWith. When a crashed writer left a log that a read-only open
// can't replay, it replays the log with one read-write open and close, then retries.
func OpenReadOnlyWith(path string, opts Options) (*Store, error) {
	s, err := openReadOnly(path, opts)
	if !errors.Is(err, errNeedsWriterPass) {
		return s, err
	}
	rw, rwErr := OpenWith(path, opts)
	if rwErr != nil {
		return nil, fmt.Errorf("%v: %w", err, rwErr)
	}
	if err := rw.Close(); err != nil {
		return nil, err
	}
	return openReadOnly(path, opts)
}

// errNeedsWriterPass marks a read-only open that one read-write open and close would fix.
var errNeedsWriterPass = errors.New("graph store needs a one-time read-write pass")

// openReadOnly opens path read-only under a shared chiton.lock, kept until Close only where holdReaderLock.
func openReadOnly(path string, opts Options) (*Store, error) {
	lock, err := acquireLock(path, false)
	if err != nil {
		return nil, err
	}
	// Check before Badger opens: its failed read-only open leaks the log's mapping, which blocks repair on Windows.
	if hasWriteLog(path) {
		lock.release()
		return nil, fmt.Errorf("%w (a writer did not close cleanly)", errNeedsWriterPass)
	}
	db, err := badgerdb.Open(badgerOptions(path, opts, true))
	if err != nil {
		lock.release()
		if isTruncateNeeded(err) {
			return nil, fmt.Errorf("open badger at %s: a writer started or crashed during the open; retry: %w", path, err)
		}
		return nil, fmt.Errorf("open badger at %s: %w", path, err)
	}
	if !holdReaderLock {
		lock.release()
		lock = nil
	}
	return &Store{db: db, readOnly: true, lock: lock}, nil
}

// badgerOptions returns the Badger options for path. Read-only opens bypass Badger's directory lock, which
// Windows can't take shared; chiton.lock coordinates readers and writers instead.
func badgerOptions(path string, opts Options, readOnly bool) badgerdb.Options {
	bopts := badgerdb.DefaultOptions(path).WithLogger(nil)
	if readOnly {
		bopts = bopts.WithReadOnly(true).WithBypassLockGuard(true)
	}
	return applyEncryption(bopts, opts.EncryptionKey)
}

// isTruncateNeeded reports Badger's "Log truncate required" error. Outside debug builds Badger wraps it with
// %+v, not %w, so errors.Is can't see it.
func isTruncateNeeded(err error) bool {
	return err != nil && strings.Contains(err.Error(), badgerdb.ErrTruncateNeeded.Error())
}
```

```go
func (s *Store) Close() error { return errors.Join(s.db.Close(), s.lock.release()) }
```

Promote `golang.org/x/sys` to a direct requirement (it is already in the module graph at v0.47.0, so nothing is downloaded):

```powershell
go mod tidy
git diff go.mod   # expect only golang.org/x/sys v0.47.0 moving to the direct require block
```

In `cmd/chiton/main.go`, replace the cluster `Long` (793-797) with:

```go
		Long: `Run Leiden community detection and betweenness metrics on the graph.

Results are written to a JSON sidecar file (chiton-out/cluster.json). The
store opens read-only, so this runs beside a read-only 'chiton serve', but not
while a writer ('sync watch', 'build', 'serve --writable') holds the store.
Restart chiton serve after clustering to load the new results.`,
```

Replace the comment at 806 with `// Open read-only so this runs beside a read-only chiton serve.`. In the serve `Long`, replace the paragraph at 876-880 (from `By default the store opens read-only` to `write path.`) with:

```
By default the store opens read-only, and other read-only commands (query,
cluster, search, report) run beside it; the read tools (semantic_search,
pending_extractions) work in this mode. Writers (build, embed, sync, analyze,
the CLI submit commands) may run beside a read-only server on Linux/macOS,
which keeps serving its startup snapshot until restarted; on Windows they fail
fast with "in use by another chiton process" until the server exits. The
server itself fails fast while a writer holds the store. Pass --writable to
allow the submit_* write tools over MCP; the server then holds the store
exclusively. The 'chiton extract submit' CLI is the alternative write path.
```

Replace the `--writable` flag usage at 929 with `"open the store read-write so the submit_* tools work over MCP (no other chiton process can open the store meanwhile)"`.

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/graph/store/... -count=1; go vet ./internal/graph/store/...; $env:GOOS='linux'; go vet ./internal/graph/store/badger/; $env:GOOS='darwin'; go vet ./internal/graph/store/badger/; Remove-Item Env:GOOS; go build ./...; go test ./... -count=1`
Expected: PASS. The GOOS vets compile `lock_unix.go`. `TestWriterOpensBesideReaderOffWindows` is skipped on Windows and `TestWriterRefusedWhileReaderOpenOnWindows` is skipped elsewhere.

- [ ] **Step 5: Docs**
- `README.md:258-259`: replace "The default read-only server lets `build`, `embed`, and `cluster` run concurrently." with "The default read-only server shares the store with other read-only commands (`query`, `cluster`, `search`). On Linux/macOS, writers such as `build` and `embed` can also run beside it (restart the server to see their changes). On Windows they fail fast with `in use by another chiton process` until the server exits, so submit through `chiton serve --writable` there."
- `docs/INSTALL.md:239` (`--writable` row): replace the second sentence with "The server then holds the store exclusively: no other chiton process (`build`, `embed`, `cluster`, `query`) can open it until the server exits."
- `docs/INSTALL.md:352-358`: replace "so `chiton build`, `embed`, and `cluster` can run concurrently" with "so other read-only commands can run beside it (and, on Linux/macOS, writers such as `build`/`embed`; on Windows a writer fails fast while the server runs)". Replace the `--writable` bullet's parenthetical with "(the server then holds the store exclusively)". After the CLI bullet, add: "On Windows the CLI path needs the read-only server stopped first."
- `docs/MCP.md:32`: set the `--writable` meaning to "Open the store read-write so the write tools work over MCP. The server then holds the store exclusively; no other chiton process can open it until it exits." `docs/MCP.md:40-43`: replace the concurrency sentence the same way as README, and the `--writable` parenthetical with "(which holds the store exclusively)".
- `CLAUDE.md:152-153`: append "On Windows, stop serve before `chiton embed` or any other write command: a writer can't open the store beside a running server."
- `skills/setup/SKILL.md:154`: replace the `Log truncate required` row with two rows: `| graph store is in use by another chiton process | a writer (sync watch, build, serve --writable) holds the store, or on Windows a running chiton serve blocks a writer | stop the other process (or the assistant's server) and retry |` and `| needs a one-time read-write pass | a chiton writer crashed, or the store predates this chiton | close other chiton processes and retry; the next open repairs it |`.
- `skills/subagent-tasks/SKILL.md:72-73`: replace the sentence with "If `chiton serve --writable` or a CLI submit fails with `in use by another chiton process`, another chiton process has the store open. Stop it (`sync watch`, `build`, `embed`, or, on Windows, a read-only server), then retry."
- `CHANGELOG.md` under `### Fixed`: "**The graph store no longer wedges with "Log truncate required".** On Windows, a writer opened while `chiton serve` had the store mapped could fail halfway and strand a 128 MiB memtable log that broke every later read-only open. chiton processes now coordinate on `chiton.lock` in the store directory and fail fast with `graph store is in use by another chiton process` instead of colliding. On Windows a writer is refused while any reader is open; on Linux/macOS writers may still open beside a read-only server. A read-only open now replays a crashed writer's log itself when nothing else holds the store."

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/store/badger/lock.go internal/graph/store/badger/lock_windows.go internal/graph/store/badger/lock_unix.go internal/graph/store/badger/lock_other.go internal/graph/store/badger/lock_test.go internal/graph/store/badger/store.go cmd/chiton/main.go go.mod README.md docs/INSTALL.md docs/MCP.md CLAUDE.md skills/setup/SKILL.md skills/subagent-tasks/SKILL.md CHANGELOG.md
git commit -m "fix(store): coordinate chiton processes on chiton.lock and self-repair a crashed writer's log"
```

### Task B2: Store layout version and migration runner

**Finding:** `.chiton/schema_version` is written by `init` (`cmd/chiton/main.go:238`, `281`) and never read anywhere (grep finds only those two lines and docs), and it describes the `.chiton/` scaffold rather than the store. The Badger store has no version key (`internal/graph/store/badger/store.go:27-33`), so a key-layout change (B3) can't be detected, migrated, or refused by an older or newer binary.
**Priority:** P1   **Effort:** M   **Depends on:** B1
**Behaviour change:** Opening a store stamped by a newer chiton fails with `ErrSchemaTooNew` ("upgrade chiton") instead of misreading it. A read-only open of an older layout performs the one-time upgrade through B1's read-write pass. **DECISION NEEDED:** recommended: auto-upgrade, including from `chiton serve`. Alternative: refuse read-only opens of old stores with "run `chiton sync once` to upgrade". `.chiton/schema_version` is left as-is: it is an init scaffold marker, not the store's version.

**Files:**
- Create: `internal/graph/store/badger/schema.go`
- Modify: `internal/graph/store/badger/store.go:25-26` (prefix comment), `OpenWith`/`openReadOnly` (from B1)
- Modify: `internal/graph/store/badger/deletion_test.go:147-158` (skip metadata keys)
- Test: `internal/graph/store/badger/schema_test.go`

**Interfaces:**
- Consumes: B1's `errNeedsWriterPass`, `openReadOnly`, `badgerOptions`, `storeLock.release`
- Produces: `badger.ErrSchemaTooNew error`; package-internal `pfxMeta = "m:"`, `getMeta(txn *badgerdb.Txn, key string) (string, error)`, `setMeta(db *badgerdb.DB, key, val string) error`, `schemaVersion`, `schemaFresh`, `migrations map[int]func(*badgerdb.DB) error`, `prepareSchema(db *badgerdb.DB, readOnly bool) error`, `migrate(db *badgerdb.DB, from, to int, steps map[int]func(*badgerdb.DB) error) error`, `(*Store).prepare() error`, `(*Store).abandon()`

- [ ] **Step 1: Write the failing test**

```go
// internal/graph/store/badger/schema_test.go
package badger

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	badgerdb "github.com/dgraph-io/badger/v4"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// schemaOf returns db's m:schema stamp.
func schemaOf(t *testing.T, db *badgerdb.DB) string {
	t.Helper()
	var v string
	if err := db.View(func(txn *badgerdb.Txn) (err error) {
		v, err = getMeta(txn, metaSchema)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	return v
}

func TestFreshStoreStampedWithSchema(t *testing.T) {
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	if got := schemaOf(t, gs.db); got != strconv.Itoa(schemaVersion) {
		t.Fatalf("fresh store schema = %q, want %d", got, schemaVersion)
	}
}

func TestStoredSchemaOfUnstampedStores(t *testing.T) {
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	if err := gs.db.Update(func(txn *badgerdb.Txn) error { return txn.Delete([]byte(metaSchema)) }); err != nil {
		t.Fatal(err)
	}
	if v, err := storedSchema(gs.db); err != nil || v != schemaFresh {
		t.Fatalf("unstamped store without nodes = v%d, %v; want fresh", v, err)
	}
	if err := gs.UpsertNodes(context.Background(), []model.Node{{ID: "a"}}); err != nil {
		t.Fatal(err)
	}
	if v, err := storedSchema(gs.db); err != nil || v != 1 {
		t.Fatalf("unstamped store with nodes = v%d, %v; want v1 (written before versioning)", v, err)
	}
}

func TestNewerSchemaRefused(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "graph.badger")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := setMeta(gs.db, metaSchema, strconv.Itoa(schemaVersion+1)); err != nil {
		t.Fatal(err)
	}
	if err := gs.Close(); err != nil {
		t.Fatal(err)
	}
	if s, err := Open(dir); !errors.Is(err, ErrSchemaTooNew) {
		if s != nil {
			s.Close()
		}
		t.Fatalf("Open of a newer store = %v, want ErrSchemaTooNew", err)
	}
	if s, err := OpenReadOnly(dir); !errors.Is(err, ErrSchemaTooNew) {
		if s != nil {
			s.Close()
		}
		t.Fatalf("OpenReadOnly of a newer store = %v, want ErrSchemaTooNew", err)
	}
}

func TestMigrateStampsEachStep(t *testing.T) {
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	var ran []int
	steps := map[int]func(*badgerdb.DB) error{
		1: func(*badgerdb.DB) error { ran = append(ran, 1); return nil },
		2: func(*badgerdb.DB) error { ran = append(ran, 2); return errors.New("boom") },
	}
	if err := migrate(gs.db, 1, 3, steps); err == nil {
		t.Fatal("migrate succeeded despite a failing step")
	}
	if !slices.Equal(ran, []int{1, 2}) {
		t.Fatalf("steps ran %v, want [1 2]", ran)
	}
	if got := schemaOf(t, gs.db); got != "2" {
		t.Fatalf("schema after the v2 step failed = %q, want 2 (only the v1 step stamped)", got)
	}
	if err := migrate(gs.db, 3, 4, steps); err == nil {
		t.Fatal("migrate without a v3 step succeeded")
	}
}
```

In `deletion_test.go`, add `"strings"` to the imports and change the key collection at 147-155 so metadata keys are ignored:

```go
	var left []string
	if err := gs.db.View(func(txn *badgerdb.Txn) error {
		return scan(txn, "", func(k, _ []byte) error {
			if !strings.HasPrefix(string(k), pfxMeta) {
				left = append(left, string(k))
			}
			return nil
		})
	}); err != nil {
		t.Fatal(err)
	}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/graph/store/badger/ -run "Schema|Migrate|TestColonEdgeKindAdjacency" -count=1`
Expected: FAIL. The build fails with `undefined: getMeta`, `metaSchema`, `schemaVersion`, `storedSchema`, `schemaFresh`, `setMeta`, `ErrSchemaTooNew`, `migrate`, `pfxMeta`.

- [ ] **Step 3: Implement**

Create `internal/graph/store/badger/schema.go`:

```go
package badger

import (
	"errors"
	"fmt"
	"strconv"

	badgerdb "github.com/dgraph-io/badger/v4"
)

// schemaVersion is the key layout this chiton reads and writes. Bump it together with a migrations entry.
const schemaVersion = 1

// schemaFresh is storedSchema's answer for a store with no nodes, which needs no migration.
const schemaFresh = 0

// Metadata keys live under 'm', which no data prefix uses.
const (
	pfxMeta    = "m:"
	metaSchema = pfxMeta + "schema" // decimal layout version
)

// ErrSchemaTooNew reports a store written by a newer chiton, which this binary must not read or write.
var ErrSchemaTooNew = errors.New("graph store was written by a newer chiton")

// migrations[v] upgrades the layout from v to v+1. A step must be idempotent: an interrupted upgrade reruns
// the first step whose version stamp is missing.
var migrations = map[int]func(*badgerdb.DB) error{}

// prepareSchema checks the layout version on open. A writable store is stamped when fresh and migrated when
// old; an old read-only store reports errNeedsWriterPass.
func prepareSchema(db *badgerdb.DB, readOnly bool) error {
	v, err := storedSchema(db)
	if err != nil {
		return err
	}
	switch {
	case v > schemaVersion:
		return fmt.Errorf("%w (layout v%d; this chiton reads up to v%d): upgrade chiton", ErrSchemaTooNew, v, schemaVersion)
	case v == schemaVersion, v == schemaFresh && readOnly:
		return nil
	case readOnly:
		return fmt.Errorf("%w (layout v%d predates v%d)", errNeedsWriterPass, v, schemaVersion)
	case v == schemaFresh:
		return setMeta(db, metaSchema, strconv.Itoa(schemaVersion))
	default:
		return migrate(db, v, schemaVersion, migrations)
	}
}

// storedSchema returns the layout stamped in m:schema, 1 for a store with nodes but no stamp (written before
// versioning), or schemaFresh for one with no nodes.
func storedSchema(db *badgerdb.DB) (int, error) {
	v := schemaFresh
	err := db.View(func(txn *badgerdb.Txn) error {
		raw, err := getMeta(txn, metaSchema)
		if err != nil {
			return err
		}
		if raw != "" {
			if v, err = strconv.Atoi(raw); err != nil {
				return fmt.Errorf("bad store layout stamp %q: %w", raw, err)
			}
			return nil
		}
		it := prefixIterator(txn, pfxNode)
		defer it.Close()
		if it.Rewind(); it.Valid() {
			v = 1
		}
		return nil
	})
	return v, err
}

// migrate upgrades db's layout from version from to version to, stamping each step as it completes.
func migrate(db *badgerdb.DB, from, to int, steps map[int]func(*badgerdb.DB) error) error {
	for v := from; v < to; v++ {
		step, ok := steps[v]
		if !ok {
			return fmt.Errorf("no migration from store layout v%d", v)
		}
		if err := step(db); err != nil {
			return fmt.Errorf("migrate store layout v%d to v%d: %w", v, v+1, err)
		}
		if err := setMeta(db, metaSchema, strconv.Itoa(v+1)); err != nil {
			return err
		}
	}
	return nil
}

// getMeta returns the metadata value under key, or "" when it is absent.
func getMeta(txn *badgerdb.Txn, key string) (string, error) {
	item, err := txn.Get([]byte(key))
	if errors.Is(err, badgerdb.ErrKeyNotFound) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	v, err := item.ValueCopy(nil)
	return string(v), err
}

// setMeta stores a metadata value in its own transaction.
func setMeta(db *badgerdb.DB, key, val string) error {
	return db.Update(func(txn *badgerdb.Txn) error { return txn.Set([]byte(key), []byte(val)) })
}
```

In `store.go`, change the prefix comment at 25-26 to `// Key prefixes. Node IDs and edge IDs are 32-char lowercase hex strings, so the prefix bytes ('d','e','m','n','s','v') never collide with key content (metadata keys are in schema.go).`. Then replace B1's `OpenWith` and `openReadOnly` with these versions and add `prepare`/`abandon` below them:

```go
func OpenWith(path string, opts Options) (*Store, error) {
	if err := os.MkdirAll(path, 0o700); err != nil {
		return nil, fmt.Errorf("open badger at %s: %w", path, err)
	}
	lock, err := acquireLock(path, true)
	if err != nil {
		return nil, err
	}
	db, err := badgerdb.Open(badgerOptions(path, opts, false))
	if err != nil {
		lock.release()
		return nil, fmt.Errorf("open badger at %s: %w", path, err)
	}
	s := &Store{db: db, lock: lock}
	if err := s.prepare(); err != nil {
		s.abandon()
		return nil, err
	}
	return s, nil
}

func openReadOnly(path string, opts Options) (*Store, error) {
	lock, err := acquireLock(path, false)
	if err != nil {
		return nil, err
	}
	// Check before Badger opens: its failed read-only open leaks the log's mapping, which blocks repair on Windows.
	if hasWriteLog(path) {
		lock.release()
		return nil, fmt.Errorf("%w (a writer did not close cleanly)", errNeedsWriterPass)
	}
	db, err := badgerdb.Open(badgerOptions(path, opts, true))
	if err != nil {
		lock.release()
		if isTruncateNeeded(err) {
			return nil, fmt.Errorf("open badger at %s: a writer started or crashed during the open; retry: %w", path, err)
		}
		return nil, fmt.Errorf("open badger at %s: %w", path, err)
	}
	s := &Store{db: db, readOnly: true, lock: lock}
	if err := s.prepare(); err != nil {
		s.abandon()
		return nil, err
	}
	if !holdReaderLock {
		s.lock.release()
		s.lock = nil
	}
	return s, nil
}

// prepare readies a just-opened store: it checks, and when writable stamps or migrates, the key layout.
func (s *Store) prepare() error {
	return prepareSchema(s.db, s.readOnly)
}

// abandon releases a store whose open failed after Badger opened, without the writes Close makes.
func (s *Store) abandon() {
	s.db.Close()
	s.lock.release()
}
```

(Keep the doc comments B1 gave `OpenWith` and `openReadOnly`.)

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/graph/store/... -count=1; go vet ./internal/graph/store/...; go test ./... -count=1`
Expected: PASS

- [ ] **Step 5: Docs**
`CHANGELOG.md` under `### Added`: "**Graph store layout version.** The Badger store now records its key layout under `m:schema`. Opening a store written by a newer chiton fails with an "upgrade chiton" error instead of misreading it, and an older layout is migrated on the first open (a read-only open does it through a one-time read-write pass). `.chiton/schema_version` is unchanged and still describes only the `.chiton/` scaffold."

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/store/badger/schema.go internal/graph/store/badger/schema_test.go internal/graph/store/badger/store.go internal/graph/store/badger/deletion_test.go CHANGELOG.md
git commit -m "feat(store): stamp a key-layout version in the store and run migrations on open"
```

### Task B3: source_uri and kind indexes, seal, and `store.KindCounter`

**Finding:** The only key prefixes are `n: e: s: d: v:` (`internal/graph/store/badger/store.go:27-33`), so `DeleteBySourceURI` (`store.go:557-591`) scans every node record for each changed file. The audit measured about 0.25 s per file at 118k nodes, and the git hooks block on it. Callers are `internal/ingest/engine.go:156` and `:449` (plus url/slack/email/teams ingest). `NodesMatching` (`store.go:534-553`) also scans all records even for a kind filter (`Kinds: finding|meeting|decision|doc_section`, used by analyze, mcp, meeting, and subagent), and no count-by-kind exists without decoding every value.
**Priority:** P1   **Effort:** L   **Depends on:** B2, C3
**Integration:** Land after Task C3. C3 routes every node/edge write and delete through the Badger seams `putNodes`/`putEdges`/`deleteNode`/`deleteEdge`, and every source_uri lookup through `sourceIDs(txn, uris)`. Put the `u:`/`k:` index maintenance inside those seams and replace `sourceIDs`'s body with a `u:` prefix scan. The `UpsertNodes`/`DeleteBySourceURI` bodies below show the index logic against the pre-C3 layout; carry the same logic into the seams instead of restoring those bodies.
**Behaviour change:** Results are unchanged; the existing exact-match and filter-equivalence tests guard this. The first open after upgrading builds the indexes once (one pass over node records). Any open after an unclean shutdown, or after an older chiton wrote the store, rebuilds them again. It detects this through a new `m:seal` stamp written at every clean close.

**Files:**
- Create: `internal/graph/store/badger/index.go`
- Modify: `internal/graph/store/badger/store.go:25-33` (prefixes), `41-50` (checks), `104` (Close), `248-265` (deleteNode), `267-275` (batch), `279-298` (UpsertNodes), `489-553` (probe + NodesMatching), `555-620` (DeleteBySourceURI, DeleteByIDs), `690-703` (EmbeddingIDs), new `CountByKind`
- Modify: `internal/graph/store/badger/schema.go` (from B2)
- Modify: `internal/graph/store/store.go:38-44` (NodeDeleter doc), new `KindCounter`; `internal/graph/store/helpers.go` (new `CountByKind`)
- Modify: `internal/graph/store/badger/deletion_test.go` (TestColonEdgeKindAdjacency expectation)
- Test: `internal/graph/store/badger/index_test.go`, `internal/graph/store/helpers_test.go`

**Interfaces:**
- Consumes: B2's `getMeta`, `setMeta`, `pfxMeta`, `migrations`, `migrate`, `prepareSchema`, `schemaOf` (test helper); B1's `badgerOptions`
- Produces (final signatures other sections consume):
  - `store.KindCounter interface { CountByKind(ctx context.Context) (nodes map[model.NodeKind]int, edges map[model.EdgeKind]int, err error) }`
  - `store.CountByKind(ctx context.Context, gs store.GraphStore) (map[model.NodeKind]int, map[model.EdgeKind]int, error)` (falls back to AllNodes/AllEdges)
  - `(*badger.Store).CountByKind` (nodes from `k:` keys; edges from `s:{src}:{kind}:{dst}` keys, where the kind sits between the first and last `:`)

- [ ] **Step 1: Write the failing test**

```go
// internal/graph/store/badger/index_test.go
package badger

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	badgerdb "github.com/dgraph-io/badger/v4"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// indexKeys returns every kind and source-URI index key in gs, in key order.
func indexKeys(t *testing.T, gs *Store) []string {
	t.Helper()
	var keys []string
	if err := gs.db.View(func(txn *badgerdb.Txn) error {
		for _, pfx := range []string{pfxKind, pfxURI} {
			if err := eachKey(txn, pfx, func(k []byte) error {
				keys = append(keys, pfx+string(k))
				return nil
			}); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return keys
}

func sortedKeys(keys ...[]byte) []string {
	out := make([]string, len(keys))
	for i, k := range keys {
		out[i] = string(k)
	}
	slices.Sort(out)
	return out
}

// rawDB opens dir with Badger directly, bypassing chiton's lock, layout, and seal handling.
func rawDB(t *testing.T, dir string) *badgerdb.DB {
	t.Helper()
	db, err := badgerdb.Open(badgerOptions(dir, Options{}, false))
	if err != nil {
		t.Fatal(err)
	}
	return db
}

func TestIndexesTrackNodeWrites(t *testing.T) {
	ctx := context.Background()
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	check := func(stage string, want []string) {
		t.Helper()
		if got := indexKeys(t, gs); !slices.Equal(got, want) {
			t.Fatalf("%s: index keys %q, want %q", stage, got, want)
		}
	}

	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "a", Kind: model.KindSymbol, SourceURI: "file:///x.go"},
		{ID: "b", Kind: model.KindFile, SourceURI: "file:///x.go"},
		{ID: "c", Kind: model.KindFile, SourceURI: "bad\xffuri"},
	}); err != nil {
		t.Fatal(err)
	}
	check("upsert", sortedKeys(
		kindKey(model.KindSymbol, "a"), uriKey("file:///x.go", "a"),
		kindKey(model.KindFile, "b"), uriKey("file:///x.go", "b"),
		kindKey(model.KindFile, "c"), uriKey("bad�uri", "c"), // keyed as the JSON record stores it
	))

	// Re-kinding and moving a, twice in one batch, leaves only its final keys.
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "a", Kind: model.KindRule, SourceURI: "file:///z.go"},
		{ID: "a", Kind: model.KindConcept, SourceURI: "file:///y.go"},
	}); err != nil {
		t.Fatal(err)
	}
	check("re-kind", sortedKeys(
		kindKey(model.KindConcept, "a"), uriKey("file:///y.go", "a"),
		kindKey(model.KindFile, "b"), uriKey("file:///x.go", "b"),
		kindKey(model.KindFile, "c"), uriKey("bad�uri", "c"),
	))

	if n, err := gs.DeleteBySourceURI(ctx, "file:///x.go"); err != nil || n != 1 {
		t.Fatalf("DeleteBySourceURI = %d, %v; want 1", n, err)
	}
	if n, err := gs.DeleteByIDs(ctx, []string{"a"}); err != nil || n != 1 {
		t.Fatalf("DeleteByIDs = %d, %v; want 1", n, err)
	}
	check("delete", sortedKeys(kindKey(model.KindFile, "c"), uriKey("bad�uri", "c")))
}

func TestCountByKind(t *testing.T) {
	ctx := context.Background()
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "a", Kind: model.KindSymbol}, {ID: "b", Kind: model.KindSymbol}, {ID: "c", Kind: model.KindFile}, {ID: "d"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{
		{ID: "ab", Src: "a", Dst: "b", Kind: model.EdgeCalls},
		{ID: "ca", Src: "c", Dst: "a", Kind: model.EdgeCalls},
		{ID: "bc", Src: "b", Dst: "c", Kind: "x:y"}, // an imported kind may contain ':'
	}); err != nil {
		t.Fatal(err)
	}
	check := func(wantNodes map[model.NodeKind]int, wantEdges map[model.EdgeKind]int) {
		t.Helper()
		nodes, edges, err := gs.CountByKind(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !maps.Equal(nodes, wantNodes) || !maps.Equal(edges, wantEdges) {
			t.Fatalf("CountByKind = %v %v, want %v %v", nodes, edges, wantNodes, wantEdges)
		}
	}
	check(map[model.NodeKind]int{model.KindSymbol: 2, model.KindFile: 1, "": 1},
		map[model.EdgeKind]int{model.EdgeCalls: 2, "x:y": 1})
	if _, err := gs.DeleteByIDs(ctx, []string{"a"}); err != nil {
		t.Fatal(err)
	}
	check(map[model.NodeKind]int{model.KindSymbol: 1, model.KindFile: 1, "": 1},
		map[model.EdgeKind]int{"x:y": 1})
}

func TestV1StoreMigratesOnReadOnlyOpen(t *testing.T) {
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "graph.badger")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "a", Kind: model.KindFile, SourceURI: "file:///a.go"},
		{ID: "b", Kind: model.KindSymbol, SourceURI: "file:///a.go"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := gs.Close(); err != nil {
		t.Fatal(err)
	}
	// Rewind to a pre-index (v1) store: node records only, no index or metadata keys.
	db := rawDB(t, dir)
	if err := db.DropPrefix([]byte(pfxURI), []byte(pfxKind), []byte(pfxMeta)); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatalf("OpenReadOnly of a v1 store: %v", err)
	}
	defer ro.Close()
	if got := schemaOf(t, ro.db); got != strconv.Itoa(schemaVersion) {
		t.Fatalf("schema after upgrade = %q, want %d", got, schemaVersion)
	}
	if got, err := ro.NodesMatching(ctx, store.NodeFilter{SourceURI: "file:///a.go"}); err != nil || len(got) != 2 {
		t.Fatalf("NodesMatching by source_uri = %d nodes, %v; want 2", len(got), err)
	}
	nodes, _, err := ro.CountByKind(ctx)
	if err != nil || !maps.Equal(nodes, map[model.NodeKind]int{model.KindFile: 1, model.KindSymbol: 1}) {
		t.Fatalf("CountByKind nodes = %v, %v", nodes, err)
	}
}

// TestUnsealedStoreIsReindexed guards the seal: a write that bypassed index upkeep (a crash, or an older
// chiton) must not hide a node from DeleteBySourceURI.
func TestUnsealedStoreIsReindexed(t *testing.T) {
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "graph.badger")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertNodes(ctx, []model.Node{{ID: "a", Kind: model.KindFile, SourceURI: "file:///x.go"}}); err != nil {
		t.Fatal(err)
	}
	if err := gs.Close(); err != nil {
		t.Fatal(err)
	}
	rec, err := json.Marshal(model.Node{ID: "b", Kind: model.KindSymbol, SourceURI: "file:///x.go"})
	if err != nil {
		t.Fatal(err)
	}
	db := rawDB(t, dir)
	if err := db.Update(func(txn *badgerdb.Txn) error { return txn.Set(nodeKey("b"), rec) }); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	gs, err = Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	if n, err := gs.DeleteBySourceURI(ctx, "file:///x.go"); err != nil || n != 2 {
		t.Fatalf("DeleteBySourceURI = %d, %v; want both nodes, including the unindexed one", n, err)
	}
}

func TestStoredStringMatchesJSON(t *testing.T) {
	for _, s := range []string{"plain", "é", "bad\xffutf8", "two\xff\xfebad", "trunc\xc3", "mix\xe2\x82"} {
		b, err := json.Marshal(s)
		if err != nil {
			t.Fatal(err)
		}
		var back string
		if err := json.Unmarshal(b, &back); err != nil {
			t.Fatal(err)
		}
		if got := storedString(s); got != back {
			t.Errorf("storedString(%q) = %q, want the JSON round trip %q", s, got, back)
		}
	}
}

// BenchmarkDeleteBySourceURI deletes (and, untimed, restores) one file's nodes among n; with the index its
// cost no longer grows with n.
func BenchmarkDeleteBySourceURI(b *testing.B) {
	for _, n := range []int{2_000, 20_000} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			ctx := context.Background()
			gs, err := Open(filepath.Join(b.TempDir(), "graph.badger"))
			if err != nil {
				b.Fatal(err)
			}
			defer gs.Close()
			nodes := make([]model.Node, n)
			var target []model.Node
			for i := range nodes {
				nodes[i] = model.Node{ID: fmt.Sprintf("%032d", i), Kind: model.KindSymbol, SourceURI: fmt.Sprintf("file:///f%d.go", i%500)}
				if i%500 == 7 {
					target = append(target, nodes[i])
				}
			}
			if err := gs.UpsertNodes(ctx, nodes); err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for range b.N {
				if _, err := gs.DeleteBySourceURI(ctx, "file:///f7.go"); err != nil {
					b.Fatal(err)
				}
				b.StopTimer()
				if err := gs.UpsertNodes(ctx, target); err != nil {
					b.Fatal(err)
				}
				b.StartTimer()
			}
		})
	}
}
```

Update `TestColonEdgeKindAdjacency` in `deletion_test.go` (add `"slices"` to the imports). Replace the final assertion (from B2's `left` loop onward) with:

```go
	want := []string{string(kindKey("", "b")), nodeKeyString("b"), string(uriKey("", "b"))}
	if !slices.Equal(left, want) {
		t.Fatalf("keys after deleting a = %q, want only b's node and index keys %q", left, want)
	}
```

Append to `internal/graph/store/helpers_test.go` (add `"maps"` to the imports):

```go
func TestCountByKindFallbackAgrees(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "a", Kind: model.KindSymbol}, {ID: "b", Kind: model.KindFile}, {ID: "c", Kind: model.KindSymbol},
	}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{
		{ID: "ab", Src: "a", Dst: "b", Kind: model.EdgeCalls},
		{ID: "cb", Src: "c", Dst: "b", Kind: model.EdgeReferences},
	}); err != nil {
		t.Fatal(err)
	}
	wantNodes := map[model.NodeKind]int{model.KindSymbol: 2, model.KindFile: 1}
	wantEdges := map[model.EdgeKind]int{model.EdgeCalls: 1, model.EdgeReferences: 1}
	for name, s := range map[string]store.GraphStore{"counter": gs, "fallback": plain{gs}} {
		nodes, edges, err := store.CountByKind(ctx, s)
		if err != nil || !maps.Equal(nodes, wantNodes) || !maps.Equal(edges, wantEdges) {
			t.Errorf("%s CountByKind = %v %v %v, want %v %v", name, nodes, edges, err, wantNodes, wantEdges)
		}
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/graph/store/... -count=1`
Expected: FAIL. The build fails with `undefined: eachKey`, `pfxKind`, `pfxURI`, `kindKey`, `uriKey`, `storedString`, `gs.CountByKind`, `store.CountByKind`. `TestUnsealedStoreIsReindexed` would pass today because the old scan finds the unindexed node; it guards the seal once the scan is gone.

- [ ] **Step 3: Implement**

Create `internal/graph/store/badger/index.go`:

```go
package badger

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strings"
	"unicode/utf8"

	badgerdb "github.com/dgraph-io/badger/v4"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// indexed holds the node fields the secondary indexes key on, decoded without the rest of the record.
type indexed struct {
	Kind      model.NodeKind `json:"kind"`
	SourceURI string         `json:"source_uri"`
}

// indexedOf returns n's indexed fields as its JSON record will store them.
func indexedOf(n model.Node) indexed {
	return indexed{Kind: model.NodeKind(storedString(string(n.Kind))), SourceURI: storedString(n.SourceURI)}
}

// storedString returns s as it reads back from JSON: encoding/json turns each invalid UTF-8 byte into U+FFFD.
func storedString(s string) string {
	if utf8.ValidString(s) {
		return s
	}
	var b strings.Builder
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError && size == 1 {
			b.WriteRune(utf8.RuneError)
		} else {
			b.WriteString(s[i : i+size])
		}
		i += size
	}
	return b.String()
}

// uriPrefix is the index prefix for a source_uri. Hashing keeps keys short and free of the \x00 separator.
func uriPrefix(uri string) string {
	sum := sha256.Sum256([]byte(uri))
	return pfxURI + hex.EncodeToString(sum[:16]) + "\x00"
}

func uriKey(uri, id string) []byte { return []byte(uriPrefix(uri) + id) }

// kindPrefix is the index prefix for a node kind, kept raw so CountByKind can read it back.
func kindPrefix(k model.NodeKind) string { return pfxKind + string(k) + "\x00" }

func kindKey(k model.NodeKind, id string) []byte { return []byte(kindPrefix(k) + id) }

// setIndex writes node id's index keys.
func setIndex(wb *badgerdb.WriteBatch, id string, x indexed) error {
	if err := wb.Set(uriKey(x.SourceURI, id), nil); err != nil {
		return err
	}
	return wb.Set(kindKey(x.Kind, id), nil)
}

// dropStaleIndex deletes the index keys prev gave node id that cur no longer shares.
func dropStaleIndex(wb *badgerdb.WriteBatch, id string, prev, cur indexed) error {
	if prev.SourceURI != cur.SourceURI {
		if err := wb.Delete(uriKey(prev.SourceURI, id)); err != nil {
			return err
		}
	}
	if prev.Kind != cur.Kind {
		return wb.Delete(kindKey(prev.Kind, id))
	}
	return nil
}

// eachKey calls fn, in key order, with the suffix of every key under pfx, reading keys only. The suffix is
// valid only during the call.
func eachKey(txn *badgerdb.Txn, pfx string, fn func(k []byte) error) error {
	it := prefixIterator(txn, pfx)
	defer it.Close()
	for it.Rewind(); it.Valid(); it.Next() {
		if err := fn(it.Item().Key()[len(pfx):]); err != nil {
			return err
		}
	}
	return nil
}

// keySuffixes returns the suffix of every key under pfx, in key order.
func keySuffixes(txn *badgerdb.Txn, pfx string) ([]string, error) {
	var out []string
	err := eachKey(txn, pfx, func(k []byte) error {
		out = append(out, string(k))
		return nil
	})
	return out, err
}

// candidateIDs returns, sorted, the node IDs the index admits for f's source_uri or else its kinds. ok is
// false when f constrains neither, which leaves a scan.
func candidateIDs(txn *badgerdb.Txn, f store.NodeFilter) (ids []string, ok bool, err error) {
	var prefixes []string
	switch {
	case f.SourceURI != "":
		prefixes = []string{uriPrefix(f.SourceURI)}
	case len(f.Kinds) > 0:
		for _, k := range f.Kinds {
			prefixes = append(prefixes, kindPrefix(k))
		}
	default:
		return nil, false, nil
	}
	for _, p := range prefixes {
		got, err := keySuffixes(txn, p)
		if err != nil {
			return nil, false, err
		}
		ids = append(ids, got...)
	}
	slices.Sort(ids)
	return slices.Compact(ids), true, nil
}

// reindex rebuilds both indexes from the node records. It is the v1→v2 migration and the repair for a store
// whose last writer didn't seal it.
func reindex(db *badgerdb.DB) error {
	return writeBatch(db, func(wb *badgerdb.WriteBatch) error {
		return db.View(func(txn *badgerdb.Txn) error {
			for _, pfx := range []string{pfxURI, pfxKind} {
				if err := eachKey(txn, pfx, func(k []byte) error {
					return wb.Delete(append([]byte(pfx), k...))
				}); err != nil {
					return err
				}
			}
			return scan(txn, pfxNode, func(k, v []byte) error {
				var x indexed
				if err := json.Unmarshal(v, &x); err != nil {
					return err
				}
				return setIndex(wb, string(k), x)
			})
		})
	})
}
```

In `store.go`, replace the prefix block (25-33) with:

```go
// Key prefixes. Node IDs and edge IDs are 32-char lowercase hex strings, so the prefix bytes
// ('d','e','k','m','n','s','u','v') never collide with key content (metadata keys are in schema.go).
const (
	pfxNode = "n:" // n:{nodeID}                        → json(Node)
	pfxEdge = "e:" // e:{edgeID}                        → json(Edge)
	pfxSrc  = "s:" // s:{src}:{kind}:{dst}              → edgeID   (forward adjacency)
	pfxDst  = "d:" // d:{dst}:{kind}:{src}              → edgeID   (reverse adjacency)
	pfxEmb  = "v:" // v:{nodeID}                        → packed little-endian []float32
	pfxURI  = "u:" // u:{hash(source_uri)}\x00{nodeID}  → empty    (source_uri index)
	pfxKind = "k:" // k:{kind}\x00{nodeID}              → empty    (kind index)
)
```

Add `_ store.KindCounter = (*Store)(nil)` to the compile-time checks (41-50). Replace `Close` (104):

```go
// Close seals a writable store (see sealed) and releases it.
func (s *Store) Close() error {
	var sealErr error
	if !s.readOnly {
		sealErr = seal(s.db)
	}
	return errors.Join(sealErr, s.db.Close(), s.lock.release())
}
```

Replace `deleteNode` (248-265) and `batch` (267-275):

```go
// deleteNode queues deletion of node id (whose indexed fields are x): its record, embedding, every adjacent
// edge with its adjacency keys, then its index keys.
func deleteNode(txn *badgerdb.Txn, wb *badgerdb.WriteBatch, id string, x indexed) error {
	adj, err := adjacentEdgeKeys(txn, id, nil)
	if err != nil {
		return err
	}
	keys := make([][]byte, 0, 4+3*len(adj))
	keys = append(keys, nodeKey(id), embKey(id))
	for i := range adj {
		keys = append(keys, edgeKey(adj[i].ID), srcKey(&adj[i]), dstKey(&adj[i]))
	}
	keys = append(keys, uriKey(x.SourceURI, id), kindKey(x.Kind, id))
	for _, k := range keys {
		if err := wb.Delete(k); err != nil {
			return err
		}
	}
	return nil
}

// writeBatch runs fill against a new write batch on db, flushing it on success and discarding it on error.
func writeBatch(db *badgerdb.DB, fill func(wb *badgerdb.WriteBatch) error) error {
	wb := db.NewWriteBatch()
	if err := fill(wb); err != nil {
		wb.Cancel()
		return err
	}
	return wb.Flush()
}

func (s *Store) batch(fill func(wb *badgerdb.WriteBatch) error) error { return writeBatch(s.db, fill) }
```

Replace `UpsertNodes` (279-298):

```go
// UpsertNodes writes nodes and keeps their index keys exact, dropping keys a changed kind or source_uri
// left behind.
func (s *Store) UpsertNodes(_ context.Context, nodes []model.Node) error {
	now := time.Now()
	return s.batch(func(wb *badgerdb.WriteBatch) error {
		return s.db.View(func(txn *badgerdb.Txn) error {
			written := map[string]indexed{} // this batch's earlier writes, which txn can't see
			for i := range nodes {
				n := nodes[i]
				if n.CreatedAt.IsZero() {
					n.CreatedAt = now
				}
				n.UpdatedAt = now
				data, err := json.Marshal(n)
				if err != nil {
					return fmt.Errorf("marshal node %s: %w", n.ID, err)
				}
				prev, had := written[n.ID]
				if !had {
					old, err := loadJSON[indexed](txn, nodeKey(n.ID))
					if err != nil {
						return err
					}
					if old != nil {
						prev, had = *old, true
					}
				}
				cur := indexedOf(n)
				written[n.ID] = cur
				if err := setIndex(wb, n.ID, cur); err != nil {
					return err
				}
				if err := wb.Set(nodeKey(n.ID), data); err != nil {
					return err
				}
				if had {
					if err := dropStaleIndex(wb, n.ID, prev, cur); err != nil {
						return err
					}
				}
			}
			return nil
		})
	})
}
```

Replace `nodeProbe`/`newNodeProbe`/`candidate` (489-530) and `NodesMatching` (532-553); `stringProbe` stays. Only attrs still need the probe:

```go
// attrProbe returns raw-bytes probes for f's attrs; a record matching f contains every one. Empty values
// (which also match an absent attr) and values whose escaping could vary get none.
func attrProbe(f store.NodeFilter) [][]byte {
	var probes [][]byte
	for _, v := range f.Attrs {
		if b := stringProbe(v); v != "" && b != nil {
			probes = append(probes, b)
		}
	}
	return probes
}

// NodesMatching returns the nodes matching f in AllNodes order. A source_uri or kind constraint loads only the
// index's candidates; otherwise it scans, decoding records that pass the attr probe. Implements store.NodeScanner.
func (s *Store) NodesMatching(_ context.Context, f store.NodeFilter) ([]model.Node, error) {
	var nodes []model.Node
	keep := func(n model.Node) {
		if f.Match(n) {
			nodes = append(nodes, n)
		}
	}
	err := s.db.View(func(txn *badgerdb.Txn) error {
		ids, ok, err := candidateIDs(txn, f)
		if err != nil {
			return err
		}
		if ok {
			for _, id := range ids {
				n, err := loadNode(txn, id)
				if err != nil {
					return err
				}
				if n != nil {
					keep(*n)
				}
			}
			return nil
		}
		probes := attrProbe(f)
		return scan(txn, pfxNode, func(_, v []byte) error {
			if slices.ContainsFunc(probes, func(p []byte) bool { return !bytes.Contains(v, p) }) {
				return nil
			}
			var n model.Node
			if err := json.Unmarshal(v, &n); err != nil {
				return err
			}
			keep(n)
			return nil
		})
	})
	return nodes, err
}
```

Replace `DeleteBySourceURI` (555-591) and `DeleteByIDs` (593-620):

```go
// DeleteBySourceURI removes the nodes whose source_uri is uri, found through the source-URI index, and all
// edges adjacent to them. Returns the number of nodes deleted.
func (s *Store) DeleteBySourceURI(_ context.Context, uri string) (int, error) {
	deleted := 0
	err := s.batch(func(wb *badgerdb.WriteBatch) error {
		return s.db.View(func(txn *badgerdb.Txn) error {
			ids, err := keySuffixes(txn, uriPrefix(uri))
			if err != nil {
				return err
			}
			for _, id := range ids {
				x, err := loadJSON[indexed](txn, nodeKey(id))
				if err != nil {
					return err
				}
				if x == nil || x.SourceURI != uri {
					continue // guards against a digest collision
				}
				if err := deleteNode(txn, wb, id, *x); err != nil {
					return err
				}
				deleted++
			}
			return nil
		})
	})
	return deleted, err
}

// DeleteByIDs deletes nodes by ID plus their embeddings, adjacent edges, and index keys, skipping missing IDs.
// Point lookups keep it O(K) in the number of IDs.
func (s *Store) DeleteByIDs(_ context.Context, ids []string) (int, error) {
	if len(ids) == 0 {
		return 0, nil
	}
	deleted := 0
	err := s.batch(func(wb *badgerdb.WriteBatch) error {
		return s.db.View(func(txn *badgerdb.Txn) error {
			for _, id := range ids {
				x, err := loadJSON[indexed](txn, nodeKey(id))
				if err != nil {
					return err
				}
				if x == nil {
					continue
				}
				if err := deleteNode(txn, wb, id, *x); err != nil {
					return err
				}
				deleted++
			}
			return nil
		})
	})
	return deleted, err
}
```

Add `CountByKind` after `AllEdges`, and replace `EmbeddingIDs` (690-703):

```go
// CountByKind tallies nodes from the kind index and edges from their forward adjacency keys, reading keys only.
// Implements store.KindCounter.
func (s *Store) CountByKind(_ context.Context) (map[model.NodeKind]int, map[model.EdgeKind]int, error) {
	nodes, edges := map[model.NodeKind]int{}, map[model.EdgeKind]int{}
	err := s.db.View(func(txn *badgerdb.Txn) error {
		if err := eachKey(txn, pfxKind, func(k []byte) error {
			if i := bytes.LastIndexByte(k, 0); i >= 0 {
				nodes[model.NodeKind(k[:i])]++
			}
			return nil
		}); err != nil {
			return err
		}
		return eachKey(txn, pfxSrc, func(k []byte) error {
			// k is {src}:{kind}:{dst}. Node IDs hold no ':', but an imported edge kind may.
			if i, j := bytes.IndexByte(k, ':'), bytes.LastIndexByte(k, ':'); i >= 0 && j > i {
				edges[model.EdgeKind(k[i+1:j])]++
			}
			return nil
		})
	})
	return nodes, edges, err
}

// EmbeddingIDs returns the node ID of every stored embedding in key order, reading keys only.
// Implements store.EmbeddingIDLister.
func (s *Store) EmbeddingIDs(_ context.Context) ([]string, error) {
	var ids []string
	err := s.db.View(func(txn *badgerdb.Txn) (err error) {
		ids, err = keySuffixes(txn, pfxEmb)
		return err
	})
	return ids, err
}
```

In `schema.go`, bump the version, register the migration, add the seal key and helpers, and replace `prepareSchema`:

```go
// schemaVersion is the key layout this chiton reads and writes. v2 added the source_uri and kind indexes
// and the seal.
const schemaVersion = 2

const (
	pfxMeta    = "m:"
	metaSchema = pfxMeta + "schema" // decimal layout version
	metaSeal   = pfxMeta + "seal"   // Badger version at the last clean close (see sealed)
)

var migrations = map[int]func(*badgerdb.DB) error{
	1: reindex,
}

// prepareSchema checks the layout version on open and that the indexes are intact. A writable store is
// stamped, migrated, or reindexed as needed; a read-only one reports errNeedsWriterPass instead.
func prepareSchema(db *badgerdb.DB, readOnly bool) error {
	v, err := storedSchema(db)
	if err != nil {
		return err
	}
	switch {
	case v > schemaVersion:
		return fmt.Errorf("%w (layout v%d; this chiton reads up to v%d): upgrade chiton", ErrSchemaTooNew, v, schemaVersion)
	case v == schemaFresh:
		if readOnly {
			return nil
		}
		return setMeta(db, metaSchema, strconv.Itoa(schemaVersion))
	case v < schemaVersion && readOnly:
		return fmt.Errorf("%w (layout v%d predates v%d)", errNeedsWriterPass, v, schemaVersion)
	case v < schemaVersion:
		return migrate(db, v, schemaVersion, migrations) // v1→v2 builds the indexes afresh
	}
	ok, err := sealed(db)
	switch {
	case err != nil || ok:
		return err
	case readOnly:
		return fmt.Errorf("%w (the last writer did not close cleanly)", errNeedsWriterPass)
	default:
		return reindex(db)
	}
}

// seal records the store's current Badger version. The seal's own commit is the next version, which sealed
// checks to prove nothing wrote after it without index upkeep (a crash, or an older chiton).
func seal(db *badgerdb.DB) error {
	return setMeta(db, metaSeal, strconv.FormatUint(db.MaxVersion(), 10))
}

// sealed reports whether the seal is still the store's newest commit.
func sealed(db *badgerdb.DB) (bool, error) {
	var raw string
	if err := db.View(func(txn *badgerdb.Txn) (err error) {
		raw, err = getMeta(txn, metaSeal)
		return err
	}); err != nil || raw == "" {
		return false, err
	}
	v, err := strconv.ParseUint(raw, 10, 64)
	return err == nil && db.MaxVersion() == v+1, nil
}
```

(A scratch probe on Badger v4.9.6 confirmed that `MaxVersion()` is 3 after a two-commit batch plus the seal, and still 3 after reopening read-write or read-only.)

In `internal/graph/store/store.go`, change the `NodeDeleter` doc (38-39) to `// NodeDeleter is an optional capability for deleting nodes by ID with point lookups. Callers type-assert it and fall back to DeleteBySourceURI without it.`. Add after `EmbeddingIDLister`:

```go
// KindCounter is an optional capability counting nodes and edges by kind without decoding values. Callers use
// the CountByKind helper, which falls back to AllNodes/AllEdges without it.
type KindCounter interface {
	CountByKind(ctx context.Context) (nodes map[model.NodeKind]int, edges map[model.EdgeKind]int, err error)
}
```

Append to `internal/graph/store/helpers.go`:

```go
// CountByKind tallies gs's nodes and edges by kind, via KindCounter when gs has it.
func CountByKind(ctx context.Context, gs GraphStore) (map[model.NodeKind]int, map[model.EdgeKind]int, error) {
	if c, ok := gs.(KindCounter); ok {
		return c.CountByKind(ctx)
	}
	all, err := gs.AllNodes(ctx)
	if err != nil {
		return nil, nil, err
	}
	nodes := map[model.NodeKind]int{}
	for _, n := range all {
		nodes[n.Kind]++
	}
	allEdges, err := gs.AllEdges(ctx)
	if err != nil {
		return nil, nil, err
	}
	edges := map[model.EdgeKind]int{}
	for _, e := range allEdges {
		edges[e.Kind]++
	}
	return nodes, edges, nil
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/graph/store/... -count=1; go vet ./internal/graph/store/...; go test ./... -count=1; go test ./internal/graph/store/badger/ -run '^$' -bench DeleteBySourceURI -benchtime 200x`
Expected: PASS. The existing `TestDeleteBySourceURIExactMatch` and `TestNodesMatchingEqualsFilteredAllNodes` pass unchanged, including the invalid-UTF-8 and `ctl\x01kind` cases. The 2000 and 20000 sub-benchmarks report ns/op within about 2x of each other (before this task the 20000 case is roughly 10x slower).

- [ ] **Step 5: Docs**
`CHANGELOG.md` under `### Changed`: "**Incremental sync no longer rescans the whole graph per file.** The Badger store keeps source_uri and kind indexes (store layout v2), so `DeleteBySourceURI` and kind or source filters read only matching nodes. The first open after upgrading builds the indexes once, and an open after an unclean shutdown rebuilds them. `store.CountByKind` counts nodes and edges from keys alone."

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/store/badger/index.go internal/graph/store/badger/index_test.go internal/graph/store/badger/store.go internal/graph/store/badger/schema.go internal/graph/store/badger/deletion_test.go internal/graph/store/store.go internal/graph/store/helpers.go internal/graph/store/helpers_test.go CHANGELOG.md
git commit -m "perf(store): index nodes by source_uri and kind, seal clean closes, add KindCounter"
```

### Task B4: Label index for read-only `SearchNodes`

**Finding:** `SearchNodes` (`internal/graph/store/badger/store.go:623-650`) JSON-decodes every node record's label on each call, about 0.35 s per miss at 118k nodes. It backs `get_node`/`get_node_neighborhood` (`store.FindNode`, `internal/mcp/server.go:163`), `query_graph` (`server.go:240`), and `shortest_path`, which calls it twice (`server.go:272`, `:280`). A read-only store never changes during its process, so the decoded labels can be kept.
**Priority:** P3   **Effort:** S   **Depends on:** B2
**Behaviour change:** none. Results and their order are identical. The first search on a read-only store pays one scan, and later ones don't. Writable stores keep scanning.

**Files:**
- Modify: `internal/graph/store/badger/store.go` (Store struct, `prepare` from B2, `SearchNodes` 622-650, imports: add `"sync"`)
- Test: `internal/graph/store/badger/search_test.go`

**Interfaces:**
- Consumes: B2's `(*Store).prepare()`
- Produces: none exported

- [ ] **Step 1: Write the failing test**

```go
// internal/graph/store/badger/search_test.go
package badger

import (
	"context"
	"fmt"
	"path/filepath"
	"slices"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func idsOf(nodes []model.Node) []string {
	out := make([]string, len(nodes))
	for i, n := range nodes {
		out[i] = n.ID
	}
	return out
}

func TestReadOnlySearchNodesMatchesScan(t *testing.T) {
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "graph.badger")
	rw, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	var nodes []model.Node
	for i, l := range []string{"Fetch", "fetchAll", "FETCH_ALL", "Straße", "ÜBER", "", "prefetcher", "other"} {
		nodes = append(nodes, model.Node{ID: fmt.Sprintf("n%02d", i), Kind: model.KindSymbol, Label: l})
	}
	if err := rw.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	type query struct {
		q     string
		limit int
	}
	want := map[query][]string{}
	for _, q := range []string{"", "fetch", "FETCH", "straße", "über", "zzz", "_"} {
		for _, limit := range []int{0, 1, 2} {
			got, err := rw.SearchNodes(ctx, q, limit)
			if err != nil {
				t.Fatal(err)
			}
			want[query{q, limit}] = idsOf(got)
		}
	}
	if err := rw.Close(); err != nil {
		t.Fatal(err)
	}

	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ro.Close()
	for k, w := range want {
		got, err := ro.SearchNodes(ctx, k.q, k.limit)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(idsOf(got), w) {
			t.Errorf("read-only SearchNodes(%q, %d) = %v, want the scan's %v", k.q, k.limit, idsOf(got), w)
		}
	}
}

func TestReadOnlySearchNodesSkipsRecordDecoding(t *testing.T) {
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "graph.badger")
	rw, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	nodes := make([]model.Node, 2000)
	for i := range nodes {
		nodes[i] = model.Node{ID: fmt.Sprintf("n%05d", i), Kind: model.KindSymbol, Label: fmt.Sprintf("sym%05d", i)}
	}
	nodes[1999].Label = "needle"
	if err := rw.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	if err := rw.Close(); err != nil {
		t.Fatal(err)
	}
	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer ro.Close()
	if got, err := ro.SearchNodes(ctx, "NEEDLE", 1); err != nil || len(got) != 1 || got[0].ID != "n01999" {
		t.Fatalf("SearchNodes(NEEDLE) = %v, %v", idsOf(got), err)
	}
	// A scan decodes 2000 labels (at least one allocation each); the index loads one record.
	allocs := testing.AllocsPerRun(10, func() { _, _ = ro.SearchNodes(ctx, "needle", 1) })
	if allocs > 500 {
		t.Fatalf("read-only SearchNodes allocated %.0f times per call, want an indexed lookup (< 500)", allocs)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/graph/store/badger/ -run TestReadOnlySearchNodes -count=1`
Expected: FAIL. `read-only SearchNodes allocated 4000+ times per call, want an indexed lookup (< 500)`. `TestReadOnlySearchNodesMatchesScan` already passes; it guards equivalence.

- [ ] **Step 3: Implement**

In `store.go` add `"sync"` to the imports and a field to `Store`:

```go
	labels   func() ([]labelEntry, error) // read-only stores: lowercased labels in key order, loaded on first search
```

Extend B2's `prepare`:

```go
// prepare readies a just-opened store: it checks, and when writable stamps or migrates, the key layout; a
// read-only store also gets its lazy label index.
func (s *Store) prepare() error {
	if err := prepareSchema(s.db, s.readOnly); err != nil {
		return err
	}
	if s.readOnly {
		s.labels = sync.OnceValues(func() ([]labelEntry, error) { return loadLabels(s.db) })
	}
	return nil
}
```

Replace `SearchNodes` (622-650) with:

```go
// labelEntry is one node in a read-only store's label index.
type labelEntry struct{ lower, id string }

// labelOnly decodes just a node record's label.
type labelOnly struct {
	Label string `json:"label"`
}

// SearchNodes returns the nodes whose label contains query (case-insensitive), in key order. A read-only
// store never changes, so it matches against a label index built on first use; a writable one scans.
func (s *Store) SearchNodes(ctx context.Context, query string, limit int) ([]model.Node, error) {
	if s.labels == nil {
		return s.scanLabels(query, limit)
	}
	idx, err := s.labels()
	if err != nil {
		return nil, err
	}
	q := strings.ToLower(query)
	var ids []string
	for _, e := range idx {
		if !strings.Contains(e.lower, q) {
			continue
		}
		ids = append(ids, e.id)
		if limit > 0 && len(ids) >= limit {
			break
		}
	}
	return s.GetNodes(ctx, ids)
}

// scanLabels is SearchNodes over the live key space: it decodes each label, then only matching records in full.
func (s *Store) scanLabels(query string, limit int) ([]model.Node, error) {
	q := strings.ToLower(query)
	var results []model.Node
	err := s.db.View(func(txn *badgerdb.Txn) error {
		return scan(txn, pfxNode, func(_, v []byte) error {
			var probe labelOnly
			if err := json.Unmarshal(v, &probe); err != nil {
				return err
			}
			if !strings.Contains(strings.ToLower(probe.Label), q) {
				return nil
			}
			var n model.Node
			if err := json.Unmarshal(v, &n); err != nil {
				return err
			}
			results = append(results, n)
			if limit > 0 && len(results) >= limit {
				return errStopScan
			}
			return nil
		})
	})
	return results, err
}

// loadLabels reads every node's lowercased label, in key order, decoding only the label.
func loadLabels(db *badgerdb.DB) ([]labelEntry, error) {
	var idx []labelEntry
	err := db.View(func(txn *badgerdb.Txn) error {
		return scan(txn, pfxNode, func(k, v []byte) error {
			var probe labelOnly
			if err := json.Unmarshal(v, &probe); err != nil {
				return err
			}
			idx = append(idx, labelEntry{lower: strings.ToLower(probe.Label), id: string(k)})
			return nil
		})
	})
	return idx, err
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/graph/store/... ./internal/mcp/... -count=1; go vet ./internal/graph/store/...`
Expected: PASS

- [ ] **Step 5: Docs**
`CHANGELOG.md` under `### Changed`: "Label lookups on a read-only store (`get_node`, `query_graph`, `shortest_path`, `chiton query`) answer from an in-memory label index after the first search instead of decoding every node."

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/store/badger/store.go internal/graph/store/badger/search_test.go CHANGELOG.md
git commit -m "perf(store): serve read-only SearchNodes from a lazily built label index"
```

### Task B5: ShortestPath BFS over adjacency keys

**Finding:** `ShortestPath` (`internal/graph/store/badger/store.go:411-474`) calls `adjacentEdges` at `store.go:432`, which does a Get and JSON decode for every edge the BFS touches. The key-only `adjacentEdgeKeys` (`store.go:202-233`) already yields ID, Src, Dst, and Kind, which is all the BFS reads, and only the final path's edges need their records (already loaded at `store.go:470`). ShortestPath has no tests.
**Priority:** P3   **Effort:** S   **Depends on:** none
**Behaviour change:** none for well-formed stores (same traversal order, so the same path). A malformed edge record off the path no longer fails the search.

**Files:**
- Modify: `internal/graph/store/badger/store.go:409-432`
- Test: `internal/graph/store/badger/path_test.go`

**Interfaces:**
- Consumes: none
- Produces: none

- [ ] **Step 1: Write the failing test**

```go
// internal/graph/store/badger/path_test.go
package badger

import (
	"context"
	"path/filepath"
	"slices"
	"testing"

	badgerdb "github.com/dgraph-io/badger/v4"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// pathIDs returns p's node and edge IDs in path order.
func pathIDs(p store.Path) (nodes, edges []string) {
	for _, n := range p.Nodes {
		nodes = append(nodes, n.ID)
	}
	for _, e := range p.Edges {
		edges = append(edges, e.ID)
	}
	return nodes, edges
}

// pathStore holds a-b-c-d (c→b points backwards), a side edge a→s, and an isolated z.
func pathStore(t *testing.T) *Store {
	t.Helper()
	ctx := context.Background()
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	var nodes []model.Node
	for _, id := range []string{"a", "b", "c", "d", "s", "z"} {
		nodes = append(nodes, model.Node{ID: id, Label: id})
	}
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{
		{ID: "ab", Src: "a", Dst: "b", Kind: model.EdgeCalls},
		{ID: "cb", Src: "c", Dst: "b", Kind: model.EdgeReferences},
		{ID: "cd", Src: "c", Dst: "d", Kind: model.EdgeCalls},
		{ID: "as", Src: "a", Dst: "s", Kind: model.EdgeContains},
	}); err != nil {
		t.Fatal(err)
	}
	return gs
}

func TestShortestPath(t *testing.T) {
	ctx := context.Background()
	gs := pathStore(t)
	for _, tc := range []struct {
		src, dst             string
		wantNodes, wantEdges []string
	}{
		{"a", "d", []string{"a", "b", "c", "d"}, []string{"ab", "cb", "cd"}},
		{"d", "a", []string{"d", "c", "b", "a"}, []string{"cd", "cb", "ab"}},
		{"a", "a", []string{"a"}, nil},
		{"a", "z", nil, nil},       // unreachable
		{"a", "missing", nil, nil}, // unknown node
	} {
		p, err := gs.ShortestPath(ctx, tc.src, tc.dst)
		if err != nil {
			t.Fatalf("ShortestPath(%s, %s): %v", tc.src, tc.dst, err)
		}
		nodes, edges := pathIDs(p)
		if !slices.Equal(nodes, tc.wantNodes) || !slices.Equal(edges, tc.wantEdges) {
			t.Errorf("ShortestPath(%s, %s) = %v %v, want %v %v", tc.src, tc.dst, nodes, edges, tc.wantNodes, tc.wantEdges)
		}
	}
}

func TestShortestPathDecodesOnlyPathEdges(t *testing.T) {
	ctx := context.Background()
	gs := pathStore(t)
	// Corrupt the side edge a→s, which the a→d search passes over.
	if err := gs.db.Update(func(txn *badgerdb.Txn) error { return txn.Set(edgeKey("as"), []byte("{")) }); err != nil {
		t.Fatal(err)
	}
	p, err := gs.ShortestPath(ctx, "a", "d")
	if err != nil {
		t.Fatalf("ShortestPath decoded an edge off the path: %v", err)
	}
	if nodes, _ := pathIDs(p); !slices.Equal(nodes, []string{"a", "b", "c", "d"}) {
		t.Fatalf("path = %v, want a b c d", nodes)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/graph/store/badger/ -run TestShortestPath -count=1`
Expected: FAIL. `ShortestPath decoded an edge off the path: unexpected end of JSON input`. `TestShortestPath` already passes; it pins the current paths as a characterization test.

- [ ] **Step 3: Implement**

In `ShortestPath`, update the doc comment and the BFS loop head (409-432). Everything after the loop is unchanged:

```go
// ShortestPath returns the shortest undirected path between src and dst via BFS over adjacency keys, loading
// only the path's nodes and edges. Returns an empty Path (no error) when no path exists.
func (s *Store) ShortestPath(_ context.Context, src, dst string) (store.Path, error) {
```

```go
		for len(queue) > 0 && !found {
			curr := queue[0]
			queue = queue[1:]
			adj, err := adjacentEdgeKeys(txn, curr, nil)
			if err != nil {
				return err
			}
			for _, e := range adj {
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/graph/store/... ./internal/mcp/... -count=1; go vet ./internal/graph/store/...`
Expected: PASS

- [ ] **Step 5: Docs**
`CHANGELOG.md` under `### Changed`: "`shortest_path` walks adjacency keys and decodes only the edges on the returned path."

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/store/badger/store.go internal/graph/store/badger/path_test.go CHANGELOG.md
git commit -m "perf(store): run ShortestPath BFS over adjacency keys and cover it with tests"
```

### Task B6: Store epoch in `cache.json`; sync the store before saving the cache

**Finding:** `cache.json` is written atomically and fsynced (`internal/cache/cache.go:85` → `internal/fsutil/fsutil.go:28`), but Badger runs with `SyncWrites` off (the default from `badgerOptions`), and neither `build` (`cmd/chiton/main.go:514`) nor the sync engine (`internal/ingest/engine.go:346`, `:492`) syncs the store first. After a power loss the cache can vouch for nodes the store lost. If the user deletes only the store directory, the cache keeps every entry, so `build` reports "N skipped" and silently leaves those files out.
**Priority:** P3   **Effort:** M   **Depends on:** B2 (apply after B4: the `prepare` shown includes B4's label line), C2
**Integration:** Land after Task C2, which makes `build` run through the sync engine. The `cmd/chiton/main.go` build-tail edits below then collapse into the engine's `saveCache`; keep the ordering rule (`store.Sync` before the cache write) and the epoch check in `cache.Load`.
**Behaviour change:** The `cache.json` layout changes to `{"store_epoch": ..., "entries": {...}}`. A cache from another store, or from before epochs, is discarded with a stderr note and every file is re-extracted once. **DECISION NEEDED:** recommended: discard pre-epoch caches (one full re-extract after upgrade). Alternative: adopt a pre-epoch cache once, which keeps the deleted-store hole open for existing users.

**Files:**
- Modify: `internal/graph/store/badger/schema.go` (from B2/B3: `metaEpoch`, `loadEpoch`, imports `crypto/rand`)
- Modify: `internal/graph/store/badger/store.go` (Store `epoch` field, `prepare`, compile-time checks, new `Epoch`/`Sync`)
- Modify: `internal/graph/store/store.go` (new `Syncer`), `internal/graph/store/helpers.go` (new `Sync`)
- Modify: `internal/cache/cache.go` (whole file)
- Modify: `cmd/chiton/main.go:363-370`, `454`, `512-514`; `cmd/chiton/sync.go:70`
- Modify: `internal/ingest/engine.go:346-348`, `489-495`; `internal/ingest/engine_test.go:43`, `:412`
- Test: `internal/cache/cache_test.go`, `internal/graph/store/badger/epoch_test.go`, `internal/ingest/engine_test.go`

**Interfaces:**
- Consumes: B2's `getMeta`, `setMeta`, `pfxMeta`, `(*Store).prepare`
- Produces: `store.Syncer interface { Sync() error }`; `store.Sync(gs store.GraphStore) error`; `(*badger.Store).Epoch() string`; `(*badger.Store).Sync() error`; `cache.Load(path, storeEpoch string) (*cache.Cache, error)`; `(*cache.Cache).Discarded() bool`

- [ ] **Step 1: Write the failing test**

```go
// internal/cache/cache_test.go
package cache

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func saved(t *testing.T, path, epoch string) {
	t.Helper()
	c, err := Load(path, epoch)
	if err != nil {
		t.Fatal(err)
	}
	c.Mark("file:///a.go", "sha-a")
	if err := c.Save(); err != nil {
		t.Fatal(err)
	}
}

func TestLoadKeepsEntriesForSameStore(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.json")
	saved(t, path, "epoch-1")
	c, err := Load(path, "epoch-1")
	if err != nil {
		t.Fatal(err)
	}
	if c.Changed("file:///a.go", "sha-a") || c.Discarded() {
		t.Fatalf("same store: Changed = %v, Discarded = %v; want the entry kept", c.Changed("file:///a.go", "sha-a"), c.Discarded())
	}
}

func TestLoadDiscardsEntriesFromAnotherStore(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.json")
	saved(t, path, "epoch-1")
	c, err := Load(path, "epoch-2")
	if err != nil {
		t.Fatal(err)
	}
	if !c.Changed("file:///a.go", "sha-a") || !c.Discarded() || len(c.URIs()) != 0 {
		t.Fatalf("other store: kept %v (Discarded = %v); want every entry dropped", c.URIs(), c.Discarded())
	}
}

func TestLoadDiscardsPreEpochCache(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.json")
	if err := os.WriteFile(path, []byte(`{"file:///a.go": "sha-a"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := Load(path, "epoch-1")
	if err != nil {
		t.Fatal(err)
	}
	if !c.Changed("file:///a.go", "sha-a") || !c.Discarded() {
		t.Fatal("a pre-epoch cache must be discarded")
	}
}

func TestSaveRecordsStoreEpoch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.json")
	saved(t, path, "epoch-1")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		StoreEpoch string            `json:"store_epoch"`
		Entries    map[string]string `json:"entries"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.StoreEpoch != "epoch-1" || got.Entries["file:///a.go"] != "sha-a" {
		t.Fatalf("saved cache = %+v", got)
	}
}
```

```go
// internal/graph/store/badger/epoch_test.go
package badger

import (
	"path/filepath"
	"testing"
)

func TestEpochIdentifiesStore(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "a")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	epoch := gs.Epoch()
	if epoch == "" {
		t.Fatal("new store has no epoch")
	}
	if err := gs.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if err := gs.Close(); err != nil {
		t.Fatal(err)
	}

	ro, err := OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	if ro.Epoch() != epoch {
		t.Errorf("reopened epoch = %q, want %q", ro.Epoch(), epoch)
	}
	if err := ro.Sync(); err != nil {
		t.Errorf("read-only Sync: %v", err)
	}
	ro.Close()

	other, err := Open(filepath.Join(t.TempDir(), "b"))
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	if other.Epoch() == epoch {
		t.Fatal("two stores share an epoch")
	}
}
```

Append to `internal/ingest/engine_test.go` (add `"errors"` to the imports):

```go
// syncFails is a store whose Sync always fails.
type syncFails struct{ store.GraphStore }

func (syncFails) Sync() error { return errors.New("disk gone") }

// TestCacheNotSavedWhenStoreSyncFails checks the cache is never persisted ahead of an unsynced store.
func TestCacheNotSavedWhenStoreSyncFails(t *testing.T) {
	h := newHarness(t, false, false)
	eng := New(syncFails{h.gs}, h.cache, Config{GraphPath: filepath.Join(h.graphDir, "graph")}, nil, nil, nil)
	p := h.write(t, "a.go", "package a\n\nfunc A() {}\n")
	if _, err := eng.SyncChanges(context.Background(), []Change{{Path: p, Op: OpUpsert}}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(h.graphDir, "cache.json")); !os.IsNotExist(err) {
		t.Fatalf("cache.json written although the store sync failed (stat err %v)", err)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/cache/ ./internal/graph/store/badger/ ./internal/ingest/ -run "TestLoad|TestSaveRecords|TestEpochIdentifiesStore|TestCacheNotSavedWhenStoreSyncFails" -count=1`
Expected: FAIL. `internal/cache` fails to build (`too many arguments in call to Load`) and `badger` fails to build (`gs.Epoch undefined`, `gs.Sync undefined`). `internal/ingest` builds and fails with `cache.json written although the store sync failed (stat err <nil>)`.

- [ ] **Step 3: Implement**

In `schema.go`, add `"crypto/rand"` to the imports, `metaEpoch = pfxMeta + "epoch" // random store identity (see Store.Epoch)` to the metadata consts, and:

```go
// loadEpoch returns the store's random identity, minting one on a writable store that has none.
func loadEpoch(db *badgerdb.DB, readOnly bool) (string, error) {
	var epoch string
	if err := db.View(func(txn *badgerdb.Txn) (err error) {
		epoch, err = getMeta(txn, metaEpoch)
		return err
	}); err != nil || epoch != "" || readOnly {
		return epoch, err
	}
	epoch = rand.Text()
	return epoch, setMeta(db, metaEpoch, epoch)
}
```

In `store.go`, add `epoch string // random identity minted with the store; see Epoch` to `Store` and `_ store.Syncer = (*Store)(nil)` to the compile-time checks. Replace `prepare` and add the two methods:

```go
// prepare readies a just-opened store: it checks, and when writable stamps or migrates, the key layout, loads
// the store epoch, and gives a read-only store its lazy label index.
func (s *Store) prepare() error {
	if err := prepareSchema(s.db, s.readOnly); err != nil {
		return err
	}
	var err error
	if s.epoch, err = loadEpoch(s.db, s.readOnly); err != nil {
		return err
	}
	if s.readOnly {
		s.labels = sync.OnceValues(func() ([]labelEntry, error) { return loadLabels(s.db) })
	}
	return nil
}

// Epoch returns the store's random identity, minted when the store is created, so a sidecar can tell that the
// store was replaced underneath it. It is empty only for a read-only store with no nodes.
func (s *Store) Epoch() string { return s.epoch }

// Sync makes every write so far durable; it is a no-op on a read-only store. Implements store.Syncer.
func (s *Store) Sync() error { return s.db.Sync() }
```

In `internal/graph/store/store.go`, add:

```go
// Syncer is an optional capability that makes every write so far durable, so a sidecar saved afterwards never
// records data the store could lose in a crash. Callers use the Sync helper.
type Syncer interface {
	Sync() error
}
```

Append to `internal/graph/store/helpers.go`:

```go
// Sync flushes gs's writes to stable storage when gs is a Syncer; otherwise it does nothing.
func Sync(gs GraphStore) error {
	if s, ok := gs.(Syncer); ok {
		return s.Sync()
	}
	return nil
}
```

Replace `internal/cache/cache.go` from the `Cache` type through `Save` (the package doc comment and `Changed`/`Mark`/`Delete`/`URIs` stay as they are):

```go
// Cache maps source_uri to the SHA256 hex string last seen for that file, for one store epoch.
type Cache struct {
	path      string
	epoch     string
	discarded bool

	mu      sync.RWMutex
	entries map[string]string
}

// fileFormat is cache.json's layout. Entries are only valid for the store whose epoch recorded them.
type fileFormat struct {
	StoreEpoch string            `json:"store_epoch"`
	Entries    map[string]string `json:"entries"`
}

// Load opens the cache at path for the store identified by storeEpoch. It starts empty when the file is
// missing or was written for another store (or before epochs), so a replaced store is fully re-extracted.
func Load(path, storeEpoch string) (*Cache, error) {
	c := &Cache{path: path, epoch: storeEpoch, entries: map[string]string{}}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return c, nil
	}
	if err != nil {
		return nil, err
	}
	var f fileFormat
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, err
	}
	if f.StoreEpoch != storeEpoch {
		c.discarded = true
		return c, nil
	}
	if f.Entries != nil {
		c.entries = f.Entries
	}
	return c, nil
}

// Discarded reports whether Load dropped entries recorded for a different store.
func (c *Cache) Discarded() bool { return c.discarded }
```

```go
// Save writes the cache and its store epoch to disk, creating the parent directory if needed. It writes
// atomically, so a crash mid-write leaves the previous cache intact and concurrent Saves never share a temp file.
func (c *Cache) Save() error {
	c.mu.RLock()
	data, err := json.MarshalIndent(fileFormat{StoreEpoch: c.epoch, Entries: c.entries}, "", "  ")
	c.mu.RUnlock()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(c.path), 0o755); err != nil {
		return err
	}
	return fsutil.WriteFileAtomic(c.path, append(data, '\n'), 0o644)
}
```

In `cmd/chiton/main.go`, replace `loadBuildCache` (363-370):

```go
// loadBuildCache loads the content-hash cache that build and sync share beside the graph store, noting on
// errOut when it belonged to another store and every file will be re-extracted.
func loadBuildCache(graphPath, storeEpoch string, errOut io.Writer) (*cache.Cache, error) {
	ch, err := cache.Load(filepath.Join(filepath.Dir(graphPath), "cache.json"), storeEpoch)
	if err != nil {
		return nil, fmt.Errorf("open cache: %w", err)
	}
	if ch.Discarded() {
		fmt.Fprintln(errOut, "chiton: the content cache was written for a different graph store; re-extracting every file")
	}
	return ch, nil
}
```

In `buildCmd`, change line 454 to `ch, err := loadBuildCache(cfg.GraphPath, gs.Epoch(), cmd.ErrOrStderr())`, and replace lines 512-514 (the comment and `saveErr := ch.Save()`) with:

```go
			// Persist cache progress even if the walk aborted, but only once the store's writes are durable,
			// so after a crash the cache never vouches for nodes the store lost.
			saveErr := gs.Sync()
			if saveErr == nil {
				saveErr = ch.Save()
			}
```

In `cmd/chiton/sync.go:70`, change the call to `ch, err := loadBuildCache(cfg.GraphPath, gs.Epoch(), cmd.ErrOrStderr())`.

In `internal/ingest/engine.go`, replace `Watch`'s final save (346-348) with `e.saveCache()`, and replace `saveCache` (489-495):

```go
// saveCache makes the store's writes durable, then persists the content cache, so after a crash the cache
// never vouches for data the store lost. It logs rather than returns errors, for deferred best-effort use.
func (e *Engine) saveCache() {
	if err := store.Sync(e.store); err != nil {
		e.log("sync: store sync failed, cache not saved: %v", err)
		return
	}
	if err := e.cache.Save(); err != nil {
		e.log("sync: cache save failed: %v", err)
	}
}
```

In `internal/ingest/engine_test.go`, change line 43 to `ch, err := cache.Load(filepath.Join(graphDir, "cache.json"), gs.Epoch())` and line 412 to `ch, err := cache.Load(filepath.Join(root, "chiton-out", "cache.json"), gs.Epoch())`.

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/cache/... ./internal/graph/store/... ./internal/ingest/... -count=1; go vet ./internal/cache/... ./internal/graph/store/... ./internal/ingest/... ./cmd/chiton/...; go build ./...; go test ./... -count=1`
Expected: PASS

- [ ] **Step 5: Docs**
`CHANGELOG.md` under `### Fixed`: "**`chiton build` no longer skips files the graph store never received.** `cache.json` now records the store's random epoch, and entries written for another store (for example after the store directory was deleted) are discarded, with a stderr note that every file will be re-extracted. Build and sync also make the store's writes durable before saving the cache. Existing caches predate the epoch and are discarded once." `docs/SECURITY.md:131` needs no change: `cache.json` stays plaintext and gains only a random epoch.

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/store/badger/schema.go internal/graph/store/badger/store.go internal/graph/store/badger/epoch_test.go internal/graph/store/store.go internal/graph/store/helpers.go internal/cache/cache.go internal/cache/cache_test.go cmd/chiton/main.go cmd/chiton/sync.go internal/ingest/engine.go internal/ingest/engine_test.go CHANGELOG.md
git commit -m "fix(cache): tie cache.json to the store epoch and sync the store before saving it"
```

### Task B7: Bound value-log preallocation; bounded value-log GC on close

**Finding:** `badgerOptions` (B1; formerly `store.go:73`/`84`) keeps Badger's default `ValueLogFileSize` of `1<<30 - 1`, and Badger preallocates twice that for the active value log (`badger@v4.9.6/value.go:536`). My probe measured a 2,147,483,646-byte `000001.vlog` while a writer was open, and an unclean shutdown leaves that file next to a 128 MiB `.mem`. Chiton's values stay inline in the LSM tree (below the 1 MiB `ValueThreshold`), so the log is essentially empty and `RunValueLogGC` is never called.
**Priority:** P3   **Effort:** S   **Depends on:** B1, B3
**Behaviour change:** none visible. The active value log preallocates 128 MiB instead of 2 GiB (measured with 64 MiB files: 134,217,728 bytes). A single node record over 64 MiB would now be rejected at write, which is far beyond any real record.

**Files:**
- Modify: `internal/graph/store/badger/store.go` (`badgerOptions`, `Close`, new `valueLogFileSize`, `collectValueLog`)
- Test: `internal/graph/store/badger/options_test.go`

**Interfaces:**
- Consumes: B1's `badgerOptions`; B3's `Close`/`seal`
- Produces: none

- [ ] **Step 1: Write the failing test**

```go
// internal/graph/store/badger/options_test.go
package badger

import (
	"os"
	"path/filepath"
	"testing"
)

func TestValueLogPreallocationBounded(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "graph.badger")
	gs, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	logs, err := filepath.Glob(filepath.Join(dir, "*.vlog"))
	if err != nil || len(logs) == 0 {
		t.Fatalf("value logs = %v, %v", logs, err)
	}
	// Badger preallocates twice the file cap while a writer is open, and a crash leaves the file at that size.
	const maxPrealloc = 128 << 20
	for _, p := range logs {
		st, err := os.Stat(p)
		if err != nil {
			t.Fatal(err)
		}
		if st.Size() > maxPrealloc {
			t.Errorf("%s is %d bytes while open, want at most %d", filepath.Base(p), st.Size(), maxPrealloc)
		}
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/graph/store/badger/ -run TestValueLogPreallocationBounded -count=1`
Expected: FAIL. `000001.vlog is 2147483646 bytes while open, want at most 134217728`.

- [ ] **Step 3: Implement**

In `store.go`, add the constant and the GC helper, and update `badgerOptions` and `Close`:

```go
// valueLogFileSize caps each value-log file. Badger preallocates twice this while a writer is open, and chiton's
// values stay inline in the LSM tree, so the 1 GiB default only left 2 GiB files behind after a crash.
const valueLogFileSize = 64 << 20

func badgerOptions(path string, opts Options, readOnly bool) badgerdb.Options {
	bopts := badgerdb.DefaultOptions(path).WithLogger(nil).WithValueLogFileSize(valueLogFileSize)
	if readOnly {
		bopts = bopts.WithReadOnly(true).WithBypassLockGuard(true)
	}
	return applyEncryption(bopts, opts.EncryptionKey)
}

// collectValueLog runs bounded value-log GC before a writable store closes. Values rarely reach the log, so
// this usually stops at once with ErrNoRewrite.
func (s *Store) collectValueLog() {
	for range 4 {
		if s.db.RunValueLogGC(0.5) != nil {
			return // ErrNoRewrite (nothing to reclaim) or ErrRejected
		}
	}
}

// Close collects and seals a writable store (see sealed), then releases it. GC runs first because its
// rewrites are commits the seal must cover.
func (s *Store) Close() error {
	var sealErr error
	if !s.readOnly {
		s.collectValueLog()
		sealErr = seal(s.db)
	}
	return errors.Join(sealErr, s.db.Close(), s.lock.release())
}
```

(Keep B1's doc comment on `badgerOptions`.)

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/graph/store/... -count=1; go vet ./internal/graph/store/...; go test ./... -count=1`
Expected: PASS

- [ ] **Step 5: Docs**
`CHANGELOG.md` under `### Changed`: "The Badger value log is capped at 64 MiB per file (Badger preallocates 128 MiB instead of 2 GiB while a writer is open), and a writable store runs a bounded value-log GC on close."

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/store/badger/store.go internal/graph/store/badger/options_test.go CHANGELOG.md
git commit -m "perf(store): cap value-log preallocation at 128 MiB and GC the value log on close"
```

---

## Section C: Ingest, build and live-sync pipeline

This section makes every indexer walk the tree the same way and write to the store the same way. C1 adds `internal/walk`: one ignore policy, `.chitonignore`, and a walker that never follows links and skips unreadable entries. C6 surfaces the walker's error count and adds ctx checks. C3 adds diff-based source replacement to the store and moves the engine and every meeting-domain ingester onto it. C2 then rebuilds `chiton build` on the engine's `SyncDir` as a parallel-extract, single-writer pipeline, and C8, C7, C4 and C5 harden that pipeline (symlinks at the read site, duplicate stub writes, Windows path case, fsnotify overflow). Land them in this order: **C1 → C6 → C3 → C2 → C8 → C7 → C4 → C5**. Section D's analyzer walks consume `walk.LoadPolicy`/`walk.Files` from C1. Section B's source_uri index belongs inside the badger seams C3 creates (`sourceIDs`, `putNodes`, `putEdges`, `deleteNode`, `deleteEdge`).

### Task C1: One walk policy for every indexer, plus `.chitonignore`

**Finding:** Three ignore lists disagree. `internal/ingest/engine.go:538-546` (`DefaultIgnoreDir`, used by `SyncDir` at :243 and the watcher at :314) skips `.git .chiton chiton-out node_modules vendor .hg .svn`. `internal/analyze/llm.go:439-446` (`skipDir`, used by `analyzer.go:118` and `llm.go:165`) omits `.hg`/`.svn`. `chiton build` (`cmd/chiton/main.go:463-511`) and `internal/bench/bench.go:91-114` skip nothing. No list covers `.venv`, `venv`, `__pycache__`, `target` or `dist`. The `.chitonignore` promised in `docs/PLAN.md:1405` was never built. As a result build and bench index dependency and build-output trees, and users have no way to exclude generated code.
**Priority:** P2   **Effort:** M   **Depends on:** none
**Behaviour change:** `sync once` and `sync watch` now also skip `.venv`, `venv`, `__pycache__`, `target` and `dist`. `bench` skips the full list. A repo-root `.chitonignore` is honoured everywhere. (`build` picks up the policy in C2.) DECISION NEEDED: the default list includes `vendor`, `target` and `dist` (recommended; `!vendor` in `.chitonignore` re-includes one of them). The alternative is to build in only VCS, output and `node_modules`, and leave the rest to `.chitonignore`.

**Files:**
- Create: `internal/walk/walk.go`
- Create: `internal/walk/walk_test.go`
- Modify: `internal/ingest/engine.go:231-289` (`SyncDir`), `:291-354` (`Watch`), `:538-546` (delete `DefaultIgnoreDir`)
- Modify: `internal/ingest/fswatch/watcher.go:16-32` (`Options`), `:231-268` (`ignoredDir`, `pathIgnored`, delete `splitPath`)
- Modify: `internal/ingest/fswatch/watcher_test.go:35-40` (`goOpts`)
- Modify: `internal/bench/bench.go:5-17` (imports), `:91-117`
- Test: `internal/ingest/engine_test.go`

**Interfaces:**
- Consumes: none
- Produces (Section D uses `walk.LoadPolicy` + `walk.Files` for `internal/analyze/analyzer.go:112-142` and `llm.go:160-220`, and deletes `skipDir` at `llm.go:439-446`):
  - `const walk.IgnoreFile = ".chitonignore"`
  - `var walk.DefaultDirs []string`
  - `type walk.Policy struct{…}`, `func walk.LoadPolicy(root string) (*walk.Policy, error)`, `func (p *walk.Policy) Skip(target string, isDir bool) bool`
  - `type walk.Stats struct{ Unreadable int }`
  - `func walk.Files(ctx context.Context, root string, p *walk.Policy, fn func(path string) error, onErr func(path string, err error)) (walk.Stats, error)`
  - `fswatch.Options.Skip func(path string, isDir bool) bool` (replaces `Options.IgnoreDir`)

- [ ] **Step 1: Write the failing test**

`internal/walk/walk_test.go`:

```go
package walk

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
)

// writeTree creates each slash-separated path under root with the given content.
func writeTree(t *testing.T, root string, files map[string]string) {
	t.Helper()
	for rel, body := range files {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// collect returns the root-relative, slash-separated files Files yields, sorted.
func collect(t *testing.T, root string, p *Policy) []string {
	t.Helper()
	var got []string
	if _, err := Files(context.Background(), root, p, func(path string) error {
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		got = append(got, filepath.ToSlash(rel))
		return nil
	}, nil); err != nil {
		t.Fatal(err)
	}
	slices.Sort(got)
	return got
}

func TestFilesAppliesDefaultsAndChitonignore(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, map[string]string{
		"main.go":               "",
		"node_modules/dep/x.js": "",
		".venv/lib/site.py":     "",
		"pkg/__pycache__/m.pyc": "",
		"target/debug/b.rs":     "",
		"vendor/v.go":           "",
		"gen/out.go":            "",
		"sub/gen/keep.go":       "",
		"docs/api/ref.md":       "",
		"docs/guide.md":         "",
		"a.pb.go":               "",
		"sub/b.pb.go":           "",
		IgnoreFile:              "# generated\n*.pb.go\n/gen/\ndocs/api\n!vendor\n\n[bad\n",
	})
	p, err := LoadPolicy(root)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{IgnoreFile, "docs/guide.md", "main.go", "sub/gen/keep.go", "vendor/v.go"}
	if got := collect(t, root, p); !slices.Equal(got, want) {
		t.Errorf("Files = %v, want %v", got, want)
	}
}

func TestPolicySkip(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, map[string]string{IgnoreFile: "build/\n"})
	p, err := LoadPolicy(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		rel   string
		isDir bool
		want  bool
	}{
		{"main.go", false, false},
		{"node_modules/x/index.js", false, true},
		{"pkg/.git", true, true},
		{"build", true, true},
		{"build", false, false}, // a trailing "/" matches directories only
		{"cmd/build/main.go", false, true},
		{".", true, false},
		{"../outside.go", false, false},
	} {
		if got := p.Skip(filepath.Join(root, filepath.FromSlash(tc.rel)), tc.isDir); got != tc.want {
			t.Errorf("Skip(%q, %v) = %v, want %v", tc.rel, tc.isDir, got, tc.want)
		}
	}
}

func TestFilesSkipsSymlinks(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, map[string]string{"notes.md": "# notes\n"})
	secret := filepath.Join(t.TempDir(), "id_rsa")
	if err := os.WriteFile(secret, []byte("PRIVATE KEY\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(secret, filepath.Join(root, "key.md")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if err := os.Symlink(filepath.Dir(secret), filepath.Join(root, "outside")); err != nil {
		t.Fatal(err)
	}
	p, err := LoadPolicy(root)
	if err != nil {
		t.Fatal(err)
	}
	if got := collect(t, root, p); !slices.Equal(got, []string{"notes.md"}) {
		t.Errorf("Files = %v, want only notes.md (symlinks skipped)", got)
	}
}

func TestFilesSkipsUnreadableDirs(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("chmod cannot make a directory unreadable on Windows")
	}
	if os.Geteuid() == 0 {
		t.Skip("root can read any directory")
	}
	root := t.TempDir()
	writeTree(t, root, map[string]string{"a.go": "", "locked/b.go": ""})
	locked := filepath.Join(root, "locked")
	if err := os.Chmod(locked, 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(locked, 0o755) })
	p, err := LoadPolicy(root)
	if err != nil {
		t.Fatal(err)
	}
	var got, failed []string
	st, err := Files(context.Background(), root, p, func(path string) error {
		got = append(got, filepath.Base(path))
		return nil
	}, func(path string, _ error) { failed = append(failed, path) })
	if err != nil {
		t.Fatalf("Files aborted on an unreadable dir: %v", err)
	}
	if st.Unreadable != 1 || !slices.Equal(failed, []string{locked}) {
		t.Errorf("Unreadable = %d, onErr paths = %v; want 1, [%s]", st.Unreadable, failed, locked)
	}
	if !slices.Equal(got, []string{"a.go"}) {
		t.Errorf("Files = %v, want [a.go]", got)
	}
}

func TestFilesStopsOnCancelAndMissingRoot(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, map[string]string{"a.go": ""})
	p, err := LoadPolicy(root)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Files(ctx, root, p, func(string) error { return nil }, nil); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled Files err = %v, want context.Canceled", err)
	}
	missing := filepath.Join(root, "nope")
	mp, err := LoadPolicy(missing)
	if err != nil {
		t.Fatalf("LoadPolicy(missing root) = %v, want the default policy", err)
	}
	if _, err := Files(context.Background(), missing, mp, func(string) error { return nil }, nil); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("missing root err = %v, want fs.ErrNotExist", err)
	}
}
```

Append to `internal/ingest/engine_test.go`:

```go
// TestSyncDirHonoursIgnorePolicy checks SyncDir skips the default directory list and the root's .chitonignore.
func TestSyncDirHonoursIgnorePolicy(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	for _, d := range []string{"gen", ".venv"} {
		if err := os.MkdirAll(filepath.Join(h.dir, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	kept := h.write(t, "a.go", "package src\nfunc A() {}\n")
	gen := h.write(t, filepath.Join("gen", "g.go"), "package gen\nfunc G() {}\n")
	venv := h.write(t, filepath.Join(".venv", "v.py"), "def v():\n    pass\n")
	h.write(t, ".chitonignore", "gen/\n")

	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	if len(h.nodesForURI(t, model.FileURI(kept))) == 0 {
		t.Error("a.go should be indexed")
	}
	for _, p := range []string{gen, venv} {
		if n := len(h.nodesForURI(t, model.FileURI(p))); n != 0 {
			t.Errorf("%s should be skipped, got %d node(s)", p, n)
		}
	}
}
```

Replace `goOpts` in `internal/ingest/fswatch/watcher_test.go:35-40`:

```go
func goOpts() Options {
	return Options{
		Skip: func(p string, _ bool) bool {
			for _, part := range strings.Split(filepath.ToSlash(p), "/") {
				if part == ".git" || part == "chiton-out" {
					return true
				}
			}
			return false
		},
		Accept: func(p string) bool { return strings.HasSuffix(p, ".go") },
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/walk/ ./internal/ingest/ ./internal/ingest/fswatch/ -count=1`
Expected: FAIL. `internal/walk` has no non-test Go files (undefined: `Files`, `LoadPolicy`, `IgnoreFile`). fswatch reports `unknown field Skip in struct literal of type Options`. `TestSyncDirHonoursIgnorePolicy` fails with `…gen/g.go should be skipped, got 2 node(s)`.

- [ ] **Step 3: Implement**

Create `internal/walk/walk.go`:

```go
// Package walk is the repository walk every indexer shares: one ignore policy (built-in dependency, build, and
// output directories plus the root's .chitonignore) and a file walk that honours it and never follows links.
package walk

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
	"syscall"
)

// IgnoreFile is read from the walk root: one gitignore-style name or glob per line, "#" comments, trailing "/"
// for directories, a leading or inner "/" anchors to the root, "!name" re-includes a DefaultDirs entry.
const IgnoreFile = ".chitonignore"

// DefaultDirs are directory base names skipped at any depth: VCS metadata, chiton's own output (so store and
// cache writes never feed the watcher), and dependency, virtualenv, and build-output trees.
var DefaultDirs = []string{
	".git", ".hg", ".svn",
	".chiton", "chiton-out",
	"node_modules", "vendor",
	".venv", "venv", "__pycache__",
	"target", "dist",
}

// Policy decides which paths under one root a walk skips. Build it with LoadPolicy.
type Policy struct {
	root     string
	dirNames map[string]bool
	patterns []pattern
}

type pattern struct {
	glob     string // path.Match pattern over slash-separated paths
	anchored bool   // match the root-relative path instead of each base name
	dirOnly  bool
}

// LoadPolicy returns the policy for root: DefaultDirs plus the patterns in root's IgnoreFile, if present.
func LoadPolicy(root string) (*Policy, error) {
	p := &Policy{root: filepath.Clean(root), dirNames: make(map[string]bool, len(DefaultDirs))}
	for _, d := range DefaultDirs {
		p.dirNames[d] = true
	}
	data, err := os.ReadFile(filepath.Join(root, IgnoreFile))
	if errors.Is(err, fs.ErrNotExist) || errors.Is(err, syscall.ENOTDIR) {
		return p, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", IgnoreFile, err)
	}
	p.parse(string(data))
	return p, nil
}

// parse adds IgnoreFile lines to p, dropping blank, comment, and malformed ones.
func (p *Policy) parse(data string) {
	for line := range strings.Lines(data) {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if name, ok := strings.CutPrefix(line, "!"); ok {
			delete(p.dirNames, strings.Trim(name, "/"))
			continue
		}
		pt := pattern{dirOnly: strings.HasSuffix(line, "/")}
		glob := strings.TrimSuffix(line, "/")
		pt.anchored = strings.Contains(glob, "/")
		pt.glob = strings.TrimPrefix(glob, "/")
		if _, err := path.Match(pt.glob, ""); pt.glob == "" || err != nil {
			continue
		}
		p.patterns = append(p.patterns, pt)
	}
}

// Skip reports whether target, or a directory between the root and it, is skipped. isDir says whether target
// is itself a directory; the root and paths outside it are never skipped.
func (p *Policy) Skip(target string, isDir bool) bool {
	rel, err := filepath.Rel(p.root, target)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return false
	}
	parts := strings.Split(filepath.ToSlash(rel), "/")
	for i, name := range parts {
		dir := isDir || i < len(parts)-1
		if (dir && p.dirNames[name]) || p.matches(strings.Join(parts[:i+1], "/"), name, dir) {
			return true
		}
	}
	return false
}

// matches reports whether an IgnoreFile pattern skips the entry at root-relative rel with base name name.
func (p *Policy) matches(rel, name string, isDir bool) bool {
	for _, pt := range p.patterns {
		if pt.dirOnly && !isDir {
			continue
		}
		subject := name
		if pt.anchored {
			subject = rel
		}
		if ok, _ := path.Match(pt.glob, subject); ok {
			return true
		}
	}
	return false
}

// Stats counts what a Files walk skipped.
type Stats struct {
	Unreadable int // entries below the root whose stat or directory read failed
}

// Files calls fn with each regular file under root that p keeps, in lexical order, skipping symlinks, junctions,
// and devices. Unreadable entries go to onErr and are skipped; a bad root, cancelled ctx, or fn error stops it.
func Files(ctx context.Context, root string, p *Policy, fn func(path string) error, onErr func(path string, err error)) (Stats, error) {
	var st Stats
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if cerr := ctx.Err(); cerr != nil {
			return cerr
		}
		if err != nil {
			if path == root {
				return err
			}
			st.Unreadable++
			if onErr != nil {
				onErr(path, err)
			}
			return nil
		}
		if d.IsDir() {
			if path != root && p.Skip(path, true) {
				return filepath.SkipDir
			}
			return nil
		}
		// A symlink or junction may point outside the repo; its in-repo target, if any, is walked directly.
		if !d.Type().IsRegular() || p.Skip(path, false) {
			return nil
		}
		return fn(path)
	})
	return st, err
}
```

In `internal/ingest/engine.go`, add the import `"github.com/Eric-Fernald/chiton/internal/walk"`. Delete `DefaultIgnoreDir` (`:538-546`). Replace `SyncDir` (`:231-289`) with:

```go
// SyncDir reconciles a directory tree in one pass. It (re-)syncs every supported file the walk policy keeps
// and, when prune is true, deletes indexed files under the tree that are gone from disk.
func (e *Engine) SyncDir(ctx context.Context, root string, prune bool) (*BatchSummary, error) {
	defer e.saveCache()
	sum := &BatchSummary{}
	seen := map[string]bool{}
	policy, err := walk.LoadPolicy(root)
	if err != nil {
		return sum, err
	}

	_, walkErr := walk.Files(ctx, root, policy, func(path string) error {
		if extast.ForFile(path) == nil {
			return nil
		}
		out, err := e.SyncFile(ctx, path)
		if err != nil {
			return err
		}
		seen[model.FileURI(path)] = true
		e.tally(sum, out)
		return nil
	}, func(path string, err error) { e.log("sync: skip %s: %v", path, err) })
	if walkErr != nil {
		return sum, walkErr
	}

	if prune {
		rootURI := model.FileURI(root)
		for _, uri := range e.cache.URIs() {
			if uri != rootURI && !strings.HasPrefix(uri, rootURI+"/") {
				continue
			}
			if seen[uri] {
				continue
			}
			// Prune only files that are gone from disk. A file this walk excluded (ignored dir, unsupported
			// type, or narrower root) must survive.
			if p, ok := model.PathFromFileURI(uri); ok {
				if _, err := os.Stat(p); err == nil {
					continue
				}
			}
			out, err := e.deleteURI(ctx, uri)
			if err != nil {
				return sum, err
			}
			e.tally(sum, out)
		}
	}

	e.flush(ctx, sum)
	return sum, nil
}
```

Replace `Watch` (`:291-354`) with:

```go
// Watch re-syncs changed files under root until ctx is cancelled, then drains in-flight work and re-clusters.
// Queues run on context.Background() so a cancel stops fsnotify but Drain still processes queued edits.
func (e *Engine) Watch(ctx context.Context, root string) error {
	policy, err := walk.LoadPolicy(root)
	if err != nil {
		return err
	}
	// A single-worker, longer-debounced queue serialises cluster passes.
	clusterQ := jobs.New(func(c context.Context, _ string) error {
		_, err := e.recluster(c)
		return err
	}, jobs.Options{Workers: 1, Debounce: e.cfg.ClusterDebounce})
	clusterQ.Start(context.Background())

	fileQ := jobs.New(e.syncKey, jobs.Options{
		Workers:  e.cfg.Workers,
		Debounce: e.cfg.Debounce,
		OnError: func(key string, err error) {
			e.log("sync: %s: %v", key, err)
		},
		OnDone: func(key string, _ error) {
			clusterQ.Enqueue("cluster")
		},
	})
	fileQ.Start(context.Background())

	w, err := fswatch.New(root, fileQ.Enqueue, fswatch.Options{
		Skip:    policy.Skip,
		Accept:  func(p string) bool { return extast.ForFile(p) != nil },
		OnError: func(err error) { e.log("watch: %v", err) },
		// A renamed/removed directory yields no per-child events, so prune the
		// graph for every file previously indexed under it.
		OnDirRemove: func(dir string) {
			if n := e.pruneDir(context.Background(), dir); n > 0 {
				e.log("watch: directory %s removed — pruned %d file(s)", dir, n)
				clusterQ.Enqueue("cluster")
			}
		},
	})
	if err != nil {
		fileQ.Stop()
		clusterQ.Stop()
		return err
	}

	e.log("watch: watching %s (%d dirs) — Ctrl-C to stop", root, w.Watched())
	runErr := w.Run(ctx) // blocks until ctx cancelled
	_ = w.Close()

	// Graceful shutdown: drain queued file work first (its OnDone re-arms the
	// cluster trigger), then drain the cluster queue, then a final flush.
	fileQ.Drain()
	clusterQ.Drain()

	flushCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if _, err := e.recluster(flushCtx); err != nil {
		e.log("watch: final re-cluster failed: %v", err)
	}
	if err := e.cache.Save(); err != nil {
		e.log("watch: cache save failed: %v", err)
	}

	if runErr != nil && !errors.Is(runErr, context.Canceled) {
		return runErr
	}
	return nil
}
```

In `internal/ingest/fswatch/watcher.go`, replace the `IgnoreDir` field and the `WatchRootOnly` doc in `Options` (`:17-32`):

```go
type Options struct {
	// Skip reports whether a path under root is ignored: a directory not to watch or descend into, or a file
	// whose events to drop. isDir says whether the path is known to be a directory.
	Skip func(path string, isDir bool) bool
	// Accept reports whether a file path is of interest (e.g. has a supported
	// extractor). Events for files Accept rejects are dropped.
	Accept func(path string) bool
	// OnError, if set, receives non-fatal watcher errors.
	OnError func(error)
	// OnDirRemove, if set, is called when a watched directory is removed or renamed away. fsnotify
	// sends no per-child events for this, so the consumer must prune what it indexed underneath.
	OnDirRemove func(dir string)
	// WatchRootOnly watches only root, never its subdirectories (e.g. a huge browser profile dir). Leave
	// Skip nil with it, since Skip also sees file-event paths and would drop wanted events.
	WatchRootOnly bool
}
```

Replace `ignoredDir`, `pathIgnored` and `splitPath` (`:231-268`), and drop the now-unused `"slices"` import:

```go
// ignoredDir reports whether the directory at dirPath is skipped.
func (w *Watcher) ignoredDir(dirPath string) bool {
	return w.opts.Skip != nil && w.opts.Skip(dirPath, true)
}

// pathIgnored reports whether path is skipped or lies under a skipped directory, so events under e.g.
// chiton-out/ never trigger a self-feeding re-extraction loop.
func (w *Watcher) pathIgnored(path string) bool {
	return w.opts.Skip != nil && w.opts.Skip(path, false)
}
```

In `internal/bench/bench.go`, replace the `"io/fs"` and `"path/filepath"` imports with `"github.com/Eric-Fernald/chiton/internal/walk"`, and replace the walk at `:91-117` with:

```go
	policy, err := walk.LoadPolicy(path)
	if err != nil {
		return nil, err
	}
	// Unreadable entries are skipped so the benchmark keeps going over the rest.
	_, walkErr := walk.Files(ctx, path, policy, func(p string) error {
		ex := extast.ForFile(p)
		if ex == nil {
			return nil
		}
		content, err := os.ReadFile(p)
		if err != nil {
			return nil
		}
		files++
		fileText[p] = string(content)
		fileTokens[p] = estimateTokens(len(content))

		ns, es, err := ex.ExtractFile(ctx, p)
		if err != nil {
			return nil
		}
		nodes = append(nodes, ns...)
		edges = append(edges, es...)
		return nil
	}, nil)
	if walkErr != nil {
		return nil, walkErr
	}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/walk/... ./internal/ingest/... ./internal/bench/... -count=1; go vet ./internal/walk/... ./internal/ingest/... ./internal/bench/...`
Expected: PASS. `TestFilesSkipsSymlinks` skips on Windows without Developer Mode (it runs in the Linux/macOS CI matrix). `TestFilesSkipsUnreadableDirs` skips on Windows.

- [ ] **Step 5: Docs**
- `README.md:192` (`chiton build` row): append "Skips VCS, `node_modules`, `vendor`, virtualenv, `target`/`dist` and chiton output directories, plus anything in a repo-root `.chitonignore`."
- `README.md`, after the Live sync paragraph ending at `:329`, add a `### Ignoring paths` subsection. It lists `walk.DefaultDirs` and the `.chitonignore` syntax (names/globs, trailing `/` = directories only, leading or inner `/` anchors to the root, `!name` re-includes a built-in directory, no `**`), and notes that `sync watch` reads the file at start, so it must be restarted after an edit.
- `CHANGELOG.md` `### Added`: "**`.chitonignore`.** A repo-root ignore file (names and globs, trailing `/` for directories, a leading or inner `/` anchors to the root, `!name` re-includes a built-in directory) now applies to `sync`, `sync watch`, and `bench`."
- `CHANGELOG.md` `### Changed`: "**One ignore list for every walk.** `sync`, `sync watch`, and `bench` skip `.git`, `.hg`, `.svn`, `.chiton`, `chiton-out`, `node_modules`, `vendor`, `.venv`, `venv`, `__pycache__`, `target`, and `dist` at any depth; `bench` previously skipped nothing."

- [ ] **Step 6: Commit**

```powershell
git add internal/walk internal/ingest/engine.go internal/ingest/engine_test.go internal/ingest/fswatch internal/bench/bench.go README.md CHANGELOG.md
git commit -m "feat(walk): shared ignore policy with .chitonignore for sync, watch, and bench"
```

### Task C6: Surface unreadable entries and honour cancellation in sync

**Finding:** `internal/ingest/engine.go:238-241` returned the first walk error, so one unreadable directory aborted `sync once`, whereas build logs and continues (`cmd/chiton/main.go:469-474`). The loops at `:220-226` (`SyncChanges`), `:238-258` (walk) and `:265-284` (prune) never checked `ctx`, so Ctrl-C or a timeout was ignored until the batch finished. C1's `walk.Files` already logs and skips unreadable entries and checks `ctx` per entry. This task closes the other two loops and reports the skipped count. (Section D does the same in `internal/analyze`.)
**Priority:** P3   **Effort:** S   **Depends on:** C1
**Behaviour change:** When anything was skipped, `sync once` prints `, N unreadable` before the re-cluster note. `sync git` and `sync once` stop between files on Ctrl-C and return `context canceled`.

**Files:**
- Modify: `internal/ingest/engine.go:120-127` (`BatchSummary`), `:213-229` (`SyncChanges`), `SyncDir` (as rewritten in C1)
- Modify: `cmd/chiton/sync.go:239-240`, `:249-254`
- Test: `internal/ingest/engine_test.go`

**Interfaces:**
- Consumes: `walk.Files`, `walk.Stats` (C1)
- Produces: `ingest.BatchSummary.Unreadable int`

- [ ] **Step 1: Write the failing test**

Append to `internal/ingest/engine_test.go` (add `"errors"` and `"runtime"` to its imports):

```go
// TestSyncChangesStopsOnCancel checks a cancelled context stops a batch before it writes anything.
func TestSyncChangesStopsOnCancel(t *testing.T) {
	h := newHarness(t, false, false)
	p := h.write(t, "a.go", "package src\nfunc A() {}\n")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := h.eng.SyncChanges(ctx, []Change{{Path: p, Op: OpUpsert}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("SyncChanges err = %v, want context.Canceled", err)
	}
	if n := len(h.nodesForURI(t, model.FileURI(p))); n != 0 {
		t.Errorf("cancelled batch wrote %d node(s)", n)
	}
}

// TestSyncDirCountsUnreadable checks one unreadable directory is counted and skipped, not fatal.
func TestSyncDirCountsUnreadable(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("chmod cannot make a directory unreadable on Windows")
	}
	if os.Geteuid() == 0 {
		t.Skip("root can read any directory")
	}
	h := newHarness(t, false, false)
	ctx := context.Background()
	locked := filepath.Join(h.dir, "locked")
	if err := os.MkdirAll(locked, 0o755); err != nil {
		t.Fatal(err)
	}
	h.write(t, filepath.Join("locked", "x.go"), "package locked\nfunc X() {}\n")
	ok := h.write(t, "a.go", "package src\nfunc A() {}\n")
	if err := os.Chmod(locked, 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(locked, 0o755) })

	sum, err := h.eng.SyncDir(ctx, h.dir, true)
	if err != nil {
		t.Fatalf("SyncDir aborted on an unreadable dir: %v", err)
	}
	if sum.Unreadable != 1 {
		t.Errorf("Unreadable = %d, want 1", sum.Unreadable)
	}
	if len(h.nodesForURI(t, model.FileURI(ok))) == 0 {
		t.Error("a.go should still be indexed")
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/ingest/ -run 'TestSyncChangesStopsOnCancel|TestSyncDirCountsUnreadable' -count=1`
Expected: FAIL. The build fails with `sum.Unreadable undefined (type *BatchSummary has no field or method Unreadable)`. Once the field exists, `TestSyncChangesStopsOnCancel` fails with `SyncChanges err = <nil>, want context.Canceled`.

- [ ] **Step 3: Implement**

In `internal/ingest/engine.go`, replace `BatchSummary`:

```go
// BatchSummary aggregates a SyncChanges / SyncDir run.
type BatchSummary struct {
	Upserted    int
	Deleted     int
	Skipped     int
	Unreadable  int // walk entries that could not be read, logged and skipped (SyncDir only)
	Reembedded  int
	Reclustered bool
}
```

Replace `SyncChanges`:

```go
// SyncChanges applies a batch of changes synchronously, then incrementally
// re-clusters and flushes the cache. Used by the git hooks and `sync git`.
func (e *Engine) SyncChanges(ctx context.Context, changes []Change) (*BatchSummary, error) {
	// Persist the cache even if a later change errors, so files already written
	// to the store are not needlessly re-extracted on the next run.
	defer e.saveCache()
	sum := &BatchSummary{}
	for _, ch := range changes {
		if err := ctx.Err(); err != nil {
			return sum, err
		}
		out, err := e.SyncChange(ctx, ch)
		if err != nil {
			return sum, err
		}
		e.tally(sum, out)
	}
	e.flush(ctx, sum)
	return sum, nil
}
```

In `SyncDir` (C1 version), keep the walk's stats and check `ctx` in the prune loop:

```go
// SyncDir reconciles a directory tree in one pass. It (re-)syncs every supported file the walk policy keeps
// and, when prune is true, deletes indexed files under the tree that are gone from disk.
func (e *Engine) SyncDir(ctx context.Context, root string, prune bool) (*BatchSummary, error) {
	defer e.saveCache()
	sum := &BatchSummary{}
	seen := map[string]bool{}
	policy, err := walk.LoadPolicy(root)
	if err != nil {
		return sum, err
	}

	stats, walkErr := walk.Files(ctx, root, policy, func(path string) error {
		if extast.ForFile(path) == nil {
			return nil
		}
		out, err := e.SyncFile(ctx, path)
		if err != nil {
			return err
		}
		seen[model.FileURI(path)] = true
		e.tally(sum, out)
		return nil
	}, func(path string, err error) { e.log("sync: skip %s: %v", path, err) })
	sum.Unreadable = stats.Unreadable
	if walkErr != nil {
		return sum, walkErr
	}

	if prune {
		rootURI := model.FileURI(root)
		for _, uri := range e.cache.URIs() {
			if err := ctx.Err(); err != nil {
				return sum, err
			}
			if uri != rootURI && !strings.HasPrefix(uri, rootURI+"/") {
				continue
			}
			if seen[uri] {
				continue
			}
			// Prune only files that are gone from disk. A file this walk excluded (ignored dir, unsupported
			// type, or narrower root) must survive.
			if p, ok := model.PathFromFileURI(uri); ok {
				if _, err := os.Stat(p); err == nil {
					continue
				}
			}
			out, err := e.deleteURI(ctx, uri)
			if err != nil {
				return sum, err
			}
			e.tally(sum, out)
		}
	}

	e.flush(ctx, sum)
	return sum, nil
}
```

In `cmd/chiton/sync.go`, change the `sync once` summary at `:239-240`:

```go
			fmt.Fprintf(cmd.OutOrStdout(), "sync once: %d upserted, %d deleted, %d skipped, %d re-embedded%s%s\n",
				sum.Upserted, sum.Deleted, sum.Skipped, sum.Reembedded, unreadableNote(sum.Unreadable), reclusterNote(sum.Reclustered))
```

and add below `reclusterNote`:

```go
// unreadableNote reports walk entries skipped as unreadable, or "" when there were none.
func unreadableNote(n int) string {
	if n == 0 {
		return ""
	}
	return fmt.Sprintf(", %d unreadable", n)
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/ingest/... ./cmd/chiton/... -count=1; go vet ./internal/ingest/... ./cmd/chiton/...`
Expected: PASS (`TestSyncDirCountsUnreadable` skips on Windows).

- [ ] **Step 5: Docs**
`CHANGELOG.md` `### Fixed`: "**`sync once` survives unreadable directories and honours Ctrl-C.** A permission error below the root is logged and counted (`N unreadable` in the summary) instead of aborting the run, and `sync git`/`sync once` stop between files when cancelled."

- [ ] **Step 6: Commit**

```powershell
git add internal/ingest/engine.go internal/ingest/engine_test.go cmd/chiton/sync.go CHANGELOG.md
git commit -m "fix(ingest): count unreadable walk entries and stop sync batches on cancel"
```

### Task C3: Diff-based source replacement that keeps other sources' edges

**Finding:** `SyncFile` calls `DeleteBySourceURI` (`internal/ingest/engine.go:156`) and then re-upserts. Badger's `deleteNode` drops every adjacent edge (`internal/graph/store/badger/store.go:248-265`). So after any edit, the symbol's `DECIDED_IN`/`DISCUSSED_IN`/`MENTIONED` edges from meeting linkage and the doc section's concept `DESCRIBES` edges are gone. They never come back: `LinkService.Pending` skips meetings already marked `extraction_status=done` (`internal/extract/meeting/linkage.go:104`). Decision and rationale nodes share the meeting's source URI (`linkage.go:210,231`), so Slack re-ingest (`internal/ingest/slack/slack.go:113`), Teams (`internal/ingest/teams/source.go:104`) and `meeting ingest` (`cmd/chiton/meeting.go:50`) delete them outright. URL (`internal/ingest/url/ingest.go:108`) and email (`internal/ingest/email/email.go:132`) re-ingest drop concept edges the same way. Edges record no owning source (`internal/graph/model/edge.go:55-64`). This task defines ownership as a rule instead of adding an on-disk field: a source owns the `EXTRACTED` edges leaving its nodes. Linkage, concept and deterministic-finding edges are all `INFERRED`/`AMBIGUOUS` or leave another source's node, so the rule needs no migration and no per-edge storage (about 60 bytes × 288k edges avoided).
**Priority:** P1   **Effort:** L   **Depends on:** none
**Integration:** In the execution order Task D1 has already landed, so `SyncFile` fills `prior` via `e.analyzer.ReadPriorFindings(ctx, uri)` (guarded by `e.cfg.Reanalyze && e.analyzer != nil`) just before `DeleteBySourceURI`. When you replace that delete with `store.ReplaceSources`, keep the `ReadPriorFindings` call immediately before it and keep passing `prior` to `ReanalyzeExtracted`. Tasks B3 (indexes) and C2 (build) build on the seams this task creates.
**Behaviour change:**
- Re-syncing a file, or re-ingesting a meeting, Slack or Teams conversation, URL or email, now keeps edges other sources attached to nodes that survive. Nodes the new extraction drops still go with all their edges.
- A surviving node whose kind, label and attrs are unchanged keeps its `extraction_status` and embedding. That means fewer re-offered extraction and linkage tasks, and an unchanged meeting is not re-offered for linkage.
- A changed node loses its embedding so `chiton embed` redoes it. Previously every node of the file lost its embedding.
- Decision and rationale nodes get `source_uri = decision://<meeting-uri>`. A `source_prefix`/scope substring on the meeting URI still matches. Legacy ones are moved on the meeting's next re-ingest.

**Files:**
- Modify: `internal/graph/store/store.go` (add `SourceUpdate`, `SourceReplacer`)
- Modify: `internal/graph/store/helpers.go` (add `ReplaceSources`)
- Modify: `internal/graph/store/badger/store.go:41-50` (interface assertions), `:248-265` (`deleteNode`), `:277-325` (`UpsertNodes`/`UpsertEdges`), `:555-591` (`DeleteBySourceURI`); add `putNodes`, `putEdges`, `deleteEdge`, `sourceIDs`, `ReplaceSources`, `replaceSource`, `dropDisownedEdges`, `sameContent`, `contentAttrs`
- Modify: `internal/ingest/engine.go:1-2` (package doc), `:129-197` (`SyncFile`)
- Modify: `internal/extract/meeting/graph.go:14-22`, `:141-152` (`Upsert`); add `DecisionURI`, `rehomeDecisions`
- Modify: `internal/extract/meeting/linkage.go:174-176`, `:206-216`, `:227-237`, `:373-400`
- Modify: `cmd/chiton/meeting.go:47-54`, `internal/ingest/teams/source.go:104-107`, `internal/ingest/slack/slack.go:106-130`, `internal/ingest/url/ingest.go:107-110,185-192`, `internal/ingest/email/email.go:130-143`
- Create: `internal/graph/store/badger/replace_test.go`
- Test: `internal/ingest/engine_test.go`, `internal/extract/meeting/linkage_test.go`

**Interfaces:**
- Consumes: none
- Produces:
  - `type store.SourceUpdate struct { URI string; Nodes []model.Node; Edges []model.Edge; Fresh bool }`
  - `type store.SourceReplacer interface { ReplaceSources(ctx context.Context, updates []store.SourceUpdate) ([]int, error) }`
  - `func store.ReplaceSources(ctx context.Context, gs store.GraphStore, updates ...store.SourceUpdate) ([]int, error)`
  - `func (s *badger.Store) ReplaceSources(ctx context.Context, updates []store.SourceUpdate) ([]int, error)`
  - `const meeting.DecisionURIScheme = "decision://"`, `func meeting.DecisionURI(meetingURI string) string`
  - Badger seams for Section B: `sourceIDs(txn, uris []string) (map[string][]string, error)` is the only source_uri lookup (B's index replaces its body). `putNodes`/`putEdges`/`deleteNode`/`deleteEdge` are the only write paths, so B's index maintenance goes there.

- [ ] **Step 1: Write the failing test**

Create `internal/graph/store/badger/replace_test.go`:

```go
package badger

import (
	"context"
	"path/filepath"
	"slices"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

func openTestStore(t *testing.T) *Store {
	t.Helper()
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	return gs
}

func edgeIDs(t *testing.T, gs *Store) []string {
	t.Helper()
	edges, err := gs.AllEdges(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, e := range edges {
		ids = append(ids, e.ID)
	}
	slices.Sort(ids)
	return ids
}

const fileURI = "file:///a.go"

func sym(id, label, doc string) model.Node {
	return model.Node{ID: id, Kind: model.KindSymbol, Label: label, SourceURI: fileURI,
		Provenance: model.ProvExtracted, Attrs: map[string]any{"docstring": doc}}
}

func owned(id, src, dst string, kind model.EdgeKind) model.Edge {
	return model.Edge{ID: id, Src: src, Dst: dst, Kind: kind, Provenance: model.ProvExtracted}
}

var fileNode = model.Node{ID: "f", Kind: model.KindFile, Label: "a.go", SourceURI: fileURI, Provenance: model.ProvExtracted}

// TestReplaceSourcesKeepsForeignEdges replays edit-then-rename: an edit keeps edges other sources attached
// to a surviving symbol and drops the owned edge it no longer emits; a rename drops the symbol and its edges.
func TestReplaceSourcesKeepsForeignEdges(t *testing.T) {
	ctx := context.Background()
	gs := openTestStore(t)
	first := store.SourceUpdate{URI: fileURI,
		Nodes: []model.Node{fileNode, sym("foo", "Foo", "v1"), sym("bar", "Bar", "v1")},
		Edges: []model.Edge{owned("f-foo", "f", "foo", model.EdgeContains), owned("f-bar", "f", "bar", model.EdgeContains),
			owned("foo-bar", "foo", "bar", model.EdgeCalls)},
	}
	if _, err := gs.ReplaceSources(ctx, []store.SourceUpdate{first}); err != nil {
		t.Fatal(err)
	}
	// Meeting linkage attaches a decision and a meeting to Foo; both symbols have vectors.
	if err := gs.UpsertNodes(ctx, []model.Node{
		{ID: "dec", Kind: model.KindDecision, Label: "keep Foo", SourceURI: "decision://meeting://m"},
		{ID: "mtg", Kind: model.KindMeeting, Label: "standup", SourceURI: "meeting://m"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{
		{ID: "dec-foo", Src: "dec", Dst: "foo", Kind: model.EdgeDecidedIn, Provenance: model.ProvInferred},
		{ID: "foo-mtg", Src: "foo", Dst: "mtg", Kind: model.EdgeDiscussedIn, Provenance: model.ProvInferred},
	}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "foo", Vec: []float32{1}}, {NodeID: "bar", Vec: []float32{1}}}); err != nil {
		t.Fatal(err)
	}

	// Edit: Foo is unchanged but no longer calls Bar, and Bar's docstring changes.
	edit := store.SourceUpdate{URI: fileURI,
		Nodes: []model.Node{fileNode, sym("foo", "Foo", "v1"), sym("bar", "Bar", "v2")},
		Edges: first.Edges[:2],
	}
	if _, err := gs.ReplaceSources(ctx, []store.SourceUpdate{edit}); err != nil {
		t.Fatal(err)
	}
	if got, want := edgeIDs(t, gs), []string{"dec-foo", "f-bar", "f-foo", "foo-mtg"}; !slices.Equal(got, want) {
		t.Fatalf("edges after edit = %v, want %v", got, want)
	}
	if got, _ := gs.EmbeddingIDs(ctx); !slices.Equal(got, []string{"foo"}) {
		t.Fatalf("embeddings after edit = %v, want only unchanged foo's", got)
	}

	// Rename Foo away: it goes, taking every edge attached to it.
	rename := store.SourceUpdate{URI: fileURI, Nodes: []model.Node{fileNode, sym("bar", "Bar", "v2")}, Edges: first.Edges[1:2]}
	deleted, err := gs.ReplaceSources(ctx, []store.SourceUpdate{rename})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(deleted, []int{1}) {
		t.Errorf("deleted = %v, want [1]", deleted)
	}
	if got := edgeIDs(t, gs); !slices.Equal(got, []string{"f-bar"}) {
		t.Fatalf("edges after rename = %v, want [f-bar]", got)
	}
	for _, id := range []string{"dec", "mtg"} {
		if n, _ := gs.GetNode(ctx, id); n == nil {
			t.Errorf("other source's node %s was deleted", id)
		}
	}
}

// TestReplaceSourcesCarriesStatusOfUnchangedNodes checks an unchanged node keeps extraction_status and a
// changed one loses it.
func TestReplaceSourcesCarriesStatusOfUnchangedNodes(t *testing.T) {
	ctx := context.Background()
	gs := openTestStore(t)
	sec := func(text string) model.Node {
		return model.Node{ID: "s", Kind: model.KindDocSection, Label: "Intro", SourceURI: fileURI,
			Attrs: map[string]any{"text": text, "level": 1}}
	}
	replace := func(n model.Node) {
		t.Helper()
		if _, err := gs.ReplaceSources(ctx, []store.SourceUpdate{{URI: fileURI, Nodes: []model.Node{fileNode, n}}}); err != nil {
			t.Fatal(err)
		}
	}
	status := func() string {
		t.Helper()
		n, err := gs.GetNode(ctx, "s")
		if err != nil || n == nil {
			t.Fatalf("GetNode(s) = %v, %v", n, err)
		}
		return model.AttrString(n.Attrs, "extraction_status")
	}
	replace(sec("hello"))
	done := sec("hello")
	done.Attrs["extraction_status"] = "done"
	if err := gs.UpsertNodes(ctx, []model.Node{done}); err != nil {
		t.Fatal(err)
	}
	replace(sec("hello"))
	if got := status(); got != "done" {
		t.Errorf("unchanged section status = %q, want done", got)
	}
	replace(sec("rewritten"))
	if got := status(); got != "" {
		t.Errorf("changed section status = %q, want it cleared", got)
	}
}
```

Append to `internal/ingest/engine_test.go`:

```go
// TestResyncKeepsForeignEdges reproduces the lost-linkage bug: an edit keeps a meeting's edges on a surviving
// symbol, and renaming the symbol drops them with it.
func TestResyncKeepsForeignEdges(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	p := h.write(t, "foo.go", "package src\nfunc Hello() {}\n")
	helloID := model.NewNodeID(model.FileURI(p), "Hello")
	if _, err := h.eng.SyncFile(ctx, p); err != nil {
		t.Fatal(err)
	}
	const decID, mtgID = "dec", "mtg"
	if err := h.gs.UpsertNodes(ctx, []model.Node{
		{ID: decID, Kind: model.KindDecision, Label: "keep Hello", SourceURI: "decision://meeting://standup"},
		{ID: mtgID, Kind: model.KindMeeting, Label: "standup", SourceURI: "meeting://standup"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := h.gs.UpsertEdges(ctx, []model.Edge{
		{ID: "d", Src: decID, Dst: helloID, Kind: model.EdgeDecidedIn, Provenance: model.ProvInferred},
		{ID: "m", Src: helloID, Dst: mtgID, Kind: model.EdgeDiscussedIn, Provenance: model.ProvInferred},
	}); err != nil {
		t.Fatal(err)
	}

	h.write(t, "foo.go", "package src\nfunc Hello() {}\nfunc World() {}\n")
	if _, err := h.eng.SyncFile(ctx, p); err != nil {
		t.Fatal(err)
	}
	if !h.hasEdge(t, decID, helloID, model.EdgeDecidedIn) || !h.hasEdge(t, helloID, mtgID, model.EdgeDiscussedIn) {
		t.Fatal("editing foo.go deleted the meeting edges on the surviving Hello symbol")
	}

	h.write(t, "foo.go", "package src\nfunc Greet() {}\n")
	if _, err := h.eng.SyncFile(ctx, p); err != nil {
		t.Fatal(err)
	}
	if h.edgeTouches(t, helloID) {
		t.Error("edges still reference the renamed-away Hello symbol")
	}
	if n, _ := h.gs.GetNode(ctx, decID); n == nil {
		t.Error("the decision node itself must survive")
	}
}
```

Append to `internal/extract/meeting/linkage_test.go`:

```go
// TestReingestKeepsLinkage checks re-ingesting a conversation keeps its decisions and code links, re-offers
// linkage only when the transcript changed, and files decisions under DecisionURI.
func TestReingestKeepsLinkage(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	tr := &Transcript{Title: "Retry Policy Sync", SourceURI: "slack://C1",
		Turns: []Turn{{Speaker: "Alice", Text: "Cap retries in fetchAll."}}}
	if _, err := Upsert(ctx, gs, tr, time.Now(), "conversation"); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertNodes(ctx, []model.Node{symbolNode("fetchAll", "file://client.go")}); err != nil {
		t.Fatal(err)
	}
	svc := NewLinkService(gs, 0.5)
	meetingID := model.NewNodeID("slack://C1", tr.Title)
	if _, err := svc.Submit(ctx, LinkResult{TaskID: meetingID, Decisions: []DecisionResult{
		{Summary: "Cap retries at 3", Confidence: 0.9, References: []string{"fetchAll"}},
	}}); err != nil {
		t.Fatal(err)
	}

	if _, err := Upsert(ctx, gs, tr, time.Now(), "conversation"); err != nil {
		t.Fatal(err)
	}
	if specs, _ := svc.Pending(ctx, 0); len(specs) != 0 {
		t.Errorf("unchanged re-ingest re-offered %d linkage task(s)", len(specs))
	}

	tr.Turns = append(tr.Turns, Turn{Speaker: "Bob", Text: "Agreed."})
	if _, err := Upsert(ctx, gs, tr, time.Now(), "conversation"); err != nil {
		t.Fatal(err)
	}
	nodes, _ := gs.AllNodes(ctx)
	var decisions []model.Node
	for _, n := range nodes {
		if n.Kind == model.KindDecision {
			decisions = append(decisions, n)
		}
	}
	if len(decisions) != 1 || decisions[0].SourceURI != DecisionURI("slack://C1") {
		t.Fatalf("decisions after re-ingest = %+v, want one under %s", decisions, DecisionURI("slack://C1"))
	}
	edges, _ := gs.AllEdges(ctx)
	ek := map[model.EdgeKind]int{}
	for _, e := range edges {
		ek[e.Kind]++
	}
	if ek[model.EdgeDecidedIn] != 2 || ek[model.EdgeDiscussedIn] != 1 {
		t.Errorf("DECIDED_IN=%d DISCUSSED_IN=%d after re-ingest, want 2 and 1", ek[model.EdgeDecidedIn], ek[model.EdgeDiscussedIn])
	}
	if specs, _ := svc.Pending(ctx, 0); len(specs) != 1 {
		t.Errorf("changed transcript should be re-offered once, got %d", len(specs))
	}
}

// TestUpsertRehomesLegacyDecisions checks a decision stored under the meeting's own URI survives re-ingest.
func TestUpsertRehomesLegacyDecisions(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	tr := &Transcript{Title: "Sync", SourceURI: "meeting://sync", Turns: []Turn{{Speaker: "A", Text: "hi"}}}
	if _, err := Upsert(ctx, gs, tr, time.Now(), "meeting"); err != nil {
		t.Fatal(err)
	}
	legacy := model.Node{ID: "old-dec", Kind: model.KindDecision, Label: "legacy", SourceURI: "meeting://sync"}
	if err := gs.UpsertNodes(ctx, []model.Node{legacy}); err != nil {
		t.Fatal(err)
	}
	if _, err := Upsert(ctx, gs, tr, time.Now(), "meeting"); err != nil {
		t.Fatal(err)
	}
	n, _ := gs.GetNode(ctx, "old-dec")
	if n == nil || n.SourceURI != DecisionURI("meeting://sync") {
		t.Fatalf("legacy decision after re-ingest = %+v, want it moved to %s", n, DecisionURI("meeting://sync"))
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/graph/store/badger/ ./internal/ingest/ ./internal/extract/meeting/ -run 'TestReplaceSources|TestResyncKeepsForeignEdges|TestReingestKeepsLinkage|TestUpsertRehomesLegacyDecisions' -count=1`
Expected: FAIL. The badger and meeting packages fail to compile (`gs.ReplaceSources undefined (type *Store has no field or method ReplaceSources)`, `undefined: store.SourceUpdate`, `undefined: DecisionURI`). `TestResyncKeepsForeignEdges` uses only existing APIs, so it compiles and fails with `editing foo.go deleted the meeting edges on the surviving Hello symbol`.

- [ ] **Step 3: Implement**

Append to `internal/graph/store/store.go`:

```go
// SourceUpdate is one source's complete new extraction for ReplaceSources. Nodes may include other sources'
// nodes (pkg:// stubs, persons), which are only upserted.
type SourceUpdate struct {
	URI   string
	Nodes []model.Node
	Edges []model.Edge
	Fresh bool // the caller knows no stored node has source_uri URI, so the prior-node lookup is skipped
}

// SourceReplacer is an optional capability replacing whole sources in one write batch while keeping edges other
// sources attached to surviving nodes. Callers use the ReplaceSources helper.
type SourceReplacer interface {
	// ReplaceSources applies updates and returns, per update, how many stale nodes it deleted.
	ReplaceSources(ctx context.Context, updates []SourceUpdate) ([]int, error)
}
```

Append to `internal/graph/store/helpers.go`:

```go
// ReplaceSources makes each update its source's stored content via SourceReplacer, else by DeleteBySourceURI
// plus upserts, which also drops edges other sources attached. It returns stale-node counts per update.
func ReplaceSources(ctx context.Context, gs GraphStore, updates ...SourceUpdate) ([]int, error) {
	if r, ok := gs.(SourceReplacer); ok {
		return r.ReplaceSources(ctx, updates)
	}
	deleted := make([]int, len(updates))
	for i, u := range updates {
		if !u.Fresh {
			n, err := gs.DeleteBySourceURI(ctx, u.URI)
			if err != nil {
				return deleted, err
			}
			deleted[i] = n
		}
		if err := gs.UpsertNodes(ctx, u.Nodes); err != nil {
			return deleted, err
		}
		if err := gs.UpsertEdges(ctx, u.Edges); err != nil {
			return deleted, err
		}
	}
	return deleted, nil
}
```

In `internal/graph/store/badger/store.go`, add `store.SourceReplacer = (*Store)(nil)` to the compile-time assertion block (`:42-50`) and add `"maps"` to the imports. Replace `deleteNode` (`:248-265`) with `deleteEdge` + `deleteNode`:

```go
// deleteEdge queues deletion of edge e's record and both adjacency keys.
func deleteEdge(wb *badgerdb.WriteBatch, e *model.Edge) error {
	for _, k := range [][]byte{edgeKey(e.ID), srcKey(e), dstKey(e)} {
		if err := wb.Delete(k); err != nil {
			return err
		}
	}
	return nil
}

// deleteNode queues deletion of node id, its embedding, and every adjacent edge with its adjacency keys.
func deleteNode(txn *badgerdb.Txn, wb *badgerdb.WriteBatch, id string) error {
	adj, err := adjacentEdgeKeys(txn, id, nil)
	if err != nil {
		return err
	}
	for _, k := range [][]byte{nodeKey(id), embKey(id)} {
		if err := wb.Delete(k); err != nil {
			return err
		}
	}
	for i := range adj {
		if err := deleteEdge(wb, &adj[i]); err != nil {
			return err
		}
	}
	return nil
}
```

Replace `UpsertNodes`/`UpsertEdges` (`:279-325`) with the shared writers:

```go
// putNodes queues nodes into wb, stamping CreatedAt when unset and UpdatedAt always.
func putNodes(wb *badgerdb.WriteBatch, nodes []model.Node, now time.Time) error {
	for i := range nodes {
		n := nodes[i]
		if n.CreatedAt.IsZero() {
			n.CreatedAt = now
		}
		n.UpdatedAt = now
		data, err := json.Marshal(n)
		if err != nil {
			return fmt.Errorf("marshal node %s: %w", n.ID, err)
		}
		if err := wb.Set(nodeKey(n.ID), data); err != nil {
			return err
		}
	}
	return nil
}

// putEdges queues edges and their adjacency keys into wb, stamping CreatedAt when unset.
func putEdges(wb *badgerdb.WriteBatch, edges []model.Edge, now time.Time) error {
	for i := range edges {
		e := edges[i]
		if e.CreatedAt.IsZero() {
			e.CreatedAt = now
		}
		data, err := json.Marshal(e)
		if err != nil {
			return fmt.Errorf("marshal edge %s: %w", e.ID, err)
		}
		eid := []byte(e.ID)
		if err := wb.Set(edgeKey(e.ID), data); err != nil {
			return err
		}
		if err := wb.Set(srcKey(&e), eid); err != nil {
			return err
		}
		if err := wb.Set(dstKey(&e), eid); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) UpsertNodes(_ context.Context, nodes []model.Node) error {
	now := time.Now()
	return s.batch(func(wb *badgerdb.WriteBatch) error { return putNodes(wb, nodes, now) })
}

func (s *Store) UpsertEdges(_ context.Context, edges []model.Edge) error {
	now := time.Now()
	return s.batch(func(wb *badgerdb.WriteBatch) error { return putEdges(wb, edges, now) })
}
```

Replace `DeleteBySourceURI` (`:555-591`) with `sourceIDs` + `DeleteBySourceURI`:

```go
// sourceIDs returns the IDs of stored nodes whose source_uri is in uris, keyed by URI. One URI gets a raw-bytes
// probe scan; several share one scan decoding each record's source_uri. Section B's index replaces this body.
func sourceIDs(txn *badgerdb.Txn, uris []string) (map[string][]string, error) {
	out := make(map[string][]string, len(uris))
	if len(uris) == 0 {
		return out, nil
	}
	want := make(map[string]bool, len(uris))
	for _, u := range uris {
		want[u] = true
	}
	var probe []byte
	if len(uris) == 1 {
		probe = stringProbe(uris[0])
	}
	err := scan(txn, pfxNode, func(_, v []byte) error {
		if probe != nil && !bytes.Contains(v, probe) {
			return nil
		}
		var n struct {
			ID        string `json:"id"`
			SourceURI string `json:"source_uri"`
		}
		if err := json.Unmarshal(v, &n); err != nil {
			return err
		}
		if want[n.SourceURI] {
			out[n.SourceURI] = append(out[n.SourceURI], n.ID)
		}
		return nil
	})
	return out, err
}

// DeleteBySourceURI removes all nodes whose source_uri matches uri and
// all edges adjacent to those nodes. Returns the number of nodes deleted.
func (s *Store) DeleteBySourceURI(_ context.Context, uri string) (int, error) {
	deleted := 0
	err := s.batch(func(wb *badgerdb.WriteBatch) error {
		return s.db.View(func(txn *badgerdb.Txn) error {
			byURI, err := sourceIDs(txn, []string{uri})
			if err != nil {
				return err
			}
			for _, id := range byURI[uri] {
				if err := deleteNode(txn, wb, id); err != nil {
					return err
				}
			}
			deleted = len(byURI[uri])
			return nil
		})
	})
	return deleted, err
}
```

Add after `DeleteByIDs`:

```go
// ReplaceSources applies each update as its source's complete new extraction in one write batch. Implements
// store.SourceReplacer; see replaceSource for what survives.
func (s *Store) ReplaceSources(_ context.Context, updates []store.SourceUpdate) ([]int, error) {
	deleted := make([]int, len(updates))
	writes := make([][]model.Node, len(updates))
	now := time.Now()
	err := s.batch(func(wb *badgerdb.WriteBatch) error {
		if err := s.db.View(func(txn *badgerdb.Txn) error {
			var uris []string
			for _, u := range updates {
				if !u.Fresh {
					uris = append(uris, u.URI)
				}
			}
			prior, err := sourceIDs(txn, uris)
			if err != nil {
				return err
			}
			for i, u := range updates {
				if deleted[i], writes[i], err = replaceSource(txn, wb, u, prior[u.URI]); err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return err
		}
		// Every delete is queued before any write, so a later Set for the same key wins.
		for i, u := range updates {
			if err := putNodes(wb, writes[i], now); err != nil {
				return err
			}
			if err := putEdges(wb, u.Edges, now); err != nil {
				return err
			}
		}
		return nil
	})
	return deleted, err
}

// replaceSource queues deletion of prior nodes u drops (with every edge) and of owned edges u no longer emits,
// and returns u.Nodes with unchanged nodes' extraction_status carried over. Changed nodes lose their vector.
func replaceSource(txn *badgerdb.Txn, wb *badgerdb.WriteBatch, u store.SourceUpdate, prior []string) (int, []model.Node, error) {
	nodes := u.Nodes
	if len(prior) == 0 {
		return 0, nodes, nil
	}
	index := make(map[string]int, len(nodes))
	for i := range nodes {
		index[nodes[i].ID] = i
	}
	keepEdge := make(map[string]bool, len(u.Edges))
	for i := range u.Edges {
		keepEdge[u.Edges[i].ID] = true
	}
	deleted, cloned := 0, false
	for _, id := range prior {
		i, ok := index[id]
		if !ok {
			if err := deleteNode(txn, wb, id); err != nil {
				return 0, nil, err
			}
			deleted++
			continue
		}
		if err := dropDisownedEdges(txn, wb, id, keepEdge); err != nil {
			return 0, nil, err
		}
		old, err := loadNode(txn, id)
		if err != nil {
			return 0, nil, err
		}
		if old == nil {
			continue
		}
		if !sameContent(old, &nodes[i]) {
			// Its embedded text may have changed, so drop the vector for `chiton embed` to redo.
			if err := wb.Delete(embKey(id)); err != nil {
				return 0, nil, err
			}
			continue
		}
		st := model.AttrString(old.Attrs, statusAttr)
		if st == "" || model.AttrString(nodes[i].Attrs, statusAttr) != "" {
			continue
		}
		if !cloned {
			nodes, cloned = slices.Clone(nodes), true
		}
		attrs := maps.Clone(nodes[i].Attrs)
		if attrs == nil {
			attrs = map[string]any{}
		}
		attrs[statusAttr] = st
		nodes[i].Attrs = attrs
	}
	return deleted, nodes, nil
}

// dropDisownedEdges queues deletion of edges leaving node id that keep omits and that id's source owns. A source
// owns its nodes' outgoing EXTRACTED edges; linkage, concept, and finding edges are INFERRED or start elsewhere.
func dropDisownedEdges(txn *badgerdb.Txn, wb *badgerdb.WriteBatch, id string, keep map[string]bool) error {
	adj, err := adjacentEdgeKeys(txn, id, nil)
	if err != nil {
		return err
	}
	for i := range adj {
		if adj[i].Src != id || keep[adj[i].ID] {
			continue
		}
		e, err := loadJSON[model.Edge](txn, edgeKey(adj[i].ID))
		if err != nil {
			return err
		}
		if e != nil && e.Provenance != model.ProvExtracted {
			continue
		}
		if err := deleteEdge(wb, &adj[i]); err != nil {
			return err
		}
	}
	return nil
}

// statusAttr is the attr the subagent flows set once they have processed a node. It is state, not content.
const statusAttr = "extraction_status"

// sameContent reports whether a and b agree on kind, label, and every attr but statusAttr: the inputs to a
// node's embedded text and to its subagent task.
func sameContent(a, b *model.Node) bool {
	return a.Kind == b.Kind && a.Label == b.Label && bytes.Equal(contentAttrs(a.Attrs), contentAttrs(b.Attrs))
}

// contentAttrs encodes attrs minus statusAttr as JSON. Sorted keys and uniform number encoding make a decoded
// stored map compare equal to a fresh one.
func contentAttrs(attrs map[string]any) []byte {
	m := make(map[string]any, len(attrs))
	for k, v := range attrs {
		if k != statusAttr {
			m[k] = v
		}
	}
	b, _ := json.Marshal(m)
	return b
}
```

In `internal/ingest/engine.go`, update the package doc (`:1-2`):

```go
// Package ingest is chiton's live-sync engine. It re-extracts changed files and replaces their nodes by diff
// (store.ReplaceSources), so edges other sources attached to surviving symbols stay.
```

Replace `SyncFile` (`:129-197`) with:

```go
// SyncFile (re-)extracts a single file into the graph and hands a vanished file to DeleteFile.
// Read and parse failures are skips, not errors, so a watcher never stops on one bad file.
func (e *Engine) SyncFile(ctx context.Context, path string) (Outcome, error) {
	ex := extast.ForFile(path)
	if ex == nil {
		return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: "unsupported file type"}, nil
	}
	uri := model.FileURI(path)
	content, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return e.deleteURI(ctx, uri)
		}
		return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: "read: " + err.Error()}, nil
	}

	hexSum := model.ContentSHA(content)
	if !e.cache.Changed(uri, hexSum) {
		return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: "unchanged"}, nil
	}

	// Extract the same bytes that were hashed, so the cached hash always matches the extracted content.
	nodes, edges, err := ex.ExtractSource(ctx, path, content)
	if err != nil {
		return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: "extract: " + err.Error()}, nil
	}

	deleted, err := store.ReplaceSources(ctx, e.store, store.SourceUpdate{URI: uri, Nodes: nodes, Edges: edges})
	if err != nil {
		return Outcome{}, fmt.Errorf("replace %s: %w", uri, err)
	}
	e.cache.Mark(uri, hexSum)

	ids := make([]string, 0, len(nodes))
	for _, n := range nodes {
		ids = append(ids, n.ID)
	}

	out := Outcome{
		Path:          path,
		Op:            OpUpsert,
		NodesUpserted: len(nodes),
		EdgesUpserted: len(edges),
		NodesDeleted:  deleted[0],
		TouchedIDs:    ids,
	}

	if e.cfg.Reembed && e.embedder != nil {
		n, err := e.reembed(ctx, nodes)
		if err != nil {
			e.log("sync: %s: re-embed failed: %v", path, err)
		}
		out.Reembedded = n
	}
	if e.cfg.Reanalyze && e.analyzer != nil {
		if _, err := e.analyzer.AnalyzeExtracted(ctx, path, content, nodes, edges); err != nil {
			e.log("sync: %s: re-analyze failed: %v", path, err)
		}
	}

	e.markTouched(ids)
	return out, nil
}
```

In `internal/extract/meeting/graph.go`, replace the URI scheme block (`:14-22`):

```go
// URI schemes for meeting-domain nodes.
const (
	// MeetingURIScheme prefixes a meeting's (and its turns') source_uri, so re-ingest replaces the whole
	// meeting via store.ReplaceSources.
	MeetingURIScheme = "meeting://"
	// DecisionURIScheme prefixes decision and rationale nodes' source_uri, followed by the meeting's URI, so
	// re-ingesting the transcript never deletes linkage results.
	DecisionURIScheme = "decision://"
	// PersonURIScheme prefixes a speaker/attendee identity, shared across
	// meetings so the same person is one node graph-wide.
	PersonURIScheme = "person://"
)

// DecisionURI returns the source_uri for the decisions and rationales linked from the meeting at meetingURI.
// Keeping meetingURI in it lets a source_prefix on the meeting still match them.
func DecisionURI(meetingURI string) string { return DecisionURIScheme + meetingURI }
```

Replace `Upsert` (`:141-152`):

```go
// Upsert replaces tr's prior version under its source_uri with its ToGraph subgraph and returns the node count,
// with noun naming it in errors. Edges other sources attached to surviving nodes (linkage, concepts) stay.
func Upsert(ctx context.Context, gs store.GraphStore, tr *Transcript, now time.Time, noun string) (int, error) {
	nodes, edges := ToGraph(tr, now)
	uri := nodes[0].SourceURI
	if err := rehomeDecisions(ctx, gs, uri); err != nil {
		return 0, fmt.Errorf("upsert %s nodes: %w", noun, err)
	}
	if _, err := store.ReplaceSources(ctx, gs, store.SourceUpdate{URI: uri, Nodes: nodes, Edges: edges}); err != nil {
		return 0, fmt.Errorf("upsert %s nodes: %w", noun, err)
	}
	return len(nodes), nil
}

// rehomeDecisions moves decision and rationale nodes still stored under meetingURI (written before
// DecisionURI existed) to DecisionURI(meetingURI), so replacing the meeting keeps them.
func rehomeDecisions(ctx context.Context, gs store.GraphStore, meetingURI string) error {
	legacy, err := store.NodesMatching(ctx, gs, store.NodeFilter{
		Kinds:     []model.NodeKind{model.KindDecision, model.KindRationale},
		SourceURI: meetingURI,
	})
	if err != nil || len(legacy) == 0 {
		return err
	}
	for i := range legacy {
		legacy[i].SourceURI = DecisionURI(meetingURI)
	}
	return gs.UpsertNodes(ctx, legacy)
}
```

In `internal/extract/meeting/linkage.go` `Submit`, after `meetingURI := meeting.SourceURI` (`:175`) add `decisionURI := DecisionURI(meetingURI)`. In the decision node literal (`:210`) and rationale node literal (`:231`), change `SourceURI: meetingURI,` to `SourceURI: decisionURI,`. Node IDs keep `model.NewNodeID(meetingURI, …)` so existing decisions keep their IDs. Replace `deleteSupersededDecisions` (`:373-400`):

```go
// deleteSupersededDecisions removes the meeting's decision and rationale nodes not in keep, under DecisionURI or
// (written before it existed) the meeting's own URI. Stores without NodeDeleter keep stale ones until a rebuild.
func (s *LinkService) deleteSupersededDecisions(ctx context.Context, meetingURI string, keep map[string]bool) error {
	var stale []string
	for _, uri := range []string{DecisionURI(meetingURI), meetingURI} {
		prior, err := store.NodesMatching(ctx, s.store, store.NodeFilter{
			Kinds:     []model.NodeKind{model.KindDecision, model.KindRationale},
			SourceURI: uri,
		})
		if err != nil {
			return fmt.Errorf("load nodes for supersede: %w", err)
		}
		for _, n := range prior {
			// The filter ignores an empty SourceURI, so the check stays for a meeting without one.
			if n.SourceURI == uri && !keep[n.ID] {
				stale = append(stale, n.ID)
			}
		}
	}
	if len(stale) == 0 {
		return nil
	}
	if d, ok := s.store.(store.NodeDeleter); ok {
		if _, err := d.DeleteByIDs(ctx, stale); err != nil {
			return fmt.Errorf("delete superseded decisions: %w", err)
		}
	}
	return nil
}
```

Callers drop their own `DeleteBySourceURI`, since `Upsert` now replaces:
- `cmd/chiton/meeting.go:47-54`:

```go
// ingestTranscript replaces any prior version of the transcript with its meeting subgraph, keeping linkage
// edges on surviving nodes. Returns the number of nodes written.
func ingestTranscript(ctx context.Context, gs store.GraphStore, tr *meeting.Transcript) (int, error) {
	return meeting.Upsert(ctx, gs, tr, time.Now(), "meeting")
}
```

- `internal/ingest/teams/source.go`: delete the `DeleteBySourceURI` block at `:104-106`, leaving `n, err := meeting.Upsert(ctx, i.store, tr, i.now(), "meeting")`.
- `internal/ingest/slack/slack.go`, replace `ingestOne` (`:106-130`):

```go
// ingestOne upserts one fetched conversation as a meeting subgraph.
func (i *Ingester) ingestOne(ctx context.Context, conv Conversation) (Report, error) {
	rep := Report{ChannelID: conv.ChannelID, ThreadTS: conv.ThreadTS, Messages: len(conv.Messages)}
	uri := ConversationURI(conv.ChannelID, conv.ThreadTS)

	tr := conversationToTranscript(conv, uri)
	if len(tr.Turns) == 0 {
		// Nothing ingestable (e.g. only join/leave notices), so clear any stale prior version.
		if _, err := i.store.DeleteBySourceURI(ctx, uri); err != nil {
			return rep, fmt.Errorf("replace prior conversation %s: %w", uri, err)
		}
		return rep, nil
	}
	// Upsert replaces the prior version by diff, so decisions linked to this conversation survive.
	n, err := meeting.Upsert(ctx, i.store, tr, i.now(), "conversation")
	if err != nil {
		return rep, err
	}
	rep.Turns = len(tr.Turns)
	rep.NodesUpserted = n
	return rep, nil
}
```

- `internal/ingest/url/ingest.go`: delete the replace block at `:107-110`, and replace the two upserts at `:185-192` with:

```go
	// Replace any prior version of this URL, keeping concept edges on sections that survive.
	if _, err := store.ReplaceSources(ctx, ing.store, store.SourceUpdate{URI: uri, Nodes: nodes, Edges: edges}); err != nil {
		return Report{}, fmt.Errorf("replace %s: %w", uri, err)
	}
```

- `internal/ingest/email/email.go`: replace `:130-143` (delete + two upserts) with:

```go
		nodes, edges, emailID, persons := buildGraph(m, uri, now)
		// Replace any prior version of this message; person nodes have their own person:// uri and survive.
		if _, err := store.ReplaceSources(ctx, i.store, store.SourceUpdate{URI: uri, Nodes: nodes, Edges: edges}); err != nil {
			return rep, fmt.Errorf("replace %s: %w", uri, err)
		}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/graph/... ./internal/ingest/... ./internal/extract/... ./internal/mcp/... ./cmd/chiton/... -count=1; go vet ./internal/graph/... ./internal/ingest/... ./internal/extract/... ./cmd/chiton/...`
Expected: PASS. The existing idempotent-replace tests (`url.TestAddIdempotentReplace`, `email.TestIngestIdempotentReplace`, `TestSubmitSupersedesPriorDecisions`, `TestSyncFileReconciles`) still pass unchanged.

- [ ] **Step 5: Docs**
- `README.md:325`: change "(stale nodes/edges replaced)" to "(stale nodes/edges replaced; edges meetings and concepts attached to surviving symbols are kept)".
- `docs/MCP.md:859-860` (`submit_meeting_links`): append "Decision and rationale nodes are stored under `decision://<meeting source_uri>`, so re-ingesting the transcript or Slack conversation keeps them and their edges."
- `skills/subagent-tasks/SKILL.md:56-59`: after "…`MENTIONED` edges", add "; decisions live under `decision://<meeting-uri>` and survive re-ingest and code edits". Then run `go test ./internal/install/... -count=1`, and if its golden differs, regenerate with `go test ./internal/install/ -run Golden -update`.
- `CHANGELOG.md` `### Fixed`: "**Re-syncing a file no longer deletes meeting and concept links.** File sync and meeting/Slack/Teams/URL/email re-ingest replace a source by diff. Nodes the new extraction drops are deleted with their edges, but edges other sources attached to surviving nodes (`DECIDED_IN`, `DISCUSSED_IN`, `MENTIONED`, concept `DESCRIBES`, finding `AFFECTS`) are kept. An unchanged node keeps its `extraction_status` and embedding; a changed one loses its embedding so `chiton embed` redoes it. Decision and rationale nodes now use `decision://<meeting-uri>`, so re-ingesting a Slack channel no longer deletes them; older ones move on the next re-ingest."

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/store internal/ingest internal/extract/meeting cmd/chiton/meeting.go README.md docs/MCP.md skills/subagent-tasks/SKILL.md CHANGELOG.md
git commit -m "fix(store): replace sources by diff so re-sync keeps other sources' edges"
```

### Task C2: `chiton build` runs the sync engine as a parallel, batched pipeline

**Finding:** `cmd/chiton/main.go:463-511` re-implements sync badly. It never calls `DeleteBySourceURI`, so a renamed symbol (`Hello`→`Greet`) keeps `Hello` forever. It never prunes deleted files and never returns `SkipDir`, so it walks `node_modules`/`vendor`/`.git`. It is serial: per file it reads, extracts, then runs `UpsertNodes` and `UpsertEdges`, and each is its own WriteBatch flush (`internal/graph/store/badger/store.go:279-325`). The audit measured `GOROOT/src` (7,826 files) at 5.9 s warm and 61 s cold, against 3.3 s for `chiton bench`'s extract-only pass, so at least 45% of a cold build is store writes. (Not re-measured here, since this pass was read-only.) One audit claim is wrong: build does check `ctx` per entry (`:466-468`).
**Priority:** P1   **Effort:** L   **Depends on:** C1, C6, C3
**Behaviour change:**
- `build` now removes symbols deleted or renamed since the last build.
- It prunes indexed files gone from disk (new `--prune`, default true, same as `sync once`).
- It skips C1's ignored directories. Previously indexed `vendor/` nodes stay until the files are deleted.
- It extracts on `GOMAXPROCS` workers and writes in batches of about 20k records.
- Its summary line is unchanged, but the node/edge counts now count written records.
- Per-file skip messages on stderr now start with `sync: skip`, and a cache-save failure now fails `sync once` too.
- DECISION NEEDED: `build` skips the incremental re-cluster (new `Config.NoRecluster`; recommended, keeps build's timing and output). The alternative is to re-cluster like `sync once` whenever `cluster.json` exists.

**Files:**
- Modify: `internal/ingest/engine.go:46-55` (`Config`), `BatchSummary`, `SyncFile`, `SyncDir`, `tally`, `flush`; add `prepared`, `prepare`, `apply`, `finish`, `writeAll`, `prune`, `skipped`, `logSkip`
- Modify: `cmd/chiton/main.go:3-41` (imports), `:433-527` (`buildCmd`)
- Modify: `internal/ingest/engine_test.go:283-284` (stale comment)
- Create: `cmd/chiton/build_test.go`
- Test: `internal/ingest/engine_test.go`

**Interfaces:**
- Consumes: `walk.LoadPolicy`, `walk.Files`, `walk.Stats` (C1); `store.SourceUpdate`, `store.ReplaceSources` (C3); `store.GetNodes` (existing)
- Produces:
  - `ingest.Config.NoRecluster bool`
  - `ingest.BatchSummary{Scanned, Upserted, Deleted, Skipped, Unchanged, Unreadable, Nodes, Edges, Reembedded int; Reclustered bool}`

- [ ] **Step 1: Write the failing test**

Append to `internal/ingest/engine_test.go` (add `"fmt"` to its imports; `store` and `badgerstore` are already imported):

```go
// countingStore records every ReplaceSources call and each node it writes.
type countingStore struct {
	*badgerstore.Store
	calls      int
	nodeWrites map[string]int
}

func (c *countingStore) ReplaceSources(ctx context.Context, updates []store.SourceUpdate) ([]int, error) {
	c.calls++
	for _, u := range updates {
		for _, n := range u.Nodes {
			c.nodeWrites[n.ID]++
		}
	}
	return c.Store.ReplaceSources(ctx, updates)
}

// TestSyncDirBatchesWrites checks SyncDir writes many extracted files in one batched call, reports scan and
// unchanged counts, and skips every file on a re-run.
func TestSyncDirBatchesWrites(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	cs := &countingStore{Store: h.gs, nodeWrites: map[string]int{}}
	eng := New(cs, h.cache, Config{GraphPath: h.eng.cfg.GraphPath}, nil, nil, nil)
	const files = 30
	for i := range files {
		h.write(t, fmt.Sprintf("f%02d.go", i), fmt.Sprintf("package src\nfunc F%02d() {}\n", i))
	}

	sum, err := eng.SyncDir(ctx, h.dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if sum.Scanned != files || sum.Upserted != files || sum.Nodes != 2*files {
		t.Fatalf("summary %+v, want %d scanned and upserted, %d nodes", sum, files, 2*files)
	}
	if cs.calls != 1 {
		t.Errorf("ReplaceSources calls = %d, want 1 batched write", cs.calls)
	}

	sum, err = eng.SyncDir(ctx, h.dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if sum.Unchanged != files || sum.Upserted != 0 {
		t.Errorf("re-run summary %+v, want all %d unchanged", sum, files)
	}
}
```

Create `cmd/chiton/build_test.go`:

```go
package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/config"
)

// TestBuildReplacesStaleSymbols checks a rebuild drops renamed symbols and deleted files, skips ignored
// directories, and keeps the summary line's shape.
func TestBuildReplacesStaleSymbols(t *testing.T) {
	proj := t.TempDir()
	t.Setenv(config.ProjectDirEnv, proj)
	src := filepath.Join(proj, "src")
	write := func(rel, body string) {
		t.Helper()
		p := filepath.Join(src, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	build := func() string {
		t.Helper()
		var out bytes.Buffer
		cmd := rootCmd()
		cmd.SetOut(&out)
		cmd.SetErr(io.Discard)
		cmd.SetArgs([]string{"build", src})
		if err := cmd.ExecuteContext(context.Background()); err != nil {
			t.Fatalf("build: %v", err)
		}
		return out.String()
	}

	write("a.go", "package src\nfunc Hello() {}\n")
	write("gone.go", "package src\nfunc Gone() {}\n")
	write(filepath.Join("node_modules", "dep", "x.js"), "function dep() {}\n")
	build()

	write("a.go", "package src\nfunc Greet() {}\n")
	if err := os.Remove(filepath.Join(src, "gone.go")); err != nil {
		t.Fatal(err)
	}
	if out := build(); !strings.Contains(out, "build: done - 1 files scanned, 0 skipped (unchanged)") {
		t.Errorf("second build output = %q, want 1 file scanned", out)
	}

	cfg, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	gs, err := openGraphStore(cfg.GraphPath, true)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	nodes, err := gs.AllNodes(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	labels := map[string]bool{}
	for _, n := range nodes {
		labels[n.Label] = true
	}
	for _, stale := range []string{"Hello", "Gone", "dep"} {
		if labels[stale] {
			t.Errorf("stale or ignored symbol %q is still in the graph", stale)
		}
	}
	if !labels["Greet"] {
		t.Error("Greet missing after rebuild")
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/ingest/ ./cmd/chiton/ -run 'TestSyncDirBatchesWrites|TestBuildReplacesStaleSymbols' -count=1`
Expected: FAIL. The ingest build fails with `sum.Scanned undefined (type *BatchSummary has no field or method Scanned)`. `TestBuildReplacesStaleSymbols` fails with `second build output = "…2 files scanned…"` (it walks `node_modules`) and `stale or ignored symbol "Hello" is still in the graph`.

- [ ] **Step 3: Implement**

In `internal/ingest/engine.go`, add imports `"runtime"` and `"github.com/Eric-Fernald/chiton/internal/walk"` (already added in C1). Add to `Config`:

```go
	NoRecluster     bool          // skip the post-batch re-cluster (build leaves clustering to 'chiton cluster')
```

Replace `BatchSummary`:

```go
// BatchSummary aggregates a SyncChanges / SyncDir run.
type BatchSummary struct {
	Scanned     int // supported files the walk found (SyncDir only)
	Upserted    int
	Deleted     int
	Skipped     int // unchanged plus unreadable or unparsable files
	Unchanged   int // files the content cache skipped
	Unreadable  int // walk entries that could not be read, logged and skipped (SyncDir only)
	Nodes       int // nodes written
	Edges       int // edges written
	Reembedded  int
	Reclustered bool
}
```

Add the pipeline pieces and replace `SyncFile`, `SyncDir`, `tally` and `flush`:

```go
const (
	syncQueue       = 256    // buffered paths and results between SyncDir's walker, pool, and writer
	batchRecords    = 20_000 // nodes plus edges per batched ReplaceSources call
	reasonUnchanged = "unchanged"
)

// prepared is one file read, hashed, and extracted off the writer goroutine.
type prepared struct {
	path, uri, sha string
	content        []byte // kept only when re-analyze needs it
	nodes          []model.Node
	edges          []model.Edge
	gone           bool    // the file vanished, so the writer deletes its URI
	out            Outcome // Skipped when there is nothing to write
}

func skipped(path, reason string) Outcome {
	return Outcome{Path: path, Op: OpUpsert, Skipped: true, Reason: reason}
}

// prepare reads, hashes, and extracts path without touching the store, so a pool can run it concurrently.
// Read and parse failures come back as skips.
func (e *Engine) prepare(ctx context.Context, path string) prepared {
	p := prepared{path: path, uri: model.FileURI(path)}
	ex := extast.ForFile(path)
	if ex == nil {
		p.out = skipped(path, "unsupported file type")
		return p
	}
	content, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			p.gone = true
		} else {
			p.out = skipped(path, "read: "+err.Error())
		}
		return p
	}
	p.sha = model.ContentSHA(content)
	if !e.cache.Changed(p.uri, p.sha) {
		p.out = skipped(path, reasonUnchanged)
		return p
	}
	// Extract the same bytes that were hashed, so the cached hash always matches the extracted content.
	if p.nodes, p.edges, err = ex.ExtractSource(ctx, path, content); err != nil {
		p.nodes, p.edges = nil, nil
		p.out = skipped(path, "extract: "+err.Error())
		return p
	}
	if e.cfg.Reanalyze && e.analyzer != nil {
		p.content = content
	}
	return p
}

// apply writes a batch of extracted files in one ReplaceSources call, then marks the cache and runs each file's
// follow-ups. Every extraction includes its file node, so a file whose node is absent needs no prior lookup.
func (e *Engine) apply(ctx context.Context, batch []prepared) ([]Outcome, error) {
	fileIDs := make([]string, len(batch))
	for i, p := range batch {
		fileIDs[i] = model.NewNodeID(p.uri, p.uri)
	}
	existing, err := store.GetNodes(ctx, e.store, fileIDs)
	if err != nil {
		return nil, fmt.Errorf("look up file nodes: %w", err)
	}
	indexed := make(map[string]bool, len(existing))
	for _, n := range existing {
		indexed[n.ID] = true
	}
	updates := make([]store.SourceUpdate, len(batch))
	for i, p := range batch {
		updates[i] = store.SourceUpdate{URI: p.uri, Nodes: p.nodes, Edges: p.edges, Fresh: !indexed[fileIDs[i]]}
	}
	deleted, err := store.ReplaceSources(ctx, e.store, updates...)
	if err != nil {
		return nil, fmt.Errorf("replace %d file(s): %w", len(batch), err)
	}
	outs := make([]Outcome, len(batch))
	for i, p := range batch {
		e.cache.Mark(p.uri, p.sha)
		outs[i] = e.finish(ctx, p, deleted[i])
	}
	return outs, nil
}

// finish re-embeds and re-analyzes a written file when configured and reports its outcome.
func (e *Engine) finish(ctx context.Context, p prepared, deleted int) Outcome {
	ids := make([]string, 0, len(p.nodes))
	for _, n := range p.nodes {
		ids = append(ids, n.ID)
	}
	out := Outcome{Path: p.path, Op: OpUpsert, NodesUpserted: len(p.nodes), EdgesUpserted: len(p.edges),
		NodesDeleted: deleted, TouchedIDs: ids}
	if e.cfg.Reembed && e.embedder != nil {
		n, err := e.reembed(ctx, p.nodes)
		if err != nil {
			e.log("sync: %s: re-embed failed: %v", p.path, err)
		}
		out.Reembedded = n
	}
	if e.cfg.Reanalyze && e.analyzer != nil {
		if _, err := e.analyzer.AnalyzeExtracted(ctx, p.path, p.content, p.nodes, p.edges); err != nil {
			e.log("sync: %s: re-analyze failed: %v", p.path, err)
		}
	}
	e.markTouched(ids)
	return out
}

// SyncFile (re-)extracts a single file into the graph and hands a vanished file to DeleteFile.
// Read and parse failures are skips, not errors, so a watcher never stops on one bad file.
func (e *Engine) SyncFile(ctx context.Context, path string) (Outcome, error) {
	p := e.prepare(ctx, path)
	switch {
	case p.gone:
		return e.deleteURI(ctx, p.uri)
	case p.out.Skipped:
		return p.out, nil
	}
	outs, err := e.apply(ctx, []prepared{p})
	if err != nil {
		return Outcome{}, err
	}
	return outs[0], nil
}

// SyncDir reconciles a directory tree: GOMAXPROCS workers read and extract each supported file the walk policy
// keeps, this goroutine writes results in batches, and prune deletes indexed files gone from disk.
func (e *Engine) SyncDir(ctx context.Context, root string, prune bool) (sum *BatchSummary, err error) {
	sum = &BatchSummary{}
	defer func() {
		if serr := e.cache.Save(); serr != nil && err == nil {
			err = fmt.Errorf("save cache: %w", serr)
		}
	}()
	policy, err := walk.LoadPolicy(root)
	if err != nil {
		return sum, err
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Only the walker writes seen, scanned, stats, and walkErr; closing paths and then results orders every
	// write before the reads below.
	seen := map[string]bool{}
	var scanned int
	var stats walk.Stats
	var walkErr error
	paths := make(chan string, syncQueue)
	go func() {
		defer close(paths)
		stats, walkErr = walk.Files(ctx, root, policy, func(path string) error {
			if extast.ForFile(path) == nil {
				return nil
			}
			seen[model.FileURI(path)] = true
			scanned++
			select {
			case paths <- path:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}, func(path string, err error) { e.log("sync: skip %s: %v", path, err) })
	}()

	results := make(chan prepared, syncQueue)
	var wg sync.WaitGroup
	for range runtime.GOMAXPROCS(0) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range paths {
				if ctx.Err() == nil {
					results <- e.prepare(ctx, path)
				}
			}
		}()
	}
	go func() {
		wg.Wait()
		close(results)
	}()

	if err := e.writeAll(ctx, results, sum); err != nil {
		cancel()
		for range results {
		}
		return sum, err
	}
	if walkErr != nil {
		return sum, walkErr
	}
	sum.Scanned, sum.Unreadable = scanned, stats.Unreadable
	if prune {
		if err := e.prune(ctx, root, seen, sum); err != nil {
			return sum, err
		}
	}
	e.flush(ctx, sum)
	return sum, nil
}

// writeAll applies prepared files in batches of about batchRecords nodes and edges, deleting vanished files and
// tallying skips as they arrive.
func (e *Engine) writeAll(ctx context.Context, results <-chan prepared, sum *BatchSummary) error {
	var batch []prepared
	records := 0
	writeBatch := func() error {
		if len(batch) == 0 {
			return nil
		}
		outs, err := e.apply(ctx, batch)
		if err != nil {
			return err
		}
		for _, out := range outs {
			e.tally(sum, out)
		}
		batch, records = nil, 0
		return nil
	}
	for p := range results {
		switch {
		case p.gone:
			out, err := e.deleteURI(ctx, p.uri)
			if err != nil {
				return err
			}
			e.tally(sum, out)
		case p.out.Skipped:
			e.logSkip(p.out)
			e.tally(sum, p.out)
		default:
			batch = append(batch, p)
			if records += len(p.nodes) + len(p.edges); records >= batchRecords {
				if err := writeBatch(); err != nil {
					return err
				}
			}
		}
	}
	return writeBatch()
}

// logSkip reports a file skipped for any reason but an unchanged hash.
func (e *Engine) logSkip(out Outcome) {
	if out.Reason != reasonUnchanged {
		e.log("sync: skip %s: %s", out.Path, out.Reason)
	}
}

// prune deletes indexed files under root that the walk did not see and that are gone from disk. A file the walk
// excluded (ignored dir, unsupported type, narrower root) but still present survives.
func (e *Engine) prune(ctx context.Context, root string, seen map[string]bool, sum *BatchSummary) error {
	rootURI := model.FileURI(root)
	for _, uri := range e.cache.URIs() {
		if err := ctx.Err(); err != nil {
			return err
		}
		if (uri != rootURI && !strings.HasPrefix(uri, rootURI+"/")) || seen[uri] {
			continue
		}
		if p, ok := model.PathFromFileURI(uri); ok {
			if _, err := os.Stat(p); err == nil {
				continue
			}
		}
		out, err := e.deleteURI(ctx, uri)
		if err != nil {
			return err
		}
		e.tally(sum, out)
	}
	return nil
}

// flush re-clusters after a batch unless NoRecluster. The cache is persisted separately by the caller's
// deferred save, so it is saved even when a change errors.
func (e *Engine) flush(ctx context.Context, sum *BatchSummary) {
	if e.cfg.NoRecluster {
		return
	}
	haveSidecar, err := e.recluster(ctx)
	if err != nil {
		e.log("sync: re-cluster failed: %v", err)
		return
	}
	sum.Reclustered = haveSidecar
}

func (e *Engine) tally(sum *BatchSummary, out Outcome) {
	switch {
	case out.Skipped:
		sum.Skipped++
		if out.Reason == reasonUnchanged {
			sum.Unchanged++
		}
	case out.Op == OpDelete:
		sum.Deleted++
	default:
		sum.Upserted++
		sum.Nodes += out.NodesUpserted
		sum.Edges += out.EdgesUpserted
	}
	sum.Reembedded += out.Reembedded
}
```

In `cmd/chiton/main.go`, drop the `"io/fs"` and `extast` imports, add `"github.com/Eric-Fernald/chiton/internal/ingest"`, and replace `buildCmd` (`:433-527`):

```go
func buildCmd() *cobra.Command {
	var prune bool
	cmd := &cobra.Command{
		Use:   "build <path>",
		Short: "Walk a directory and extract its graph",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}

			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "build: scanning %s\n", args[0])
			fmt.Fprintf(out, "build: graph -> %s\n", cfg.GraphPath)

			gs, err := openGraphStore(cfg.GraphPath, false)
			if err != nil {
				return err
			}
			defer gs.Close()

			ch, err := loadBuildCache(cfg.GraphPath)
			if err != nil {
				return err
			}

			// Build is SyncDir without re-embed, re-analyze, or re-cluster: the same walk policy, diff-based
			// replacement, prune, and batched writes as `sync once`.
			start := time.Now()
			eng := ingest.New(gs, ch, ingest.Config{GraphPath: cfg.GraphPath, NoRecluster: true}, nil, nil, errLogf(cmd))
			sum, err := eng.SyncDir(cmd.Context(), args[0], prune)
			if err != nil {
				return err
			}
			fmt.Fprintf(out, "build: done - %d files scanned, %d skipped (unchanged), %d nodes, %d edges [%s]\n",
				sum.Scanned, sum.Unchanged, sum.Nodes, sum.Edges, time.Since(start).Round(time.Millisecond))
			return nil
		},
	}
	cmd.Flags().BoolVar(&prune, "prune", true, "Delete nodes for files under the tree that no longer exist")
	return cmd
}
```

In `internal/ingest/engine_test.go:283-284`, change the comment to "Index a file under a dir SyncDir's walk ignores (vendor), simulating a store written before the ignore policy existed."

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/ingest/... ./cmd/chiton/... -count=1 -race; go vet ./internal/ingest/... ./cmd/chiton/...`
Expected: PASS, with no race reports. (`-race` needs CGO; if it is unavailable on Windows, run without it here and rely on CI's Linux/macOS job.)

Then benchmark against the audit baseline from a scratch project (not the repo's store):

```powershell
$p = Join-Path $env:TEMP "chiton-build-bench"; New-Item -ItemType Directory -Force $p | Out-Null
go build -o (Join-Path $p "chiton.exe") ./cmd/chiton
Push-Location $p; .\chiton.exe build (Join-Path (go env GOROOT) "src"); .\chiton.exe build (Join-Path (go env GOROOT) "src"); Pop-Location
```

Expected: the second run reports `N files scanned, N skipped (unchanged), 0 nodes, 0 edges`. Record both timings in the PR against the audit's 61 s cold / 5.9 s warm.

- [ ] **Step 5: Docs**
- `README.md:192` (`chiton build` row): add "Removes symbols and files deleted since the last build (`--prune`, default true) and extracts in parallel."
- `skills/setup/SKILL.md:104-105`: change to "`chiton build` is incremental — a SHA-256 content cache skips unchanged files, and a rebuild removes symbols and files deleted since the last one, so re-running it is cheap."
- `CHANGELOG.md` `### Fixed`: "**`chiton build` runs the sync engine.** It removes symbols renamed or deleted since the last build, prunes files gone from disk (`--prune`, default true), skips ignored directories and `.chitonignore` paths, extracts on `GOMAXPROCS` workers, and batches store writes. The summary line is unchanged; per-file skip messages on stderr now start with `sync: skip`."

- [ ] **Step 6: Commit**

```powershell
git add internal/ingest/engine.go internal/ingest/engine_test.go cmd/chiton/main.go cmd/chiton/build_test.go README.md skills/setup/SKILL.md CHANGELOG.md
git commit -m "fix(build): delegate chiton build to SyncDir with a parallel, batched pipeline"
```

### Task C8: Never read a symlink when indexing

**Finding:** Indexing follows symlinks out of the repo. `cmd/chiton/main.go:475-481` checks only `d.IsDir()` and `os.ReadFile`s a symlinked file. `internal/ingest/engine.go:137` (`SyncFile`, reached from the watcher, `sync git` and `SyncDir`) does the same, and so does `internal/analyze/llm.go:188` (Section D). A Markdown file with no headings stores its whole body in `attrs["text"]` (`internal/extract/ast/markdown.go:59-66`), so a repo containing `notes.md -> ~/.ssh/id_rsa` puts the key in the graph and in `pending_extractions` sent to a cloud LLM. C1's `walk.Files` already yields regular files only, and C2 moved build onto it. This task closes the direct-read path: a watcher event or git change naming a symlink still reaches `prepare`.
**Priority:** P2   **Effort:** S   **Depends on:** C1, C2
**Behaviour change:** Symlinks, junctions and other non-regular files are never indexed, even when a watcher event or `sync git` names one. DECISION NEEDED: skip every symlink (recommended; an in-repo target is indexed at its real path anyway). The alternative is to follow symlinks whose resolved target is inside the root.

**Files:**
- Modify: `internal/ingest/engine.go` (`prepare`, from C2)
- Modify: `docs/SECURITY.md:486-501` (threat table)
- Test: `internal/ingest/engine_test.go`

**Interfaces:**
- Consumes: `walk.Files` (C1), `Engine.prepare` (C2)
- Produces: none (Section D gets the same guarantee by using `walk.Files` in `internal/analyze`)

- [ ] **Step 1: Write the failing test**

Append to `internal/ingest/engine_test.go`:

```go
// TestSyncSkipsSymlinks checks neither a direct sync nor a reconcile reads a symlink, so a link to a file outside
// the repo (a key, a .env) never reaches the graph.
func TestSyncSkipsSymlinks(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	secret := filepath.Join(t.TempDir(), "id_rsa.md")
	if err := os.WriteFile(secret, []byte("PRIVATE KEY MATERIAL\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(h.dir, "notes.md")
	if err := os.Symlink(secret, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	out, err := h.eng.SyncFile(ctx, link)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Skipped || out.Reason != "not a regular file" {
		t.Errorf("SyncFile(symlink) = %+v, want skipped as not a regular file", out)
	}
	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	if n := len(h.nodesForURI(t, model.FileURI(link))); n != 0 {
		t.Errorf("symlinked file indexed with %d node(s)", n)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/ingest/ -run TestSyncSkipsSymlinks -count=1`
Expected: FAIL on Linux/macOS with `SyncFile(symlink) = {… Skipped:false …}, want skipped as not a regular file`. On Windows without Developer Mode it SKIPs (`symlinks unavailable`), so confirm the failure in CI or WSL.

- [ ] **Step 3: Implement**

Replace `prepare` in `internal/ingest/engine.go`:

```go
// prepare reads, hashes, and extracts path without touching the store, so a pool can run it concurrently.
// Read and parse failures come back as skips.
func (e *Engine) prepare(ctx context.Context, path string) prepared {
	p := prepared{path: path, uri: model.FileURI(path)}
	ex := extast.ForFile(path)
	if ex == nil {
		p.out = skipped(path, "unsupported file type")
		return p
	}
	fi, err := os.Lstat(path)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		p.gone = true
		return p
	case err != nil:
		p.out = skipped(path, "stat: "+err.Error())
		return p
	case !fi.Mode().IsRegular():
		// Never follow a symlink or junction: its target may be outside the repo (a key, a .env).
		p.out = skipped(path, "not a regular file")
		return p
	}
	content, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			p.gone = true
		} else {
			p.out = skipped(path, "read: "+err.Error())
		}
		return p
	}
	p.sha = model.ContentSHA(content)
	if !e.cache.Changed(p.uri, p.sha) {
		p.out = skipped(path, reasonUnchanged)
		return p
	}
	// Extract the same bytes that were hashed, so the cached hash always matches the extracted content.
	if p.nodes, p.edges, err = ex.ExtractSource(ctx, path, content); err != nil {
		p.nodes, p.edges = nil, nil
		p.out = skipped(path, "extract: "+err.Error())
		return p
	}
	if e.cfg.Reanalyze && e.analyzer != nil {
		p.content = content
	}
	return p
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/ingest/... -count=1; go vet ./internal/ingest/...`
Expected: PASS (the symlink tests skip on Windows without Developer Mode).

- [ ] **Step 5: Docs**
- `docs/SECURITY.md` §6 "In scope" table (`:486-501`): add a row: "| A symlink in an indexed repo pointing outside it (e.g. `notes.md -> ~/.ssh/id_rsa`) pulling its target into the graph and `pending_extractions` | `build`, `sync`, and `sync watch` index regular files only; symlinks and junctions are skipped at the walk and at the read | **Shipped** (`analyze` via Section D) |".
- `CHANGELOG.md` `### Security`: "**Indexing no longer follows symlinks.** `build`, `sync`, and `sync watch` index regular files only; a symlink or junction (e.g. `notes.md -> ~/.ssh/id_rsa`) is skipped, so its target never lands in the graph or in `pending_extractions`."

- [ ] **Step 6: Commit**

```powershell
git add internal/ingest/engine.go internal/ingest/engine_test.go docs/SECURITY.md CHANGELOG.md
git commit -m "fix(ingest): never read symlinks or junctions when syncing files"
```

### Task C7: Stop writing duplicate nodes and edges

**Finding:** The Go extractor appends a `pkg://` stub node and a `CALLS` edge per call site, and an import node and edge per import spec, with no dedupe (`internal/extract/ast/golang.go:48-53`, `:139-160`). The regex extractors dedupe through `fileGraph` (`internal/extract/ast/extractor.go:85-141`). Across files, every importing file rewrites the shared `pkg://` stubs (`importNode`, `extractor.go:329-340`). The audit measured 241,904 nodes / 287,648 edges emitted for 118,240 / 230,497 unique ones on `GOROOT/src`, i.e. 51% / 20% redundant writes.
**Priority:** P2   **Effort:** S   **Depends on:** C2
**Behaviour change:** Go extraction yields each node/edge once per file. For a repeated `func init()` the first declaration's span wins, where the store previously kept the last. A `build`/`sync once` run writes each shared stub once, so `build`'s node/edge totals drop.

**Files:**
- Modify: `internal/extract/ast/golang.go:21-167` (`ExtractSource`)
- Modify: `internal/ingest/engine.go` (`apply`, `writeAll`, `SyncFile` from C2; add `dropWritten`)
- Test: `internal/extract/ast/golang_test.go`, `internal/ingest/engine_test.go`

**Interfaces:**
- Consumes: `fileGraph`/`newFileGraph` (existing, `extractor.go:85-141`); `countingStore` test helper (C2)
- Produces: none

- [ ] **Step 1: Write the failing test**

Append to `internal/extract/ast/golang_test.go`:

```go
// TestGoExtractorDedupes checks repeated calls, a doubly imported package, and duplicate init funcs yield each
// node and edge once.
func TestGoExtractorDedupes(t *testing.T) {
	src := `package p

import (
	"fmt"
	f "fmt"
)

func a() { b(); b(); fmt.Println(); f.Println() }

func b() {}

func init() {}

func init() {}
`
	nodes, edges, err := (&GoExtractor{}).ExtractFile(context.Background(), writeTempFile(t, "p.go", src))
	if err != nil {
		t.Fatalf("ExtractFile: %v", err)
	}
	seen := map[string]bool{}
	for _, n := range nodes {
		if seen[n.ID] {
			t.Errorf("duplicate node %q", n.Label)
		}
		seen[n.ID] = true
	}
	seen = map[string]bool{}
	for _, e := range edges {
		if seen[e.ID] {
			t.Errorf("duplicate %s edge %s -> %s", e.Kind, e.Src, e.Dst)
		}
		seen[e.ID] = true
	}
}
```

Append to `internal/ingest/engine_test.go`:

```go
// TestSyncDirWritesSharedStubsOnce checks a package stub every file references is written once per run.
func TestSyncDirWritesSharedStubsOnce(t *testing.T) {
	h := newHarness(t, false, false)
	cs := &countingStore{Store: h.gs, nodeWrites: map[string]int{}}
	eng := New(cs, h.cache, Config{GraphPath: h.eng.cfg.GraphPath}, nil, nil, nil)
	for _, name := range []string{"A", "B", "C"} {
		h.write(t, name+".go", "package src\nimport \"fmt\"\nfunc "+name+"() { fmt.Println() }\n")
	}
	if _, err := eng.SyncDir(context.Background(), h.dir, true); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{model.NewNodeID("pkg://fmt", "fmt"), model.NewNodeID("pkg://fmt", "Println")} {
		if got := cs.nodeWrites[id]; got != 1 {
			t.Errorf("stub %s written %d times, want 1", id, got)
		}
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/extract/ast/ ./internal/ingest/ -run 'TestGoExtractorDedupes|TestSyncDirWritesSharedStubsOnce' -count=1`
Expected: FAIL with `duplicate node "fmt"`, `duplicate node "init"`, `duplicate IMPORTS edge …`, `duplicate CALLS edge …`, and `stub … written 3 times, want 1`.

- [ ] **Step 3: Implement**

Replace `ExtractSource` in `internal/extract/ast/golang.go`:

```go
// ExtractSource parses a single .go file into file, top-level func and type, and imported-package nodes.
// CALLS covers direct intra-file calls and pkg.Func selector calls. Receiver method calls need go/types.
func (g *GoExtractor) ExtractSource(_ context.Context, path string, src []byte) ([]model.Node, []model.Edge, error) {
	fset := token.NewFileSet()
	// Skip object resolution, since the extractor reads only identifier names, never ast.Object links.
	f, err := parser.ParseFile(fset, path, src, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		return nil, nil, fmt.Errorf("parse %s: %w", path, err)
	}

	// fg drops repeat nodes and edges (a call site per call, a package imported twice, repeated init funcs).
	fg := newFileGraph(newFileNode(path, src, map[string]any{"language": "go", "package": f.Name.Name}))

	// symbols maps plain func and type names to node IDs for intra-file CALLS resolution. It omits methods
	// because a bare call can't name one and same-named methods on different types would overwrite each other.
	symbols := map[string]string{}
	type funcBody struct {
		id   string
		body *goast.BlockStmt
	}
	var bodies []funcBody

	// Imports become package nodes and IMPORTS edges. Aliases map to package paths for selector calls.
	importAliases := make(map[string]string, len(f.Imports))
	for _, imp := range f.Imports {
		pkgPath := strings.Trim(imp.Path.Value, `"`)
		fg.addImport(pkgPath)

		alias := pkgPath[strings.LastIndexByte(pkgPath, '/')+1:]
		if imp.Name != nil && imp.Name.Name != "_" && imp.Name.Name != "." {
			alias = imp.Name.Name
		}
		importAliases[alias] = pkgPath
	}

	// Top-level declarations → symbol nodes + CONTAINS edges.
	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *goast.FuncDecl:
			label := funcLabel(d)
			pos := fset.Position(d.Pos())
			end := fset.Position(d.End())

			attrs := map[string]any{"lang_kind": "func"}
			if d.Doc != nil {
				if text := strings.TrimSpace(d.Doc.Text()); text != "" {
					attrs["docstring"] = text
				}
			}

			sym := model.Node{
				ID:         model.NewNodeID(fg.uri, label),
				Kind:       model.KindSymbol,
				Label:      label,
				SourceURI:  fg.uri,
				Span:       &model.SourceSpan{StartLine: pos.Line, EndLine: end.Line},
				Provenance: model.ProvExtracted,
				Confidence: 1.0,
				Attrs:      attrs,
			}
			fg.addNode(sym)
			if d.Recv == nil {
				symbols[d.Name.Name] = sym.ID
			}
			if d.Body != nil {
				bodies = append(bodies, funcBody{id: sym.ID, body: d.Body})
			}
			fg.addEdge(containsEdge(fg.fileID, sym.ID))

		case *goast.GenDecl:
			for _, spec := range d.Specs {
				ts, ok := spec.(*goast.TypeSpec)
				if !ok {
					continue
				}
				pos := fset.Position(ts.Pos())
				attrs := map[string]any{"lang_kind": "type"}
				if d.Doc != nil {
					if text := strings.TrimSpace(d.Doc.Text()); text != "" {
						attrs["docstring"] = text
					}
				}
				sym := model.Node{
					ID:         model.NewNodeID(fg.uri, ts.Name.Name),
					Kind:       model.KindSymbol,
					Label:      ts.Name.Name,
					SourceURI:  fg.uri,
					Span:       &model.SourceSpan{StartLine: pos.Line},
					Provenance: model.ProvExtracted,
					Confidence: 1.0,
					Attrs:      attrs,
				}
				fg.addNode(sym)
				symbols[ts.Name.Name] = sym.ID
				fg.addEdge(containsEdge(fg.fileID, sym.ID))
			}
		}
	}

	// Add CALLS edges for direct calls to this file's symbols and pkg.Func calls where pkg is an import alias.
	for _, fb := range bodies {
		callerID := fb.id
		goast.Inspect(fb.body, func(n goast.Node) bool {
			call, ok := n.(*goast.CallExpr)
			if !ok {
				return true
			}
			switch fun := call.Fun.(type) {
			case *goast.Ident:
				calleeID, ok := symbols[fun.Name]
				if !ok || calleeID == callerID {
					return true
				}
				fg.addEdge(callsEdge(callerID, calleeID))
			case *goast.SelectorExpr:
				xIdent, ok := fun.X.(*goast.Ident)
				if !ok {
					return true
				}
				pkgPath, ok := importAliases[xIdent.Name]
				if !ok {
					return true
				}
				pkgURI := "pkg://" + pkgPath
				calleeNode := model.Node{
					ID:         model.NewNodeID(pkgURI, fun.Sel.Name),
					Kind:       model.KindSymbol,
					Label:      xIdent.Name + "." + fun.Sel.Name,
					SourceURI:  pkgURI,
					Provenance: model.ProvExtracted,
					Confidence: 1.0,
					Attrs:      map[string]any{"external": true},
				}
				fg.addNode(calleeNode)
				fg.addEdge(callsEdge(callerID, calleeNode.ID))
			}
			return true
		})
	}

	return fg.nodes, fg.edges, nil
}
```

In `internal/ingest/engine.go`, give `apply` a run-level set of already-written stubs and add `dropWritten`:

```go
// apply writes a batch of extracted files in one ReplaceSources call, then marks the cache and runs each file's
// follow-ups. shared, when non-nil, holds other sources' nodes (pkg:// stubs) already written this run.
func (e *Engine) apply(ctx context.Context, batch []prepared, shared map[string]bool) ([]Outcome, error) {
	if shared != nil {
		for i := range batch {
			batch[i].nodes = dropWritten(batch[i].nodes, batch[i].uri, shared)
		}
	}
	fileIDs := make([]string, len(batch))
	for i, p := range batch {
		fileIDs[i] = model.NewNodeID(p.uri, p.uri)
	}
	// Every extraction includes its file node, so a file whose node is absent needs no prior-node lookup.
	existing, err := store.GetNodes(ctx, e.store, fileIDs)
	if err != nil {
		return nil, fmt.Errorf("look up file nodes: %w", err)
	}
	indexed := make(map[string]bool, len(existing))
	for _, n := range existing {
		indexed[n.ID] = true
	}
	updates := make([]store.SourceUpdate, len(batch))
	for i, p := range batch {
		updates[i] = store.SourceUpdate{URI: p.uri, Nodes: p.nodes, Edges: p.edges, Fresh: !indexed[fileIDs[i]]}
	}
	deleted, err := store.ReplaceSources(ctx, e.store, updates...)
	if err != nil {
		return nil, fmt.Errorf("replace %d file(s): %w", len(batch), err)
	}
	outs := make([]Outcome, len(batch))
	for i, p := range batch {
		e.cache.Mark(p.uri, p.sha)
		outs[i] = e.finish(ctx, p, deleted[i])
	}
	return outs, nil
}

// dropWritten removes nodes of other sources (pkg:// stubs) already in written and records the rest. Stubs
// carry no per-file state, so rewriting one for every importing file is pure overhead.
func dropWritten(nodes []model.Node, uri string, written map[string]bool) []model.Node {
	out := nodes[:0]
	for _, n := range nodes {
		if n.SourceURI != uri {
			if written[n.ID] {
				continue
			}
			written[n.ID] = true
		}
		out = append(out, n)
	}
	return out
}
```

In `SyncFile`, call `e.apply(ctx, []prepared{p}, nil)`. In `writeAll`, declare `shared := map[string]bool{}` next to `var batch []prepared` and call `e.apply(ctx, batch, shared)` inside `writeBatch`.

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/extract/... ./internal/ingest/... ./internal/analyze/... -count=1; go vet ./internal/extract/... ./internal/ingest/...`
Expected: PASS (`TestGoExtractorCallSites` and `TestGoExtractorSameNamedMethods` unchanged).

- [ ] **Step 5: Docs**
`CHANGELOG.md` `### Changed`: "**Fewer duplicate writes.** The Go extractor emits each node and edge once per file, and a `build`/`sync once` run writes each shared `pkg://` stub once; `build`'s node and edge counts drop accordingly."

- [ ] **Step 6: Commit**

```powershell
git add internal/extract/ast/golang.go internal/extract/ast/golang_test.go internal/ingest/engine.go internal/ingest/engine_test.go CHANGELOG.md
git commit -m "perf(ingest): dedupe Go extraction and write shared stubs once per run"
```

### Task C4: Canonical Windows path case in `file://` URIs

**Finding:** `model.FileURI` (`internal/graph/model/uri.go:10-18`) keeps the input's case, so `chiton build c:\repo` and `chiton build .` index the same file under two URIs. Node IDs hash the URI (`internal/graph/model/node.go:59-62`), so the result is two node sets. The mismatch is real on this machine: the scratch directory reported as `…\Temp\claude\C--Programming-Repos…` is `c--Programming-Repos…` on disk, and `filepath.EvalSymlinks` returns the on-disk form. `golang.org/x/sys` is only an indirect dependency (`go.mod`). The stdlib is enough: `filepath.EvalSymlinks` upper-cases the drive letter and normalises each component on Windows (`$GOROOT/src/path/filepath/symlink_windows.go`), and `syscall.FindFirstFile` gives a single element's stored name. No new dependency is needed. Existing stores need no Section B migration. A migration would have to re-ID every node and edge, because IDs derive from the URI. Instead, prune drops a non-canonical cache alias once its canonical twin is re-indexed: a one-time re-index of the affected files.
**Priority:** P2   **Effort:** M   **Depends on:** C2
**Behaviour change:** On Windows, `file://` URIs use an upper-case drive letter and on-disk case, and resolve symlinked directory components. On the next `build`/`sync once`, files an older build stored under another case are re-indexed once and the alias is dropped. Decisions or concepts linked to the alias IDs are lost (unavoidable, since the IDs change). macOS case-insensitive volumes stay out of scope.

**Files:**
- Modify: `internal/graph/model/uri.go:10-18`
- Create: `internal/graph/model/uri_windows.go`
- Create: `internal/graph/model/uri_other.go`
- Create: `internal/graph/model/uri_windows_test.go`
- Modify: `internal/graph/model/helpers_test.go:57-83` (`TestFileURI`)
- Modify: `internal/ingest/engine.go` (`prune` from C2, `pruneDir` `:356-371`; add `underURI`)
- Modify (test-only, Section D's package — coordinate): `internal/analyze/incremental_test.go:48,75,103,147` (these compare hand-built URIs against `FileURI`-derived finding scopes)
- Test: `internal/ingest/engine_test.go`

**Interfaces:**
- Consumes: `Engine.prune` (C2)
- Produces: none (`model.FileURI` keeps its signature)

- [ ] **Step 1: Write the failing test**

Create `internal/graph/model/uri_windows_test.go`:

```go
//go:build windows

package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestFileURICanonicalCase checks every case spelling of one file, and a missing file in the same directory,
// map to one URI with an upper-case drive letter.
func TestFileURICanonicalCase(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "Mixed.go")
	if err := os.WriteFile(p, []byte("package x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	want := FileURI(p)
	if d := want[len(fileScheme)]; d < 'A' || d > 'Z' {
		t.Fatalf("FileURI(%q) = %q, want an upper-case drive letter", p, want)
	}
	for _, variant := range []string{strings.ToLower(p), strings.ToUpper(p)} {
		if got := FileURI(variant); got != want {
			t.Errorf("FileURI(%q) = %q, want %q", variant, got, want)
		}
	}
	gone := FileURI(filepath.Join(strings.ToUpper(dir), "Gone.go"))
	if w := strings.TrimSuffix(want, "Mixed.go") + "Gone.go"; gone != w {
		t.Errorf("FileURI(missing file) = %q, want %q", gone, w)
	}
}
```

Append to `internal/ingest/engine_test.go` (add `"strings"` to its imports):

```go
// TestSyncDirPrunesCaseAliases checks a cache entry an older build left under a lower-cased URI is dropped
// once the file is re-indexed under its canonical URI.
func TestSyncDirPrunesCaseAliases(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("path case aliases only arise on case-insensitive Windows paths")
	}
	h := newHarness(t, false, false)
	ctx := context.Background()
	p := h.write(t, "a.go", "package src\nfunc A() {}\n")
	alias := "file://" + filepath.ToSlash(strings.ToLower(p))
	if err := h.gs.UpsertNodes(ctx, []model.Node{{ID: model.NewNodeID(alias, alias), Kind: model.KindFile,
		Label: "a.go", SourceURI: alias}}); err != nil {
		t.Fatal(err)
	}
	h.cache.Mark(alias, "stale")

	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	if n := len(h.nodesForURI(t, alias)); n != 0 {
		t.Errorf("alias %s kept %d node(s)", alias, n)
	}
	if len(h.nodesForURI(t, model.FileURI(p))) == 0 {
		t.Error("the canonical URI should be indexed")
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/graph/model/ ./internal/ingest/ -run 'TestFileURICanonicalCase|TestSyncDirPrunesCaseAliases' -count=1` (on Windows)
Expected: FAIL. You should see `FileURI("c:\users\…\mixed.go") = "file://c:/users/…/mixed.go", want "file://C:/Users/…/Mixed.go"` and `alias file://c:/… kept 1 node(s)`.

- [ ] **Step 3: Implement**

Replace `FileURI` in `internal/graph/model/uri.go`:

```go
// FileURI returns the canonical file:// source URI for path: absolute, forward slashes, and on Windows the
// on-disk case of every existing component (see canonicalPath). If path can't be made absolute, it is used as given.
func FileURI(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		return fileScheme + filepath.ToSlash(path)
	}
	return fileScheme + filepath.ToSlash(canonicalPath(abs))
}
```

Create `internal/graph/model/uri_other.go`:

```go
//go:build !windows

package model

// canonicalPath returns abs unchanged; only Windows paths get case normalisation.
func canonicalPath(abs string) string { return abs }
```

Create `internal/graph/model/uri_windows.go`:

```go
//go:build windows

package model

import (
	"path/filepath"
	"strings"
	"sync"
	"syscall"
)

// canonicalDirs caches directories resolved to their on-disk form, since FileURI runs several times per file.
var canonicalDirs sync.Map

// canonicalPath returns abs with an upper-case drive letter and every existing component in its on-disk case,
// so c:\repo\x.go and C:\Repo\x.go name one file. A missing file keeps its given base name.
func canonicalPath(abs string) string {
	dir, base := filepath.Split(abs)
	if base == "" {
		return canonicalDir(filepath.Clean(abs))
	}
	if name, ok := onDiskName(abs); ok {
		base = name
	}
	return filepath.Join(canonicalDir(filepath.Clean(dir)), base)
}

// canonicalDir resolves dir with filepath.EvalSymlinks, which on Windows also normalises case, falling back to
// its parent for a directory that does not exist. Only successful resolutions are cached.
func canonicalDir(dir string) string {
	if v, ok := canonicalDirs.Load(dir); ok {
		return v.(string)
	}
	if c, err := filepath.EvalSymlinks(dir); err == nil {
		canonicalDirs.Store(dir, c)
		return c
	}
	parent := filepath.Dir(dir)
	if parent == dir {
		return upperDrive(dir)
	}
	return filepath.Join(canonicalDir(parent), filepath.Base(dir))
}

// onDiskName returns the stored name of path's last element, if it exists.
func onDiskName(path string) (string, bool) {
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return "", false
	}
	var data syscall.Win32finddata
	h, err := syscall.FindFirstFile(p, &data)
	if err != nil {
		return "", false
	}
	_ = syscall.FindClose(h)
	return syscall.UTF16ToString(data.FileName[:]), true
}

func upperDrive(p string) string {
	if len(p) >= 2 && p[1] == ':' {
		return strings.ToUpper(p[:1]) + p[1:]
	}
	return p
}
```

In `internal/graph/model/helpers_test.go` `TestFileURI`, right after `wd, err := os.Getwd()` and its error check, add:

```go
	if runtime.GOOS == "windows" {
		// FileURI reports on-disk case, which the working directory's spelling may not match.
		if wd, err = filepath.EvalSymlinks(wd); err != nil {
			t.Fatal(err)
		}
	}
```

and change the round-trip check to compare with `strings.EqualFold` on Windows (add `"strings"` to the imports):

```go
		abs, _ := filepath.Abs(in)
		p, ok := PathFromFileURI(FileURI(in))
		if same := p == abs || (runtime.GOOS == "windows" && strings.EqualFold(p, abs)); !ok || !same {
			t.Errorf("PathFromFileURI(FileURI(%q)) = %q, %v; want %q", in, p, ok, abs)
		}
```

In `internal/ingest/engine.go`, add `underURI`, use it in `pruneDir`, and replace `prune`:

```go
// underURI reports whether uri is dirURI or a file:// URI beneath it.
func underURI(uri, dirURI string) bool {
	return uri == dirURI || strings.HasPrefix(uri, strings.TrimSuffix(dirURI, "/")+"/")
}

// prune deletes indexed files under root that are gone from disk, plus non-canonical aliases (older builds kept
// the input path's case on Windows) of files this walk re-indexed. Present but excluded files survive.
func (e *Engine) prune(ctx context.Context, root string, seen map[string]bool, sum *BatchSummary) error {
	rootURI := model.FileURI(root)
	for _, uri := range e.cache.URIs() {
		if err := ctx.Err(); err != nil {
			return err
		}
		p, ok := model.PathFromFileURI(uri)
		if !ok || seen[uri] {
			continue
		}
		canon := model.FileURI(p)
		if !underURI(canon, rootURI) {
			continue
		}
		if canon == uri || !seen[canon] {
			if _, err := os.Stat(p); err == nil {
				continue
			}
		}
		out, err := e.deleteURI(ctx, uri)
		if err != nil {
			return err
		}
		e.tally(sum, out)
	}
	return nil
}

// pruneDir removes every file indexed under dir (a renamed or deleted directory yields no per-child
// events) and returns how many it pruned.
func (e *Engine) pruneDir(ctx context.Context, dir string) int {
	prefix := model.FileURI(dir)
	pruned := 0
	for _, uri := range e.cache.URIs() {
		if underURI(uri, prefix) {
			if _, err := e.deleteURI(ctx, uri); err != nil {
				e.log("sync: prune %s failed: %v", uri, err)
				continue
			}
			pruned++
		}
	}
	return pruned
}
```

In `internal/analyze/incremental_test.go:48,75,103,147`, replace each hand-built `"file://" + filepath.ToSlash(mustAbs(t, X))` with `model.FileURI(X)`, importing `github.com/Eric-Fernald/chiton/internal/graph/model` if missing. The finding scopes they are compared against come from `FileURI`. (`llm_test.go:309` and `analyzer_test.go:171` need no change: they only assert rejection or survival.)

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/graph/model/... ./internal/ingest/... ./internal/analyze/... ./internal/extract/... -count=1; go vet ./internal/graph/model/... ./internal/ingest/...`
Expected: PASS on Windows and in CI (`uri_windows_test.go` compiles only on Windows; `TestSyncDirPrunesCaseAliases` skips elsewhere).

- [ ] **Step 5: Docs**
`CHANGELOG.md` `### Fixed`: "**Windows path case no longer duplicates files.** `file://` URIs use an upper-case drive letter and on-disk case, so `chiton build c:\repo` and `chiton build .` index one copy. The next `build`/`sync once` drops entries an older build stored under a different case (those files are re-indexed once)."

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/model internal/ingest/engine.go internal/ingest/engine_test.go internal/analyze/incremental_test.go CHANGELOG.md
git commit -m "fix(model): canonical Windows path case in file URIs and prune case aliases"
```

### Task C5: Rescan after an fsnotify event overflow

**Finding:** `Watcher.Run` passes every error to `OnError` (`internal/ingest/fswatch/watcher.go:96-102`), and the engine only logs it (`internal/ingest/engine.go:316`). When a bulk checkout overflows the buffer, fsnotify reports `ErrEventOverflow` (`fsnotify@v1.10.1/backend_windows.go:582`, `backend_inotify.go:398`), and the dropped changes are never synced. The graph silently diverges until the next manual `sync once`.
**Priority:** P2   **Effort:** S   **Depends on:** C1, C4
**Behaviour change:** After an overflow, `sync watch` logs `watch: event overflow — rescanning <root>` and re-enqueues every file under the root plus every indexed file gone from disk. The per-file queue (content cache) then syncs only what changed. Repeated overflows coalesce into one rescan.

**Files:**
- Modify: `internal/ingest/fswatch/watcher.go` (`Options`, `Run`; add `handleError`, import `"errors"`)
- Modify: `internal/ingest/engine.go` (`Watch`; add `rescan`)
- Test: `internal/ingest/fswatch/watcher_test.go`, `internal/ingest/engine_test.go`

**Interfaces:**
- Consumes: `walk.Files`, `walk.Policy` (C1); `underURI` (C4)
- Produces: `fswatch.Options.OnOverflow func()`

- [ ] **Step 1: Write the failing test**

Append to `internal/ingest/fswatch/watcher_test.go` (add `"errors"` to its imports):

```go
// TestOverflowCallsOnOverflow checks an event overflow reaches OnOverflow as well as OnError, and other errors
// reach only OnError.
func TestOverflowCallsOnOverflow(t *testing.T) {
	var overflows, errs int
	opts := goOpts()
	opts.OnOverflow = func() { overflows++ }
	opts.OnError = func(error) { errs++ }
	w, err := New(t.TempDir(), func(string) {}, opts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = w.Close() })

	w.handleError(fsnotify.ErrEventOverflow)
	w.handleError(errors.New("other"))
	if overflows != 1 || errs != 2 {
		t.Errorf("overflows=%d errors=%d, want 1 and 2", overflows, errs)
	}
}
```

Append to `internal/ingest/engine_test.go` (add `"github.com/Eric-Fernald/chiton/internal/walk"` to its imports):

```go
// TestRescanAfterOverflow checks the overflow rescan enqueues changed, new, and vanished files, and that
// processing them reconciles the graph.
func TestRescanAfterOverflow(t *testing.T) {
	h := newHarness(t, false, false)
	ctx := context.Background()
	pa := h.write(t, "a.go", "package src\nfunc A() {}\n")
	pb := h.write(t, "b.go", "package src\nfunc B() {}\n")
	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	// Changes the watcher dropped: edit a.go, delete b.go, add c.go.
	h.write(t, "a.go", "package src\nfunc A2() {}\n")
	if err := os.Remove(pb); err != nil {
		t.Fatal(err)
	}
	pc := h.write(t, "c.go", "package src\nfunc C() {}\n")

	policy, err := walk.LoadPolicy(h.dir)
	if err != nil {
		t.Fatal(err)
	}
	var queued []string
	if err := h.eng.rescan(ctx, h.dir, policy, func(p string) { queued = append(queued, p) }); err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, p := range queued {
		got[model.FileURI(p)] = true
	}
	for _, want := range []string{pa, pb, pc} {
		if !got[model.FileURI(want)] {
			t.Errorf("rescan did not enqueue %s (queued %v)", want, queued)
		}
	}

	for _, p := range queued {
		if err := h.eng.syncKey(ctx, p); err != nil {
			t.Fatal(err)
		}
	}
	if _, ok := h.nodesForURI(t, model.FileURI(pa))["A2"]; !ok {
		t.Error("edited a.go not re-synced")
	}
	if n := len(h.nodesForURI(t, model.FileURI(pb))); n != 0 {
		t.Errorf("deleted b.go still has %d node(s)", n)
	}
	if len(h.nodesForURI(t, model.FileURI(pc))) == 0 {
		t.Error("new c.go not indexed")
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/ingest/ ./internal/ingest/fswatch/ -run 'TestOverflowCallsOnOverflow|TestRescanAfterOverflow' -count=1`
Expected: FAIL to compile: `unknown field OnOverflow in struct literal`, `w.handleError undefined`, `h.eng.rescan undefined`.

- [ ] **Step 3: Implement**

In `internal/ingest/fswatch/watcher.go`, add `"errors"` to the imports and add to `Options`, after `OnError`:

```go
	// OnOverflow, if set, runs when fsnotify reports dropped events (ErrEventOverflow, e.g. a bulk checkout on
	// Windows), so the consumer can rescan instead of silently missing changes.
	OnOverflow func()
```

Replace `Run` and add `handleError`:

```go
// Run processes events until ctx is cancelled, then returns ctx.Err().
func (w *Watcher) Run(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err, ok := <-w.w.Errors:
			if !ok {
				return nil
			}
			w.handleError(err)
		case ev, ok := <-w.w.Events:
			if !ok {
				return nil
			}
			w.handle(ev)
		}
	}
}

// handleError reports a watcher error and, when fsnotify dropped events, calls OnOverflow first.
func (w *Watcher) handleError(err error) {
	if err == nil {
		return
	}
	if errors.Is(err, fsnotify.ErrEventOverflow) && w.opts.OnOverflow != nil {
		w.opts.OnOverflow()
	}
	if w.opts.OnError != nil {
		w.opts.OnError(err)
	}
}
```

In `internal/ingest/engine.go`, add `rescan` and replace `Watch`:

```go
// rescan enqueues every supported file under root and every indexed file under it gone from disk, so after an
// overflow the file queue re-syncs what changed and deletes what vanished; the content cache skips the rest.
func (e *Engine) rescan(ctx context.Context, root string, policy *walk.Policy, enqueue func(path string)) error {
	seen := map[string]bool{}
	if _, err := walk.Files(ctx, root, policy, func(path string) error {
		if extast.ForFile(path) != nil {
			seen[model.FileURI(path)] = true
			enqueue(path)
		}
		return nil
	}, func(path string, err error) { e.log("watch: rescan: skip %s: %v", path, err) }); err != nil {
		return err
	}
	rootURI := model.FileURI(root)
	for _, uri := range e.cache.URIs() {
		if seen[uri] || !underURI(uri, rootURI) {
			continue
		}
		if p, ok := model.PathFromFileURI(uri); ok {
			if _, err := os.Stat(p); errors.Is(err, fs.ErrNotExist) {
				enqueue(p)
			}
		}
	}
	return nil
}

// Watch re-syncs changed files under root until ctx is cancelled, then drains in-flight work and re-clusters.
// Queues run on context.Background() so a cancel stops fsnotify but Drain still processes queued edits.
func (e *Engine) Watch(ctx context.Context, root string) error {
	policy, err := walk.LoadPolicy(root)
	if err != nil {
		return err
	}
	// A single-worker, longer-debounced queue serialises cluster passes.
	clusterQ := jobs.New(func(c context.Context, _ string) error {
		_, err := e.recluster(c)
		return err
	}, jobs.Options{Workers: 1, Debounce: e.cfg.ClusterDebounce})
	clusterQ.Start(context.Background())

	fileQ := jobs.New(e.syncKey, jobs.Options{
		Workers:  e.cfg.Workers,
		Debounce: e.cfg.Debounce,
		OnError: func(key string, err error) {
			e.log("sync: %s: %v", key, err)
		},
		OnDone: func(key string, _ error) {
			clusterQ.Enqueue("cluster")
		},
	})
	fileQ.Start(context.Background())

	// An overflow means fsnotify dropped changes; one coalesced rescan feeds the whole tree to fileQ.
	rescanQ := jobs.New(func(c context.Context, _ string) error {
		e.log("watch: event overflow — rescanning %s", root)
		return e.rescan(c, root, policy, fileQ.Enqueue)
	}, jobs.Options{Workers: 1, Debounce: e.cfg.Debounce})
	rescanQ.Start(context.Background())

	w, err := fswatch.New(root, fileQ.Enqueue, fswatch.Options{
		Skip:       policy.Skip,
		Accept:     func(p string) bool { return extast.ForFile(p) != nil },
		OnError:    func(err error) { e.log("watch: %v", err) },
		OnOverflow: func() { rescanQ.Enqueue("rescan") },
		// A renamed/removed directory yields no per-child events, so prune the
		// graph for every file previously indexed under it.
		OnDirRemove: func(dir string) {
			if n := e.pruneDir(context.Background(), dir); n > 0 {
				e.log("watch: directory %s removed — pruned %d file(s)", dir, n)
				clusterQ.Enqueue("cluster")
			}
		},
	})
	if err != nil {
		rescanQ.Stop()
		fileQ.Stop()
		clusterQ.Stop()
		return err
	}

	e.log("watch: watching %s (%d dirs) — Ctrl-C to stop", root, w.Watched())
	runErr := w.Run(ctx) // blocks until ctx cancelled
	_ = w.Close()

	// Graceful shutdown: drain a pending rescan (it feeds fileQ), then queued file work (its OnDone re-arms the
	// cluster trigger), then the cluster queue, then a final flush.
	rescanQ.Drain()
	fileQ.Drain()
	clusterQ.Drain()

	flushCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if _, err := e.recluster(flushCtx); err != nil {
		e.log("watch: final re-cluster failed: %v", err)
	}
	if err := e.cache.Save(); err != nil {
		e.log("watch: cache save failed: %v", err)
	}

	if runErr != nil && !errors.Is(runErr, context.Canceled) {
		return runErr
	}
	return nil
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/ingest/... -count=1; go vet ./internal/ingest/...`
Expected: PASS (`TestWatchGracefulShutdown` still passes with the extra queue).

- [ ] **Step 5: Docs**
- `README.md` Live sync paragraph (`:325-329`): add "If the OS drops file events (a bulk checkout can overflow the Windows buffer), `sync watch` rescans the tree automatically."
- `CHANGELOG.md` `### Fixed`: "**`sync watch` recovers from event overflow.** When fsnotify reports dropped events, the watcher rescans the tree instead of silently missing changes."

- [ ] **Step 6: Commit**

```powershell
git add internal/ingest/fswatch internal/ingest/engine.go internal/ingest/engine_test.go README.md CHANGELOG.md
git commit -m "fix(watch): rescan the tree after an fsnotify event overflow"
```

---

## Section D: Analyzer, clustering and git hooks

This section makes live-sync analysis linear, makes the git hooks fail open and run without the store, repairs the Leiden aggregation that collapsed the graph into a few giant communities, and cuts the cost of incremental and full clustering. Order: D1 (prune rework) before D2 (hooks reuse its prune path) and D6 (re-cluster seeds use D1's prior findings). D4 adds `model.IsExternal`, which D6 consumes. D5 edits `circular_import.go` after D4. D3 and D7 are independent. Every task was applied to a scratch copy of the repo: `go build ./...`, `go vet ./...`, `gofmt -l .` and `go test ./... -count=1` all pass with D1 through D7 applied together. The timings below were measured on this machine (32 logical CPUs) with scratch stores built from `$(go env GOROOT)/src` and from a copy of this repo.

### Task D1: Prune live-sync findings from the file's own edges, not a store scan

**Finding:** Each live-sync re-analyze goes `SyncFile` → `AnalyzeExtracted` (`internal/ingest/engine.go:190`) → `pruneResolved` (`internal/analyze/incremental.go:56`). That call loads every deterministic finding in the store (`internal/analyze/analyzer.go:198-201`, filter `kind`+`rule_provenance` only), decodes them all, and only then checks `target_uri` in Go (`:210`). `persist` also re-upserts every applicable rule node for every file (`analyzer.go:165-170`). The cost grows with files × findings. Reproduced: `sync once` over GOROOT `src/net` (390 files) took 10.1 s with the default `--analyze` and 0.8 s without; `src/runtime` (890 files) took 26.7 s and 3.1 s. Adding `target_uri` to the probe on its own only halves this (4.7 s / 13.0 s), because every call still iterates every node record. The fix is to find the file's findings by adjacency: all 76 FileRule findings carry an AFFECTS edge to the file node or one of its symbols. Those edges have to be read before `DeleteBySourceURI` removes them. With that change: net 3.4 s, runtime 8.8 s. The remaining cost is linear rule evaluation, about 6 ms per file.
**Priority:** P1   **Effort:** M   **Depends on:** none
**Integration:** Lands before Task C3 in the execution order, against today's `SyncFile`, which calls `DeleteBySourceURI`. When C3 later swaps that for `store.ReplaceSources`, the `ReadPriorFindings` call must stay immediately before it, since findings on nodes C3 drops lose their AFFECTS edges there.
**Behaviour change:** none. The pruned set is the same, and rule nodes are written once per `Analyzer` rather than once per file.

**Files:**
- Modify: `internal/analyze/analyzer.go:3-16` (imports), `:20-27` (Analyzer), `:79` (Analyze), `:159-216` (persist, pruneResolved)
- Modify: `internal/analyze/incremental.go:14-66` (AnalyzeFile, AnalyzeExtracted, analyzeTarget)
- Modify: `internal/ingest/engine.go:156-159`, `:189-193` (SyncFile)
- Test: `internal/analyze/prune_test.go` (new)

**Interfaces:**
- Consumes: none
- Produces: `type analyze.PriorFindings struct{ /* unexported */ }`; `func (a *Analyzer) ReadPriorFindings(ctx context.Context, uri string) (*PriorFindings, error)`; `func (a *Analyzer) ReanalyzeExtracted(ctx context.Context, path string, content []byte, nodes []model.Node, edges []model.Edge, prior *PriorFindings) (*Result, error)`. `AnalyzeExtracted` keeps its signature (it becomes `ReanalyzeExtracted(..., nil)`).

- [ ] **Step 1: Write the failing test**

Create `internal/analyze/prune_test.go`:

```go
package analyze_test

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

// storeSpy records the prune queries and rule-node upserts that reach the wrapped store.
type storeSpy struct {
	*badgerstore.Store
	filters     []store.NodeFilter
	ruleUpserts int
}

func (s *storeSpy) NodesMatching(ctx context.Context, f store.NodeFilter) ([]model.Node, error) {
	s.filters = append(s.filters, f)
	return s.Store.NodesMatching(ctx, f)
}

func (s *storeSpy) UpsertNodes(ctx context.Context, nodes []model.Node) error {
	for _, n := range nodes {
		if n.Kind == model.KindRule {
			s.ruleUpserts++
		}
	}
	return s.Store.UpsertNodes(ctx, nodes)
}

// syncFile mimics ingest.Engine.SyncFile: read prior findings, replace path's nodes, then re-analyze.
func syncFile(t *testing.T, gs store.GraphStore, az *analyze.Analyzer, path, src string) {
	t.Helper()
	ctx := context.Background()
	write(t, path, src)
	uri := model.FileURI(path)
	prior, err := az.ReadPriorFindings(ctx, uri)
	if err != nil {
		t.Fatal(err)
	}
	nodes, edges, err := extast.ForFile(path).ExtractSource(ctx, path, []byte(src))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gs.DeleteBySourceURI(ctx, uri); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, edges); err != nil {
		t.Fatal(err)
	}
	if _, err := az.ReanalyzeExtracted(ctx, path, []byte(src), nodes, edges, prior); err != nil {
		t.Fatalf("ReanalyzeExtracted: %v", err)
	}
}

// TestReanalyzeExtractedPrunesFromPrior checks live sync prunes a resolved finding without querying the
// store, and leaves another file's findings alone.
func TestReanalyzeExtractedPrunesFromPrior(t *testing.T) {
	spy := &storeSpy{Store: openStore(t)}
	az := analyze.New(spy, nil)
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a.go"), filepath.Join(dir, "b.go")
	syncFile(t, spy, az, a, "package a\n\n// TODO: fix a\nfunc A() {}\n")
	syncFile(t, spy, az, b, "package a\n\n// TODO: fix b\nfunc B() {}\n")

	spy.filters = nil
	syncFile(t, spy, az, a, "package a\n\nfunc A() {}\n")
	if len(spy.filters) != 0 {
		t.Errorf("re-sync with prior findings queried the store: %+v", spy.filters)
	}
	if got := countFindings(t, spy.Store, model.FileURI(a), todoRule); got != 0 {
		t.Errorf("a.go's resolved TODO survived: %d left", got)
	}
	if got := countFindings(t, spy.Store, model.FileURI(b), todoRule); got != 1 {
		t.Errorf("b.go's TODO was touched: want 1, got %d", got)
	}
}

// TestAnalyzeFile_PruneScopedToFile checks a single-file prune queries only that file's findings and rejects a
// probe false positive: another file's finding that merely mentions this file's URI.
func TestAnalyzeFile_PruneScopedToFile(t *testing.T) {
	ctx := context.Background()
	spy := &storeSpy{Store: openStore(t)}
	az := analyze.New(spy, nil)
	dir := t.TempDir()
	pathA, pathB := filepath.Join(dir, "a.go"), filepath.Join(dir, "b.go")
	uriA, uriB := model.FileURI(pathA), model.FileURI(pathB)
	write(t, pathA, "package a\n\n// TODO: fix a\nfunc A() {}\n")
	write(t, pathB, "package a\n\n// TODO: fix b\nfunc B() {}\n")
	for _, p := range []string{pathA, pathB} {
		if _, err := az.AnalyzeFile(ctx, p); err != nil {
			t.Fatalf("AnalyzeFile %s: %v", p, err)
		}
	}
	decoyID := "decoyfinding0000000000000000bb01"
	decoy := model.Node{
		ID: decoyID, Kind: model.KindFinding, Label: todoRule, SourceURI: "finding://" + decoyID,
		Provenance: model.ProvExtracted, Confidence: 1,
		Attrs: map[string]any{
			"rule_id": todoRule, "message": "see " + uriA, "target_uri": uriB,
			"rule_provenance": string(analyze.ProvenanceDeterministic),
		},
	}
	if err := spy.UpsertNodes(ctx, []model.Node{decoy}); err != nil {
		t.Fatal(err)
	}

	write(t, pathA, "package a\n\nfunc A() {}\n")
	spy.filters = nil
	if _, err := az.AnalyzeFile(ctx, pathA); err != nil {
		t.Fatal(err)
	}
	if len(spy.filters) != 1 || spy.filters[0].Attrs["target_uri"] != uriA {
		t.Errorf("prune queries = %+v, want one scoped to target_uri %s", spy.filters, uriA)
	}
	if got := countFindings(t, spy.Store, uriA, todoRule); got != 0 {
		t.Errorf("a.go's resolved TODO survived: %d left", got)
	}
	if got := countFindings(t, spy.Store, uriB, todoRule); got != 2 {
		t.Errorf("b.go's findings (real + decoy) were touched: want 2, got %d", got)
	}
}

// TestPersistUpsertsRuleNodesOnce checks an Analyzer writes each rule node once, not on every file.
func TestPersistUpsertsRuleNodesOnce(t *testing.T) {
	spy := &storeSpy{Store: openStore(t)}
	az := analyze.New(spy, nil)
	dir := t.TempDir()
	for _, name := range []string{"a.go", "b.go"} {
		p := filepath.Join(dir, name)
		write(t, p, "package a\n\n// TODO: x\nfunc F() {}\n")
		before := spy.ruleUpserts
		if _, err := az.AnalyzeFile(context.Background(), p); err != nil {
			t.Fatal(err)
		}
		if name == "b.go" && spy.ruleUpserts != before {
			t.Errorf("second file re-upserted %d rule nodes", spy.ruleUpserts-before)
		}
	}
	if spy.ruleUpserts == 0 {
		t.Error("first file upserted no rule nodes")
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/analyze/ -run "TestReanalyzeExtractedPrunesFromPrior|TestAnalyzeFile_PruneScopedToFile|TestPersistUpsertsRuleNodesOnce" -count=1`
Expected: FAIL — build failed: `az.ReadPriorFindings undefined` and `az.ReanalyzeExtracted undefined`. If those two methods are stubbed, the next failures are `prune queries = [...]`, because the filter has no `target_uri`, and `second file re-upserted N rule nodes`.

- [ ] **Step 3: Implement**

`internal/analyze/analyzer.go`: add `"maps"` and `"slices"` to the imports. Add two fields to `Analyzer`:

```go
type Analyzer struct {
	store    store.GraphStore
	registry *Registry
	now      func() time.Time

	fileRulesOnce sync.Once
	fileRules     fileRuleSet // AnalyzeFile's rule list, which the fileRuleSet method builds lazily

	rulesMu   sync.Mutex
	rulesDone map[string]bool // rule node IDs this Analyzer has upserted; rule metadata never changes
}
```

In `Analyze` (line 79), the prune call becomes `a.pruneResolved(ctx, w.scannedURIs, ids, nil, nil)`. Replace `persist` and `pruneResolved` (lines 159-216) with:

```go
// persist upserts rule nodes (so VIOLATES edges resolve), finding nodes, and VIOLATES and AFFECTS edges.
// It returns the set of finding IDs so callers need not hash each finding again.
func (a *Analyzer) persist(ctx context.Context, rules []Rule, findings []Finding) (map[string]bool, error) {
	now := a.now()

	nodes := make([]model.Node, 0, len(rules)+len(findings))
	var fresh []string // rule nodes this call upserts for the first time
	for _, r := range rules {
		n := RuleAsNode(r.Meta(), now)
		if a.rulePersisted(n.ID) {
			continue
		}
		fresh = append(fresh, n.ID)
		nodes = append(nodes, n)
	}
	edges := make([]model.Edge, 0, len(findings)*2)
	ids := make(map[string]bool, len(findings))
	for _, f := range findings {
		id := f.FindingID()
		ids[id] = true
		nodes, edges = f.appendGraph(nodes, edges, id, RuleNodeID(f.RuleID), now)
	}
	if err := a.store.UpsertNodes(ctx, nodes); err != nil {
		return nil, fmt.Errorf("upsert nodes: %w", err)
	}
	if err := a.store.UpsertEdges(ctx, edges); err != nil {
		return nil, fmt.Errorf("upsert edges: %w", err)
	}
	a.markRulesPersisted(fresh)
	return ids, nil
}

// rulePersisted reports whether this Analyzer already upserted rule node id.
func (a *Analyzer) rulePersisted(id string) bool {
	a.rulesMu.Lock()
	defer a.rulesMu.Unlock()
	return a.rulesDone[id]
}

func (a *Analyzer) markRulesPersisted(ids []string) {
	a.rulesMu.Lock()
	defer a.rulesMu.Unlock()
	if a.rulesDone == nil {
		a.rulesDone = make(map[string]bool, len(ids))
	}
	for _, id := range ids {
		a.rulesDone[id] = true
	}
}

// pruneByURIMax is the scanned-file count up to which pruneResolved queries findings per target_uri, so the
// store's byte probe skips other files' records undecoded; larger passes load every finding once.
const pruneByURIMax = 8

// pruneResolved deletes a scanned file's deterministic findings that are missing from newIDs. LLM findings
// are pruned separately. A non-nil ruleScope limits the delete to the rule IDs that ran, and a non-nil prior
// supplies the candidates instead of a store query.
func (a *Analyzer) pruneResolved(ctx context.Context, scanned, newIDs, ruleScope map[string]bool, prior *PriorFindings) error {
	if len(scanned) == 0 {
		return nil
	}
	var findings []model.Node
	if prior != nil {
		findings = prior.nodes
	} else {
		var err error
		if findings, err = a.deterministicFindings(ctx, scanned); err != nil {
			return fmt.Errorf("load nodes for prune: %w", err)
		}
	}
	var stale []model.Node
	for _, n := range findings {
		if ruleScope != nil && !ruleScope[model.AttrString(n.Attrs, "rule_id")] {
			continue // this pass did not run the rule, so leave its findings intact
		}
		if !scanned[model.AttrString(n.Attrs, "target_uri")] || newIDs[n.ID] {
			continue
		}
		stale = append(stale, n)
	}
	return deleteFindingNodes(ctx, a.store, stale)
}

// deterministicFindings loads the deterministic findings that may target a scanned file.
func (a *Analyzer) deterministicFindings(ctx context.Context, scanned map[string]bool) ([]model.Node, error) {
	filter := func(uri string) store.NodeFilter {
		attrs := map[string]string{"rule_provenance": string(ProvenanceDeterministic)}
		if uri != "" {
			attrs["target_uri"] = uri
		}
		return store.NodeFilter{Kinds: []model.NodeKind{model.KindFinding}, Attrs: attrs}
	}
	if len(scanned) > pruneByURIMax {
		return store.NodesMatching(ctx, a.store, filter(""))
	}
	var out []model.Node
	for _, uri := range slices.Sorted(maps.Keys(scanned)) {
		ns, err := store.NodesMatching(ctx, a.store, filter(uri))
		if err != nil {
			return nil, err
		}
		out = append(out, ns...)
	}
	return out, nil
}
```

(`RuleNodeID(f.RuleID)` equals the old `ruleNodes` lookup: `RuleAsNode` keys each node by `RuleNodeID(m.ID)`.)

`internal/analyze/incremental.go`: `AnalyzeFile` ends with `return a.analyzeTarget(ctx, ft, start, nil)`. Replace `AnalyzeExtracted` (lines 24-29) with:

```go
// AnalyzeExtracted is AnalyzeFile for a caller that already read path (content) and extracted it
// (nodes and edges as ExtractSource returned them), so the file isn't read and parsed twice.
func (a *Analyzer) AnalyzeExtracted(ctx context.Context, path string, content []byte, nodes []model.Node, edges []model.Edge) (*Result, error) {
	return a.ReanalyzeExtracted(ctx, path, content, nodes, edges, nil)
}

// PriorFindings holds a file's deterministic findings as ReadPriorFindings saw them.
type PriorFindings struct{ nodes []model.Node }

// ReadPriorFindings returns uri's deterministic findings by following the AFFECTS edges into its file node
// and the symbols it CONTAINS, a read the size of the file. Call it before replacing uri's nodes, since
// deleting them drops those edges.
func (a *Analyzer) ReadPriorFindings(ctx context.Context, uri string) (*PriorFindings, error) {
	kinds := []model.EdgeKind{model.EdgeContains, model.EdgeAffects}
	sub, err := a.store.Neighbors(ctx, model.NewNodeID(uri, uri), 2, kinds)
	if err != nil {
		return nil, fmt.Errorf("read prior findings for %s: %w", uri, err)
	}
	p := &PriorFindings{}
	for _, n := range sub.Nodes {
		if n.Kind == model.KindFinding && model.AttrString(n.Attrs, "rule_provenance") == string(ProvenanceDeterministic) {
			p.nodes = append(p.nodes, n)
		}
	}
	return p, nil
}

// ReanalyzeExtracted is AnalyzeExtracted for live sync: it prunes from prior, read by ReadPriorFindings
// before the caller replaced path's nodes, instead of querying the store. A nil prior queries.
func (a *Analyzer) ReanalyzeExtracted(ctx context.Context, path string, content []byte, nodes []model.Node, edges []model.Edge, prior *PriorFindings) (*Result, error) {
	start := a.now()
	return a.analyzeTarget(ctx, newFileTarget(path, content, nodes, edges), start, prior)
}
```

In `analyzeTarget` (lines 31-66), change only the signature and the prune call. The body is otherwise unchanged:

```go
func (a *Analyzer) analyzeTarget(ctx context.Context, ft FileTarget, start time.Time, prior *PriorFindings) (*Result, error) {
	// ... unchanged down to the prune ...
	// Limit the prune to the rules that ran, because this file's GraphScope findings must survive the edit.
	if err := a.pruneResolved(ctx, scanned, ids, ruleScope, prior); err != nil {
		return nil, err
	}
	// ... unchanged ...
}
```

`internal/ingest/engine.go`, `SyncFile`: read the prior findings immediately before the delete at line 156, and pass them to the re-analyze at line 190:

```go
	// Read the file's findings while the AFFECTS edges into its nodes still exist; the delete drops them.
	var prior *analyze.PriorFindings
	if e.cfg.Reanalyze && e.analyzer != nil {
		if prior, err = e.analyzer.ReadPriorFindings(ctx, uri); err != nil {
			e.log("sync: %s: %v", path, err) // a nil prior makes the re-analyze query the store instead
		}
	}
	deleted, err := e.store.DeleteBySourceURI(ctx, uri)
```

```go
	if e.cfg.Reanalyze && e.analyzer != nil {
		if _, err := e.analyzer.ReanalyzeExtracted(ctx, path, content, nodes, edges, prior); err != nil {
			e.log("sync: %s: re-analyze failed: %v", path, err)
		}
	}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/analyze/... ./internal/ingest/... -count=1; go vet ./internal/analyze/... ./internal/ingest/...`
Expected: PASS. The existing `TestAnalyzeFile_*`, `TestPrune*`, `TestReanalyze` and `TestAnalyzeExtractedMatchesAnalyzeFile` show the prune results are unchanged. Optional timing check against a scratch store: `$env:CHITON_GRAPH_PATH="$env:TEMP\d1\graph"; chiton sync once "$(go env GOROOT)\src\net"` should take about 3.4 s (was 10.1 s).

- [ ] **Step 5: Docs**

`CHANGELOG.md`, under `## [Unreleased]` → `### Changed`, add:

```markdown
- **Live sync with `--analyze` is linear again.** Re-analyzing a file used to load and decode every
  deterministic finding in the store to prune that file's; it now reads the file's findings through their
  AFFECTS edges before replacing its nodes, and an Analyzer writes each rule node once. `sync once` over
  Go's `src/net` (390 files) dropped from 10.1 s to 3.4 s.
```

- [ ] **Step 6: Commit**

```powershell
git add internal/analyze/analyzer.go internal/analyze/incremental.go internal/analyze/prune_test.go internal/ingest/engine.go CHANGELOG.md
git commit -m "perf(analyze): prune live-sync findings via AFFECTS edges, not a store scan"
```

### Task D2: Git hooks gate in memory, honour the timeout, and fail open

**Finding:** `hooks run` opens the store read-write before analysing anything (`cmd/chiton/main.go:1183-1187`). A running `sync watch`, which holds the Badger lock, or an encrypted store with no TTY (git gives hooks no stdin), therefore fails `git commit` with a store error. A malformed `hooks.yaml` also blocks (`:1177-1180`). `timeout-seconds` builds a deadline (`:1190-1194`) that nothing reads: `Analyze`'s walk never checks `ctx` (`internal/analyze/analyzer.go:112-142`), and neither does `LLMService.Pending` (`internal/analyze/llm.go:160-220`). One unreadable directory aborts either walk (`analyzer.go:113-114`, `llm.go:161-162`). The gate also scans the whole working tree, not the commit, and the post-commit/post-checkout `sync git` hooks fail when the store is locked. A failed post-checkout hook makes `git checkout` exit non-zero. No deterministic rule needs the store: `FileRule.MatchFile` and `GraphRule.MatchGraph` take only in-memory targets (`internal/analyze/rule.go:108-117`).
**Priority:** P1   **Effort:** L   **Depends on:** D1, C1
**Integration:** Task C1 puts the shared ignore policy in the leaf package `internal/walk`, so `analyze` can import it without a cycle. Implement `ignoredDir` below as `p.Skip(path, true)` (the walked path, not a relative one; `Skip` computes the relative path itself) on a `*walk.Policy` loaded once per scan with `walk.LoadPolicy(root)`, instead of a local list.
**Behaviour change:** Yes.
- `hooks run --stage pre-commit` checks the staged blobs of the files in the commit, in memory, with FileScope rules only. Pre-push still scans the tree, also in memory.
- A store that is missing, locked or passphrase-protected, a bad `hooks.yaml` (the default policy is used), a git error, or a run past `timeout-seconds` prints a warning and exits 0. Only a finding blocks.
- Findings are recorded to the store when it opens without prompting; otherwise they are skipped with a note.
- `sync git --stage …` warns instead of failing.
- `chiton analyze` skips unreadable directories and reports how many it skipped.

**DECISION NEEDED:** gate on the staged files only (recommended). The alternative keeps the whole-tree scan at pre-commit, which blocks commits on findings in files the commit doesn't touch and would bring back GraphScope rules. Timeouts fail open (recommended). The alternative is to fail closed on timeout only.

**Files:**
- Modify: `internal/analyze/analyzer.go:3-16`, `:42-157` (Result, Analyze, walk/graph helpers)
- Create: `internal/analyze/walk.go`
- Modify: `internal/analyze/incremental.go` (`analyzeTarget` as left by D1)
- Modify: `internal/analyze/llm.go:3-21`, `:144-172` (Pending)
- Modify: `internal/ingest/git/git.go:5-11`, `:112-124`
- Modify: `internal/hooks/run.go:13-65`
- Modify: `cmd/chiton/main.go:3-18` (import `errors`), `:1158-1215` (hooksRunCmd); add `recordHookScan`
- Modify: `cmd/chiton/sync.go:130-182`
- Modify: `cmd/chiton/analyze.go:78-79`
- Test: `internal/analyze/scan_test.go`, `internal/ingest/git/staged_test.go`, `internal/hooks/run_test.go`, `cmd/chiton/hooks_test.go` (all new)

**Interfaces:**
- Consumes: D1's `PriorFindings`, `pruneResolved(ctx, scanned, newIDs, ruleScope map[string]bool, prior *PriorFindings) error`. Task C1's shared ignore helper: `ignoredDir` below is its only call site in `internal/analyze`. C1's helper must not live in `internal/ingest`, because `ingest` imports `analyze` and that would be an import cycle. `internal/fsutil` or a new leaf package works.
- Produces: `func (a *Analyzer) Scan(ctx context.Context, root string) (*Result, error)`; `type SourceFile struct{ Path string; Content []byte }`; `func (a *Analyzer) ScanSources(ctx context.Context, files []SourceFile) (*Result, error)`; `func (a *Analyzer) Record(ctx context.Context, res *Result) error`; `Result.Skipped []string`; `type gitpkg.StagedFile struct{ Path string; Content []byte }`; `func gitpkg.StagedFiles(ctx context.Context, root string, keep func(path string) bool) ([]StagedFile, error)`; `hooks.Run(ctx context.Context, opts RunOptions) (*Result, error)` (the store parameter is removed); `hooks.Result.Scan *analyze.Result`, `hooks.Result.Skipped []string`.

- [ ] **Step 1: Write the failing tests**

`internal/analyze/scan_test.go`:

```go
package analyze_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
)

// TestScanSkipsUnreadableDir checks one unreadable directory is reported in Skipped instead of aborting
// the whole scan. chmod cannot deny directory reads on Windows or to root, so the test skips there.
func TestScanSkipsUnreadableDir(t *testing.T) {
	if runtime.GOOS == "windows" || os.Geteuid() == 0 {
		t.Skip("needs POSIX permissions and a non-root user")
	}
	root := t.TempDir()
	write(t, filepath.Join(root, "ok.go"), "package a\n\n// TODO: fix\nfunc A() {}\n")
	locked := filepath.Join(root, "locked")
	if err := os.Mkdir(locked, 0o755); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(locked, "b.go"), "package b\n")
	if err := os.Chmod(locked, 0o000); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(locked, 0o755) })

	res, err := analyze.New(nil, nil).Scan(context.Background(), root)
	if err != nil {
		t.Fatalf("Scan aborted on an unreadable dir: %v", err)
	}
	if res.FilesScanned != 1 || len(res.Skipped) != 1 || res.Skipped[0] != locked {
		t.Errorf("FilesScanned=%d Skipped=%v, want 1 file and %s skipped", res.FilesScanned, res.Skipped, locked)
	}
}

// TestScanStopsWhenContextDone checks a done context ends the walk with ctx's error.
func TestScanStopsWhenContextDone(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "a.go"), "package a\n")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := analyze.New(nil, nil).Scan(ctx, root); !errors.Is(err, context.Canceled) {
		t.Fatalf("Scan with a cancelled context = %v, want context.Canceled", err)
	}
}

// TestScanSourcesSkipsGraphRules checks the in-memory scan runs file rules only and never needs a store.
func TestScanSourcesSkipsGraphRules(t *testing.T) {
	dir := t.TempDir()
	res, err := analyze.New(nil, nil).ScanSources(context.Background(), []analyze.SourceFile{
		{Path: filepath.Join(dir, "a.go"), Content: []byte("package a\n\n// TODO: fix\nfunc helper() {}\n")},
		{Path: filepath.Join(dir, "notes.txt"), Content: []byte("TODO: not source\n")},
	})
	if err != nil {
		t.Fatal(err)
	}
	var todo bool
	for _, f := range res.Findings {
		todo = todo || f.RuleID == todoRule
		if f.RuleID == "architecture/dead-symbol" {
			t.Errorf("GraphScope rule ran on a partial file set: %+v", f)
		}
	}
	if !todo || res.FilesScanned != 1 {
		t.Errorf("FilesScanned=%d findings=%+v, want 1 file with a TODO finding", res.FilesScanned, res.Findings)
	}
}
```

`internal/ingest/git/staged_test.go`:

```go
package git

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestStagedFiles checks StagedFiles returns index content, not the working copy, and skips deletions,
// untracked files, and paths keep rejects.
func TestStagedFiles(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	runGit := gitRunner(t, root)
	write := func(name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runGit("init")
	write("gone.go", "package p\n")
	runGit("add", "gone.go")
	runGit("commit", "-m", "base")

	write("a.go", "package p // staged\n")
	write("notes.txt", "staged text\n")
	runGit("add", "a.go", "notes.txt")
	runGit("rm", "-q", "gone.go")
	write("a.go", "package p // working copy\n")
	write("untracked.go", "package p\n")

	files, err := StagedFiles(context.Background(), root, func(p string) bool { return filepath.Ext(p) == ".go" })
	if err != nil {
		t.Fatalf("StagedFiles: %v", err)
	}
	if len(files) != 1 || filepath.Base(files[0].Path) != "a.go" {
		t.Fatalf("want only a.go, got %+v", files)
	}
	if got := string(files[0].Content); got != "package p // staged\n" {
		t.Errorf("content = %q, want the staged blob", got)
	}
}
```

`internal/hooks/run_test.go`:

```go
package hooks

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules" // register rules
)

const leak = "package a\n\nconst Key = \"AKIAIOSFODNN7EXAMPLE\"\n"

// gitIn runs git in dir with a fixed identity, failing t on error.
func gitIn(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
	cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestRunPreCommitGatesStagedContent checks the pre-commit gate judges the staged blobs: a staged secret
// blocks even after the working copy drops it, and unstaged or untracked secrets never block.
func TestRunPreCommitGatesStagedContent(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	gitIn(t, repo, "init")
	path := filepath.Join(repo, "leak.go")
	writeFile(t, path, leak)
	gitIn(t, repo, "add", "leak.go")
	writeFile(t, path, "package a\n")
	opts := RunOptions{Stage: StagePreCommit, Config: Default().PreCommit, Root: repo}

	res, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !res.Blocks() {
		t.Fatalf("staged secret should block: %+v", res)
	}

	gitIn(t, repo, "add", "leak.go")
	writeFile(t, path, leak)
	writeFile(t, filepath.Join(repo, "wip.go"), leak)
	if res, err = Run(context.Background(), opts); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Blocks() || res.FilesScanned != 1 {
		t.Errorf("clean index: FilesScanned=%d, blocking=%d; want 1 file, 0 blocking", res.FilesScanned, res.BlockingCount)
	}
}

// TestRunHonoursContext checks a done context stops the scan, which is how timeout-seconds takes effect.
func TestRunHonoursContext(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "a.go"), leak)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Run(ctx, RunOptions{Stage: StagePrePush, Config: Default().PrePush, Root: root}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Run with a cancelled context = %v, want context.Canceled", err)
	}
}
```

`cmd/chiton/hooks_test.go`:

```go
package main

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	chitoncrypto "github.com/Eric-Fernald/chiton/internal/crypto"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
	"github.com/spf13/cobra"
)

// gitRepo creates a repository holding files, staged; commit also commits them.
func gitRepo(t *testing.T, files map[string]string, commit bool) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	git("init")
	for name, body := range files {
		p := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		git("add", filepath.ToSlash(name))
	}
	if commit {
		git("commit", "-m", "c")
	}
	return dir
}

// unavailableStore points config at a store a hook cannot open: held open (locked, as by a running
// `sync watch`) or beside a keyfile with no passphrase available.
func unavailableStore(t *testing.T, kind string) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("CHITON_PROJECT_DIR", "")
	t.Setenv(chitoncrypto.EnvVarPassphrase, "")
	graph := filepath.Join(t.TempDir(), "graph")
	t.Setenv("CHITON_GRAPH_PATH", graph)
	switch kind {
	case "locked":
		gs, err := badgerstore.Open(graph)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { gs.Close() })
	case "encrypted":
		if err := os.MkdirAll(graph, 0o755); err != nil {
			t.Fatal(err)
		}
		kf, err := chitoncrypto.DefaultKeyFile()
		if err != nil {
			t.Fatal(err)
		}
		if err := kf.Save(filepath.Join(filepath.Dir(graph), chitoncrypto.KeyfileName), false); err != nil {
			t.Fatal(err)
		}
	}
}

func execute(t *testing.T, cmd *cobra.Command, args ...string) (stderr string, err error) {
	t.Helper()
	var out, errOut bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&errOut)
	cmd.SetArgs(args)
	err = cmd.ExecuteContext(context.Background())
	return errOut.String(), err
}

// TestHooksRunFailsOpen checks an unusable store or policy file never blocks a commit, while a staged
// finding still does.
func TestHooksRunFailsOpen(t *testing.T) {
	for _, kind := range []string{"locked", "encrypted"} {
		t.Run(kind, func(t *testing.T) {
			unavailableStore(t, kind)
			repo := gitRepo(t, map[string]string{"a.go": "package a\n\nfunc A() {}\n", ".chiton/hooks.yaml": "pre-commit: ["}, false)
			stderr, err := execute(t, hooksRunCmd(), "--stage", "pre-commit", "--dir", repo)
			if err != nil {
				t.Fatalf("hooks run blocked a clean commit: %v\n%s", err, stderr)
			}
			for _, want := range []string{"using the default policy", "findings not recorded"} {
				if !strings.Contains(stderr, want) {
					t.Errorf("stderr lacks %q:\n%s", want, stderr)
				}
			}

			leaky := gitRepo(t, map[string]string{"k.go": "package a\n\nconst Key = \"AKIAIOSFODNN7EXAMPLE\"\n"}, false)
			if _, err := execute(t, hooksRunCmd(), "--stage", "pre-commit", "--dir", leaky); err == nil {
				t.Error("a staged critical finding should still block")
			}
		})
	}
}

// TestSyncGitHookFailsOpen checks the post-commit hook warns instead of failing when the store is locked.
func TestSyncGitHookFailsOpen(t *testing.T) {
	unavailableStore(t, "locked")
	repo := gitRepo(t, map[string]string{"a.go": "package a\n"}, true)
	stderr, err := execute(t, syncGitCmd(), "--stage", "post-commit", "--root", repo)
	if err != nil || !strings.Contains(stderr, "graph not updated") {
		t.Fatalf("sync git hook = %v, stderr %q; want a warning and exit 0", err, stderr)
	}
}
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `go test ./internal/analyze/ ./internal/ingest/git/ ./internal/hooks/ ./cmd/chiton/ -run "TestScan|TestStagedFiles|TestRun|TestHooksRunFailsOpen|TestSyncGitHookFailsOpen" -count=1`
Expected: FAIL. In analyze, the build fails with `analyze.New(nil, nil).Scan undefined` and `undefined: analyze.SourceFile`. In git: `undefined: StagedFiles`. In hooks: `not enough arguments in call to Run`. In cmd/chiton: `TestHooksRunFailsOpen/locked: hooks run blocked a clean commit: parse …hooks.yaml…`, and `TestSyncGitHookFailsOpen: sync git hook = open store: … Cannot acquire directory lock …`.

- [ ] **Step 3: Implement**

Create `internal/analyze/walk.go`. It is the one walker for both the deterministic and LLM passes:

```go
package analyze

import (
	"context"
	"io/fs"
	"path/filepath"

	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
)

// walkSources calls fn with each file under root that an extractor supports, skipping ignored directories.
// Unreadable entries go to skipped instead of failing the walk, and ctx's error ends it.
func walkSources(ctx context.Context, root string, fn func(path string) error) (skipped []string, err error) {
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if cerr := ctx.Err(); cerr != nil {
			return cerr
		}
		if err != nil {
			if path == root {
				return err // an unreadable root is the caller's error, not a skip
			}
			skipped = append(skipped, path)
			if d != nil && d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			if ignoredDir(root, path, d) {
				return filepath.SkipDir
			}
			return nil
		}
		if extast.ForFile(path) == nil {
			return nil
		}
		return fn(path)
	})
	return skipped, err
}

// ignoredDir reports whether the walk skips dir. Task C1's shared ignore policy replaces this body.
func ignoredDir(root, path string, d fs.DirEntry) bool {
	return path != root && skipDir(d.Name())
}
```

`internal/analyze/analyzer.go`: change the imports to `context, errors, fmt, maps, os, slices, strings, sync, time` plus the three internal packages. `io/fs` and `path/filepath` move to `walk.go`. Replace `Result` through `matchGraphRules` (lines 42-157) with:

```go
// Result summarises one analyze run.
type Result struct {
	FilesScanned int
	RulesRun     int
	Findings     []Finding
	Duration     time.Duration
	Skipped      []string // paths the walk or extractor could not read; the scan went on without them

	rules     []Rule          // rule nodes Record upserts
	scanned   map[string]bool // file URIs whose resolved findings Record prunes
	ruleScope map[string]bool // rule IDs Record may prune; nil means every rule
	prior     *PriorFindings  // prune candidates read before a re-sync; nil queries the store
}

// Analyze runs FileScope rules over each supported file under root, then GraphScope rules over
// the whole extracted set, and upserts every finding plus the rules they violated.
func (a *Analyzer) Analyze(ctx context.Context, root string) (*Result, error) {
	start := a.now()
	res, err := a.Scan(ctx, root)
	if err != nil {
		return nil, err
	}
	if err := a.Record(ctx, res); err != nil {
		return nil, err
	}
	res.Duration = a.now().Sub(start)
	return res, nil
}

// Scan is Analyze without the store: it runs every FileScope and GraphScope rule over the supported files
// under root, stopping once ctx is done, and returns findings that Record can persist.
func (a *Analyzer) Scan(ctx context.Context, root string) (*Result, error) {
	start := a.now()
	rules := a.registry.All()

	var fileRules []FileRule
	var graphRules []GraphRule
	for _, r := range rules {
		switch v := r.(type) {
		case FileRule:
			fileRules = append(fileRules, v)
		case GraphRule:
			graphRules = append(graphRules, v)
		}
	}

	w, err := a.walkAndMatch(ctx, root, fileRules)
	if err != nil {
		return nil, err
	}
	graphFindings, err := a.matchGraphRules(ctx, graphRules, w.nodes, w.edges)
	if err != nil {
		return nil, err
	}
	return &Result{
		FilesScanned: w.filesScanned,
		// Count only rules actually executed here; LLM rules are dispatched
		// separately via LLMService and are neither FileRule nor GraphRule.
		RulesRun: len(fileRules) + len(graphRules),
		Findings: append(w.findings, graphFindings...),
		Duration: a.now().Sub(start),
		Skipped:  w.skipped,
		rules:    rules,
		scanned:  w.scannedURIs, // every rule ran, so ruleScope stays nil
	}, nil
}

// SourceFile is one file's path and content for ScanSources, such as a blob staged in the git index.
type SourceFile struct {
	Path    string
	Content []byte
}

// ScanSources runs the FileScope rules over in-memory files without touching the store. GraphScope rules
// need the whole corpus (a partial one reports live symbols as dead), so they are skipped.
func (a *Analyzer) ScanSources(ctx context.Context, files []SourceFile) (*Result, error) {
	start := a.now()
	res := newFileResult()
	for _, sf := range files {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		ex := extast.ForFile(sf.Path)
		if ex == nil {
			continue
		}
		nodes, edges, err := ex.ExtractSource(ctx, sf.Path, sf.Content)
		if err != nil {
			res.Skipped = append(res.Skipped, sf.Path)
			continue
		}
		a.matchFile(ctx, newFileTarget(sf.Path, sf.Content, nodes, edges), res)
	}
	res.Duration = a.now().Sub(start)
	return res, nil
}

// Record upserts res's rule nodes and findings, then deletes the findings of res's files, from the rules
// res ran, that res no longer holds. The Analyzer needs a store.
func (a *Analyzer) Record(ctx context.Context, res *Result) error {
	if a.store == nil {
		return errors.New("analyze: Record needs an Analyzer with a store")
	}
	ids, err := a.persist(ctx, res.rules, res.Findings)
	if err != nil {
		return err
	}
	return a.pruneResolved(ctx, res.scanned, ids, res.ruleScope, res.prior)
}

// walkResult accumulates the per-file outputs of walkAndMatch.
type walkResult struct {
	nodes    []model.Node
	edges    []model.Edge
	findings []Finding
	// scannedURIs limits pruneResolved to the files analysed in this pass, so `analyze <subdir>` leaves
	// findings for other files alone.
	scannedURIs  map[string]bool
	filesScanned int
	skipped      []string
}

// walkAndMatch walks root, extracts each supported source file, and runs the
// applicable FileScope rules over it.
func (a *Analyzer) walkAndMatch(ctx context.Context, root string, fileRules []FileRule) (*walkResult, error) {
	w := &walkResult{scannedURIs: map[string]bool{}}
	langs := make([][]string, len(fileRules))
	for i, fr := range fileRules {
		langs[i] = fr.Meta().Languages
	}
	skipped, walkErr := walkSources(ctx, root, func(path string) error {
		ft, nodes, ok := loadFileTarget(ctx, path)
		if !ok {
			return nil
		}
		w.filesScanned++
		if ft.URI != "" {
			w.scannedURIs[ft.URI] = true
		}

		ext := FileExt(path)
		for i, fr := range fileRules {
			if ruleApplies(langs[i], ext) {
				w.findings = append(w.findings, fr.MatchFile(ctx, ft)...)
			}
		}

		w.nodes = append(w.nodes, nodes...)
		w.edges = append(w.edges, ft.Edges...)
		return nil
	})
	if walkErr != nil {
		return nil, fmt.Errorf("walk: %w", walkErr)
	}
	w.skipped = skipped
	return w, nil
}

// matchGraphRules runs the GraphScope rules over the full extracted set, stopping once ctx is done.
func (a *Analyzer) matchGraphRules(ctx context.Context, graphRules []GraphRule, nodes []model.Node, edges []model.Edge) ([]Finding, error) {
	gt := GraphTarget{Nodes: nodes, Edges: edges}
	var findings []Finding
	for _, gr := range graphRules {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		findings = append(findings, gr.MatchGraph(ctx, gt)...)
	}
	return findings, nil
}
```

`internal/analyze/incremental.go`: replace `analyzeTarget` (D1's version) with a thin wrapper over the shared per-file matcher. Add two helpers:

```go
// analyzeTarget runs the FileRules applicable to ft and replaces ft's findings from those rules.
func (a *Analyzer) analyzeTarget(ctx context.Context, ft FileTarget, start time.Time, prior *PriorFindings) (*Result, error) {
	res := newFileResult()
	res.prior = prior
	a.matchFile(ctx, ft, res)
	if err := a.Record(ctx, res); err != nil {
		return nil, err
	}
	res.Duration = a.now().Sub(start)
	return res, nil
}

// newFileResult starts a Result for FileRule-only passes. Its prune covers only the rules that ran, so a
// file's GraphScope findings survive a single-file re-analyze.
func newFileResult() *Result {
	return &Result{scanned: map[string]bool{}, ruleScope: map[string]bool{}}
}

// matchFile runs the FileRules applicable to ft and adds ft, the rules, and their findings to res.
func (a *Analyzer) matchFile(ctx context.Context, ft FileTarget, res *Result) {
	ext := FileExt(ft.Path)
	rs := a.fileRuleSet()
	for i, fr := range rs.rules {
		if !ruleApplies(rs.langs[i], ext) {
			continue
		}
		if !res.ruleScope[rs.ids[i]] {
			res.ruleScope[rs.ids[i]] = true
			res.rules = append(res.rules, fr)
		}
		res.Findings = append(res.Findings, fr.MatchFile(ctx, ft)...)
	}
	res.RulesRun = len(res.rules)
	res.FilesScanned++
	if ft.URI != "" {
		res.scanned[ft.URI] = true
	}
}
```

`internal/analyze/llm.go`: drop the `io/fs` and `extast` imports. In `Pending`, rename the `_ context.Context` parameter to `ctx` and replace the head of the walk (lines 160-172) so the body starts at the extension check. The rest of the callback, including the `filepath.SkipAll` limit, is unchanged:

```go
func (s *LLMService) Pending(ctx context.Context, root, ruleFilter string, limit int) ([]LLMTaskSpec, error) {
	// ... unchanged setup ...
	var specs []LLMTaskSpec
	_, walkErr := walkSources(ctx, root, func(path string) error {
		ext := FileExt(path)
		// LLM rules target source code, so skip doc and image files even though they have extractors.
		if !isCodeFile(ext) {
			return nil
		}
		// ... unchanged: applicable rules, read, spec building, limit ...
	})
	if walkErr != nil {
		return nil, fmt.Errorf("walk: %w", walkErr)
	}
	return specs, nil
}
```

`internal/ingest/git/git.go`: the imports become `bufio, bytes, context, fmt, io, os/exec, path/filepath, strconv, strings`. Replace `run` (lines 112-124) and add the staged-file reader:

```go
func run(root string, args ...string) (string, error) {
	out, err := runBytes(context.Background(), root, nil, args...)
	return string(out), err
}

// runBytes runs git in root with stdin (nil for none) and returns stdout; ctx ending kills git.
func runBytes(ctx context.Context, root string, stdin io.Reader, args ...string) ([]byte, error) {
	// core.quotepath=false stops git octal-escaping non-ASCII paths (e.g. "caf\303\251.go"), which
	// would otherwise skip or mis-delete UTF-8-named files.
	full := append([]string{"-C", root, "-c", "core.quotepath=false"}, args...)
	cmd := exec.CommandContext(ctx, "git", full...)
	cmd.Stdin = stdin
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if cerr := ctx.Err(); cerr != nil {
			return nil, cerr
		}
		return nil, fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}

// StagedFile is one path's content as staged in the index.
type StagedFile struct {
	Path    string // absolute
	Content []byte
}

// StagedFiles returns the index content of each added, copied, modified, or renamed path keep accepts,
// read through one `git cat-file --batch`, so a pre-commit gate checks exactly what the commit records.
func StagedFiles(ctx context.Context, root string, keep func(path string) bool) ([]StagedFile, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	names, err := runBytes(ctx, root, nil, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
	if err != nil {
		return nil, err
	}
	var files []StagedFile
	var req bytes.Buffer
	for rel := range strings.SplitSeq(string(names), "\x00") {
		// cat-file --batch reads one object name per line, so a path holding a newline can't be requested.
		if rel == "" || strings.ContainsRune(rel, '\n') {
			continue
		}
		if p := abs(absRoot, rel); keep(p) {
			files = append(files, StagedFile{Path: p})
			req.WriteString(":" + rel + "\n")
		}
	}
	if len(files) == 0 {
		return nil, nil
	}
	blobs, err := runBytes(ctx, root, &req, "cat-file", "--batch")
	if err != nil {
		return nil, err
	}
	r := bufio.NewReader(bytes.NewReader(blobs))
	for i := range files {
		if files[i].Content, err = readBatchObject(r); err != nil {
			return nil, fmt.Errorf("git cat-file: %w", err)
		}
	}
	return files, nil
}

// readBatchObject reads one `git cat-file --batch` record, "<oid> <type> <size>\n<content>\n". A missing
// object ("<name> missing\n") yields nil content.
func readBatchObject(r *bufio.Reader) ([]byte, error) {
	header, err := r.ReadString('\n')
	if err != nil {
		return nil, err
	}
	if strings.HasSuffix(header, " missing\n") {
		return nil, nil
	}
	fields := strings.Fields(header)
	if len(fields) != 3 {
		return nil, fmt.Errorf("unexpected header %q", header)
	}
	size, err := strconv.Atoi(fields[2])
	if err != nil {
		return nil, fmt.Errorf("unexpected header %q", header)
	}
	buf := make([]byte, size+1) // the content plus its trailing LF
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	return buf[:size], nil
}
```

`internal/hooks/run.go`: the imports become `context, fmt, io, time`, `analyze`, `extast "…/internal/extract/ast"` and `gitpkg "…/internal/ingest/git"`. The `store` import is dropped. Replace `RunOptions`, `Result` and `Run`. `Blocks` and `FormatResult` are unchanged.

```go
// RunOptions controls one hook-stage invocation.
type RunOptions struct {
	Stage  Stage
	Config StageConfig
	Root   string // repository root
}

// Result is what Run returns to the caller.
type Result struct {
	Stage           Stage             `json:"stage"`
	FilesScanned    int               `json:"files_scanned"`
	RulesRun        int               `json:"rules_run"`
	TotalFindings   int               `json:"total_findings"`
	BlockingMin     analyze.Severity  `json:"blocking_min,omitempty"`
	BlockingCount   int               `json:"blocking_count"`
	BlockingDetails []analyze.Finding `json:"blocking_details,omitempty"`
	Skipped         []string          `json:"skipped,omitempty"`
	Duration        time.Duration     `json:"duration"`
	// Scan is the analysis behind the verdict, which the caller may Record into a store.
	Scan *analyze.Result `json:"-"`
}

// Run analyzes in memory without a store (pre-commit checks the staged blobs, pre-push walks Root) and
// collects findings at or above the stage's MinBlockingSeverity. Bound it with Config.TimeoutSeconds via ctx.
func Run(ctx context.Context, opts RunOptions) (*Result, error) {
	az := analyze.New(nil, nil)
	start := time.Now()
	var res *analyze.Result
	var err error
	if opts.Stage == StagePreCommit {
		res, err = scanStaged(ctx, az, opts.Root)
	} else {
		res, err = az.Scan(ctx, opts.Root)
	}
	if err != nil {
		return nil, fmt.Errorf("analyze: %w", err)
	}

	minSev := opts.Config.MinBlockingSeverity()
	out := &Result{
		Stage:         opts.Stage,
		FilesScanned:  res.FilesScanned,
		RulesRun:      res.RulesRun,
		TotalFindings: len(res.Findings),
		BlockingMin:   minSev,
		Skipped:       res.Skipped,
		Duration:      time.Since(start),
		Scan:          res,
	}
	if minSev == "" {
		return out, nil
	}
	threshold := analyze.SeverityRank(minSev)
	for _, f := range res.Findings {
		if analyze.SeverityRank(f.Severity) >= threshold {
			out.BlockingCount++
			out.BlockingDetails = append(out.BlockingDetails, f)
		}
	}
	return out, nil
}

// scanStaged runs the file rules over the blobs staged in root's index, so the gate sees what the commit holds.
func scanStaged(ctx context.Context, az *analyze.Analyzer, root string) (*analyze.Result, error) {
	staged, err := gitpkg.StagedFiles(ctx, root, func(p string) bool { return extast.ForFile(p) != nil })
	if err != nil {
		return nil, err
	}
	files := make([]analyze.SourceFile, len(staged))
	for i, s := range staged {
		files[i] = analyze.SourceFile{Path: s.Path, Content: s.Content}
	}
	return az.ScanSources(ctx, files)
}
```

`cmd/chiton/main.go`: add `"errors"` to the imports. In `hooksRunCmd`, replace `Long` and `RunE`:

```go
		Long: `Read .chiton/hooks.yaml and gate one stage in memory: pre-commit checks the
content staged for the commit (file rules only), pre-push scans the whole tree.
It exits non-zero only when a finding crosses the stage's block-on threshold. A
bad policy file falls back to the defaults, and a timeout (timeout-seconds), a
git error, or an unusable graph store prints a warning and lets the commit or
push through. Findings are recorded when the store opens without prompting.

This is what the installed .git/hooks/pre-commit and pre-push scripts call.
You usually don't invoke it directly.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			stage := hooks.Stage(stageRaw)
			if !slices.Contains(hooks.Stages(), stage) {
				return fmt.Errorf("unknown stage %q - supported: pre-commit, pre-push", stageRaw)
			}
			errOut := cmd.ErrOrStderr()

			// Load falls back to the default policy on a bad file, so warn and carry on with it.
			cfg, err := hooks.Load(filepath.Join(dir, ".chiton", "hooks.yaml"))
			if err != nil {
				fmt.Fprintf(errOut, "chiton %s: warning: %v; using the default policy\n", stage, err)
			}
			stageCfg := cfg.For(stage)

			ctx := cmd.Context()
			if stageCfg.TimeoutSeconds > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, time.Duration(stageCfg.TimeoutSeconds)*time.Second)
				defer cancel()
			}

			res, err := hooks.Run(ctx, hooks.RunOptions{Stage: stage, Config: stageCfg, Root: dir})
			if err != nil {
				// Only a finding may block a commit or push; a broken or slow analysis warns and lets it through.
				if errors.Is(err, context.DeadlineExceeded) {
					err = fmt.Errorf("analysis exceeded timeout-seconds (%d)", stageCfg.TimeoutSeconds)
				}
				fmt.Fprintf(errOut, "chiton %s: warning: %v; not blocking\n", stage, err)
				return nil
			}
			hooks.FormatResult(cmd.OutOrStdout(), res)
			recordHookScan(ctx, errOut, stage, res.Scan)
			if res.Blocks() {
				return fmt.Errorf("chiton %s: blocked by %d finding(s) at or above %q",
					res.Stage, res.BlockingCount, res.BlockingMin)
			}
			return nil
		},
```

Add after `hooksRunCmd`:

```go
// recordHookScan persists a hook's findings when the store opens without creating it or prompting. A missing,
// locked, or passphrase-protected store only prints a note, since the gate already ran in memory.
func recordHookScan(ctx context.Context, w io.Writer, stage hooks.Stage, scan *analyze.Result) {
	cfg, err := config.Load()
	if err != nil {
		return
	}
	if _, err := os.Stat(cfg.GraphPath); err != nil {
		return // no graph yet, and a hook must not create one
	}
	gs, err := openGraphStore(cfg.GraphPath, false)
	if err == nil {
		defer gs.Close()
		err = analyze.New(gs, nil).Record(ctx, scan)
	}
	if err != nil {
		fmt.Fprintf(w, "chiton %s: note: findings not recorded (%v)\n", stage, err)
	}
}
```

(Git gives hooks no stdin, so `GetPassphrase` returns `ErrNoPassphrase` rather than prompting. A user running `chiton hooks run` in a terminal can still get a prompt.)

`cmd/chiton/sync.go`: move the body of `syncGitCmd`'s `RunE` (lines 145-175) unchanged into `runSyncGit`, and make hook invocations fail open:

```go
		RunE: func(cmd *cobra.Command, args []string) error {
			err := runSyncGit(cmd, &f, root, since, stage, args)
			if err != nil && stage != "" {
				// From a post-commit/post-checkout hook, git has already finished (and post-checkout's exit
				// status becomes checkout's), so a busy or locked store only warns.
				fmt.Fprintf(cmd.ErrOrStderr(), "chiton sync git (%s): warning: %v; graph not updated\n", stage, err)
				return nil
			}
			return err
		},
```

```go
// runSyncGit re-indexes the files git changed, as `chiton sync git` describes.
func runSyncGit(cmd *cobra.Command, f *syncEngineFlags, root, since, stage string, args []string) error {
	if !gitpkg.Available(root) {
		return fmt.Errorf("sync git: %s is not a git work tree (or git is not on PATH)", root)
	}
	changes, skip, err := resolveGitChanges(root, since, stage, args)
	if err != nil {
		return err
	}
	out := cmd.OutOrStdout()
	if skip {
		fmt.Fprintln(out, "sync git: nothing to do (file checkout, not a branch switch)")
		return nil
	}
	if len(changes) == 0 {
		fmt.Fprintln(out, "sync git: no changed files")
		return nil
	}

	eng, closeStore, err := openSyncEngine(cmd, f)
	if err != nil {
		return err
	}
	defer closeStore()

	sum, err := eng.SyncChanges(cmd.Context(), ingest.GitChanges(changes))
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "sync git: %d upserted, %d deleted, %d skipped, %d re-embedded%s\n",
		sum.Upserted, sum.Deleted, sum.Skipped, sum.Reembedded, reclusterNote(sum.Reclustered))
	return nil
}
```

`cmd/chiton/analyze.go`, after the summary line (78-79):

```go
			if len(res.Skipped) > 0 {
				fmt.Fprintf(cmd.ErrOrStderr(), "analyze: skipped %d unreadable path(s), e.g. %s\n", len(res.Skipped), res.Skipped[0])
			}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/analyze/... ./internal/ingest/... ./internal/hooks/... ./internal/mcp/... ./cmd/chiton/... -count=1; go vet ./internal/analyze/... ./internal/hooks/... ./internal/ingest/git/... ./cmd/chiton/...`
Expected: PASS. `TestScanSkipsUnreadableDir` skips on Windows and runs on the Linux and macOS CI legs. Cross-check the POSIX build with `$env:GOOS="linux"; go vet ./internal/analyze/; $env:GOOS=""`.

- [ ] **Step 5: Docs**

`README.md:311-312`: replace the two-line paragraph that starts "Wire findings into git" with:

```markdown
Wire findings into git with `chiton hooks install`; the per-repo
`.chiton/hooks.yaml` sets the block-on severity threshold per stage. The
pre-commit gate checks the staged content of the files in the commit (file
rules only), pre-push scans the whole tree, and both run in memory. A locked or
passphrase-protected store, a git error, or a run past `timeout-seconds` prints
a warning instead of blocking, and a bad policy file falls back to the defaults.
```

`CHANGELOG.md` → `### Changed`, add:

```markdown
- **The pre-commit hook now gates the commit, not the working tree.** It analyzes the staged blobs of the
  committed files in memory, with file rules only, since graph rules need the whole corpus; pre-push still
  scans the tree. Neither needs the graph store any more, and findings are recorded only when the store
  opens without prompting.
```

`CHANGELOG.md` → `### Fixed`, add:

```markdown
- **Git hooks no longer block commits on infrastructure errors.** A locked store (a running `sync watch`),
  an encrypted store with no passphrase, a git error, or a malformed `hooks.yaml` used to fail
  `git commit`, and a failing `sync git --stage …` also failed `git checkout`. These now warn and exit 0.
  `timeout-seconds` is enforced (the analyzer checks its context per file and per graph rule), and an
  unreadable directory is skipped and counted instead of aborting `chiton analyze` or `analyze pending`.
```

- [ ] **Step 6: Commit**

```powershell
git add internal/analyze/analyzer.go internal/analyze/walk.go internal/analyze/incremental.go internal/analyze/llm.go internal/analyze/scan_test.go internal/ingest/git/git.go internal/ingest/git/staged_test.go internal/hooks/run.go internal/hooks/run_test.go cmd/chiton/main.go cmd/chiton/sync.go cmd/chiton/analyze.go cmd/chiton/hooks_test.go README.md CHANGELOG.md
git commit -m "fix(hooks): gate staged content in memory, honour timeout, fail open on store errors"
```

### Task D3: Keep intra-community weight when Leiden aggregates

**Finding:** `aggregate` skips `d == c` (`internal/graph/cluster/leiden.go:225-235`). A super-node's internal weight therefore never reaches `aggDeg` or `totalW`: its degree becomes only its external weight, and `totalW` shrinks to the inter-community weight. The null-model term in `moveNodes` (`γ·kv·Σc/2m`, `:159`, `:167`) becomes tiny, and every merge looks profitable. `runLeiden` also restarts every level from singletons (`:253`) instead of from the unrefined partition. Result: communities collapse toward connected components. Reproduced on a scratch graph of this repo, 3,226 nodes: 27 communities, the largest holding 2,823 nodes (87.5%). Zachary's karate club (one component) comes out as a single community, Q = 0.000. After the fix: 49 communities with the largest at 492 nodes. Karate Q = 0.4198 at the default seed, and at least 0.392 over 200 seeds.
**Priority:** P1   **Effort:** M   **Depends on:** none
**Behaviour change:** Community IDs and sizes in `cluster.json`, `god_nodes`, `graph_summary` and `report` change substantially: many smaller, meaningful communities replace one giant one. Users should re-run `chiton cluster`.

**Files:**
- Modify: `internal/graph/cluster/leiden.go:95-102` (add `seededState` after `singletonState`), `:136-187` (moveNodes), `:200-238` (aggregate), `:240-274` (runLeiden)
- Test: `internal/graph/cluster/leiden_quality_test.go` (new)

**Interfaces:**
- Consumes: none
- Produces: none (`Run`'s signature is unchanged)

- [ ] **Step 1: Write the failing test**

Create `internal/graph/cluster/leiden_quality_test.go`:

```go
package cluster

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// karateEdges is Zachary's karate club (34 members, 78 ties), 1-based as published.
var karateEdges = [][2]int{
	{1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}, {1, 9}, {1, 11}, {1, 12}, {1, 13}, {1, 14},
	{1, 18}, {1, 20}, {1, 22}, {1, 32}, {2, 3}, {2, 4}, {2, 8}, {2, 14}, {2, 18}, {2, 20}, {2, 22}, {2, 31},
	{3, 4}, {3, 8}, {3, 9}, {3, 10}, {3, 14}, {3, 28}, {3, 29}, {3, 33}, {4, 8}, {4, 13}, {4, 14},
	{5, 7}, {5, 11}, {6, 7}, {6, 11}, {6, 17}, {7, 17}, {9, 31}, {9, 33}, {9, 34}, {10, 34}, {14, 34},
	{15, 33}, {15, 34}, {16, 33}, {16, 34}, {19, 33}, {19, 34}, {20, 34}, {21, 33}, {21, 34}, {23, 33},
	{23, 34}, {24, 26}, {24, 28}, {24, 30}, {24, 33}, {24, 34}, {25, 26}, {25, 28}, {25, 32}, {26, 32},
	{27, 30}, {27, 34}, {28, 34}, {29, 32}, {29, 34}, {30, 33}, {30, 34}, {31, 33}, {31, 34}, {32, 33},
	{32, 34}, {33, 34},
}

// graphOf builds nodes "n0".."n<n-1>" and one edge per pair.
func graphOf(n int, pairs [][2]int) ([]model.Node, []model.Edge) {
	nodes := make([]model.Node, n)
	for i := range nodes {
		id := fmt.Sprint("n", i)
		nodes[i] = makeNode(id, id)
	}
	edges := make([]model.Edge, len(pairs))
	for i, p := range pairs {
		edges[i] = makeEdge(fmt.Sprint("n", p[0]), fmt.Sprint("n", p[1]))
	}
	return nodes, edges
}

// cliques returns k cliques of size nodes each; ring adds one edge from each clique to the next.
func cliques(k, size int, ring bool) [][2]int {
	var pairs [][2]int
	for c := range k {
		base := c * size
		for i := range size {
			for j := i + 1; j < size; j++ {
				pairs = append(pairs, [2]int{base + i, base + j})
			}
		}
		if ring {
			pairs = append(pairs, [2]int{base, ((c + 1) % k) * size})
		}
	}
	return pairs
}

// modularity computes Newman's Q (γ = 1) of comm over the undirected, deduplicated graph.
func modularity(nodes []model.Node, edges []model.Edge, comm map[string]int) float64 {
	idx := make(map[string]int, len(nodes))
	for i, n := range nodes {
		idx[n.ID] = i
	}
	g := buildGraph(len(nodes), edges, idx)
	twoM := 2 * g.totalW
	var internal float64
	tot := map[int]float64{}
	for u := range g.n {
		cu := comm[nodes[u].ID]
		tot[cu] += g.deg[u]
		for _, nb := range g.adj[u] {
			if comm[nodes[nb.id].ID] == cu {
				internal += nb.weight
			}
		}
	}
	q := internal / twoM
	for _, t := range tot {
		q -= (t / twoM) * (t / twoM)
	}
	return q
}

// assertBlocks checks each consecutive run of size nodes shares one community and no two runs share one.
func assertBlocks(t *testing.T, comm map[string]int, k, size int) {
	t.Helper()
	owner := map[int]int{}
	for c := range k {
		want := comm[fmt.Sprint("n", c*size)]
		for i := range size {
			if got := comm[fmt.Sprint("n", c*size+i)]; got != want {
				t.Fatalf("clique %d split: n%d in %d, n%d in %d", c, c*size, want, c*size+i, got)
			}
		}
		if prev, dup := owner[want]; dup {
			t.Fatalf("cliques %d and %d merged into community %d", prev, c, want)
		}
		owner[want] = c
	}
}

// TestTwoCliquesJoinedByOneEdge: a single bridge must not merge two dense cliques.
func TestTwoCliquesJoinedByOneEdge(t *testing.T) {
	pairs := append(cliques(2, 5, false), [2]int{0, 5})
	nodes, edges := graphOf(10, pairs)
	for seed := range int64(10) {
		assertBlocks(t, Run(nodes, edges, 1.0, rand.New(rand.NewSource(seed))), 2, 5)
	}
}

// TestRingOfCliques: ten K5s in a ring (below modularity's resolution limit) give one community per clique.
func TestRingOfCliques(t *testing.T) {
	nodes, edges := graphOf(50, cliques(10, 5, true))
	for seed := range int64(10) {
		assertBlocks(t, Run(nodes, edges, 1.0, rand.New(rand.NewSource(seed))), 10, 5)
	}
}

// TestKarateClubModularity: published Louvain/Leiden partitions of the karate club reach Q ≈ 0.42.
func TestKarateClubModularity(t *testing.T) {
	pairs := make([][2]int, len(karateEdges))
	for i, e := range karateEdges {
		pairs[i] = [2]int{e[0] - 1, e[1] - 1}
	}
	nodes, edges := graphOf(34, pairs)
	for seed := range int64(20) {
		if q := modularity(nodes, edges, Run(nodes, edges, 1.0, rand.New(rand.NewSource(seed)))); q < 0.37 {
			t.Errorf("seed %d: modularity %.4f, want >= 0.37", seed, q)
		}
	}
}
```

(The ring uses 10 × K5 on purpose. With 30 × K5, modularity's resolution limit legitimately pairs up adjacent cliques.)

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/graph/cluster/ -run "TestTwoCliquesJoinedByOneEdge|TestRingOfCliques|TestKarateClubModularity" -count=1`
Expected: FAIL — `cliques 0 and 1 merged into community 0` (both clique tests), and `seed 0: modularity 0.0000, want >= 0.37` for every seed.

- [ ] **Step 3: Implement**

In `internal/graph/cluster/leiden.go`, add after `singletonState`:

```go
// seededState starts g's nodes in the communities comm gives, whose IDs must be below g.n.
func seededState(g *graph, comm []int) *pstate {
	commDeg := make([]float64, g.n)
	for v, c := range comm {
		commDeg[c] += g.deg[v]
	}
	return &pstate{comm: comm, commDeg: commDeg}
}
```

Replace `moveNodes`. Its bool result has no reader once `runLeiden` no longer stops on "nothing moved"; `refine` already ignores it. The inner move logic is unchanged:

```go
// moveNodes moves nodes to the neighbouring community with the best modularity gain for up to 20 sweeps.
// A non-nil coarse keeps each move inside the node's coarse community.
func moveNodes(g *graph, st *pstate, γ float64, rng *rand.Rand, coarse []int) {
	if g.totalW == 0 {
		return
	}
	twoM := 2 * g.totalW
	order := make([]int, g.n)
	nc := newCommWeights(len(st.commDeg))

	for iter := 0; iter < 20; iter++ {
		moved := false
		permutation(order, rng)
		for _, v := range order {
			cOld := st.comm[v]
			kv := g.deg[v]

			nc.clear()
			nc.add(g.adj[v], st.comm)

			// Baseline score for staying in cOld (v excluded from cOld's degree).
			degOld := st.commDeg[cOld] - kv
			base := nc.w[cOld] - γ*kv*degOld/twoM

			bestComm := cOld
			bestGain := 0.0
			for _, c := range nc.list {
				if c == cOld || (coarse != nil && coarse[c] != coarse[v]) {
					continue
				}
				score := nc.w[c] - γ*kv*st.commDeg[c]/twoM
				if gain := score - base; gain > bestGain+eps {
					bestGain = gain
					bestComm = c
				}
			}
			if bestComm == cOld {
				continue
			}
			st.commDeg[cOld] -= kv
			st.commDeg[bestComm] += kv
			st.comm[v] = bestComm
			moved = true
		}
		if !moved {
			break
		}
	}
}
```

Replace `aggregate` and `runLeiden`:

```go
// aggregate builds a reduced graph whose super-nodes are the communities in comm, weighting each edge by the
// summed inter-community edge weight. It also returns comm compacted to 0-based IDs.
// A super-node's degree is its members' total, so its internal weight counts as a self-loop (twice) and
// totalW is unchanged; the self-loop stays out of adj because it moves with the node and cancels in every gain.
func aggregate(g *graph, comm []int) (*graph, []int) {
	comm = compactComm(comm)
	numComm := 0
	for _, c := range comm {
		numComm = max(numComm, c+1)
	}

	members := make([][]int, numComm)
	aggDeg := make([]float64, numComm)
	for u, c := range comm {
		members[c] = append(members[c], u)
		aggDeg[c] += g.deg[u]
	}

	aggAdj := make([][]nbr, numComm)
	cw := newCommWeights(numComm)
	for c, us := range members {
		cw.clear()
		for _, u := range us {
			cw.add(g.adj[u], comm)
		}
		for _, d := range cw.list {
			if d != c {
				aggAdj[c] = append(aggAdj[c], nbr{d, cw.w[d]})
			}
		}
	}
	return &graph{n: numComm, adj: aggAdj, deg: aggDeg, totalW: g.totalW}, comm
}

// runLeiden alternates local moving, refinement, and aggregation until refinement stops merging nodes. Each
// aggregate level starts from the unrefined partition, as Leiden requires, not from singletons.
func runLeiden(g0 *graph, γ float64, rng *rand.Rand) []int {
	// nodeToSuper[orig] = super-node index in the current-level graph.
	nodeToSuper := make([]int, g0.n)
	for i := range nodeToSuper {
		nodeToSuper[i] = i
	}

	g := g0
	st := singletonState(g)
	for {
		moveNodes(g, st, γ, rng, nil)
		refined := refine(g, st.comm, γ, rng)
		agg, compactedRefined := aggregate(g, refined)
		if agg.n >= g.n {
			break // refinement merged nothing, so st.comm is final
		}
		for orig, s := range nodeToSuper {
			nodeToSuper[orig] = compactedRefined[s]
		}
		// Each refined community sits inside one unrefined community, which seeds its aggregate node.
		seed := make([]int, agg.n)
		for u, r := range compactedRefined {
			seed[r] = st.comm[u]
		}
		g = agg
		st = seededState(g, compactComm(seed))
	}

	out := make([]int, len(nodeToSuper))
	for orig, s := range nodeToSuper {
		out[orig] = st.comm[s]
	}
	return compactComm(out)
}
```

Each loop iteration strictly shrinks `g.n` (`agg.n < g.n`), so the loop terminates.

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/graph/... ./internal/ingest/... ./internal/report/... ./internal/mcp/... -count=1; go vet ./internal/graph/cluster/`
Expected: PASS. The existing `TestTwoCliques`, `TestRunDeterministic`, `TestCompactIDs`, the incremental tests and `TestFullMatchesSeparatePasses` still hold.

- [ ] **Step 5: Docs**

`CHANGELOG.md` → `### Fixed`, add:

```markdown
- **Leiden no longer collapses the graph into a few giant communities.** Aggregation dropped each
  community's internal weight from the super-node degree and the total, so every merge looked profitable,
  and each level restarted from singletons instead of the unrefined partition. On this repo's graph the
  largest community fell from 87.5% of nodes to 15%, and Zachary's karate club now scores Q ≈ 0.42 (was 0).
  Re-run `chiton cluster` to refresh `cluster.json`.
```

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/cluster/leiden.go internal/graph/cluster/leiden_quality_test.go CHANGELOG.md
git commit -m "fix(cluster): keep intra-community weight across Leiden aggregation"
```

### Task D4: Let circular-import see cycles through relative imports

**Finding:** Every IMPORTS edge ends at an external package stub. Both `importNode` (`internal/extract/ast/extractor.go:329-340`, `attrs.external=true`) and the Go extractor (`golang.go:48-52`) build it. `CircularImport` skips external targets (`internal/analyze/rules/architecture/circular_import.go:41-43`), so on extracted graphs the rule cannot fire. Its only tests use hand-made file-to-file edges (`architecture_test.go:33-46`). Resolving imports inside the extractor would add cross-file edges into other files' nodes. That breaks the engine's invariant that `DeleteBySourceURI` is safe (`internal/ingest/engine.go:1-2`), so the smaller correct change is inside the rule: map each relative import spec back to the scanned file it names. Go is left out on purpose: it has no relative imports, and `go build` already rejects import cycles.
**Priority:** P3   **Effort:** M   **Depends on:** none
**Behaviour change:** `chiton analyze` (and pre-push) now reports `architecture/circular-import` (severity high) for JS/TS (`./x`, `../x`, index files) and Python (`.x`, `..x.y`) relative-import cycles. Default pre-push blocks only on `critical`, so nothing new blocks by default.

**Files:**
- Modify: `internal/graph/model/node.go` (append `IsExternal`)
- Modify: `internal/analyze/rules/architecture/circular_import.go:3-11` (imports), `:13-15` (doc), `:29-45` (MatchGraph loop), `:73-78` (replace `isExternal` with the resolver)
- Modify: `internal/analyze/rules/architecture/dead_symbol.go:43`
- Test: `internal/analyze/rules/architecture/cycle_test.go` (new)

**Interfaces:**
- Consumes: none
- Produces: `func model.IsExternal(n model.Node) bool` (D6 uses it)

- [ ] **Step 1: Write the failing test**

Create `internal/analyze/rules/architecture/cycle_test.go`:

```go
package architecture

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
)

// extractAll runs the real extractors over in-memory sources rooted at dir, keyed by slash path.
func extractAll(t *testing.T, dir string, srcs [][2]string) analyze.GraphTarget {
	t.Helper()
	var tg analyze.GraphTarget
	for _, s := range srcs {
		p := filepath.Join(dir, filepath.FromSlash(s[0]))
		nodes, edges, err := extast.ForFile(p).ExtractSource(context.Background(), p, []byte(s[1]))
		if err != nil {
			t.Fatalf("extract %s: %v", s[0], err)
		}
		tg.Nodes = append(tg.Nodes, nodes...)
		tg.Edges = append(tg.Edges, edges...)
	}
	return tg
}

// TestCircularImport_ResolvesRelativeImports checks cycles built from extracted relative imports, whose
// IMPORTS edges all land on external package nodes: two TypeScript packages that import each other through
// index files, and two Python modules in one package. A file that only imports into a cycle is not on it.
func TestCircularImport_ResolvesRelativeImports(t *testing.T) {
	tg := extractAll(t, t.TempDir(), [][2]string{
		{"web/a/index.ts", "import { b } from '../b'\nexport const a = 1\n"},
		{"web/b/index.ts", "import { a } from '../a'\nexport const b = 2\n"},
		{"web/c/index.ts", "import { a } from '../a'\n"},
		{"pkg/x.py", "from .y import g\n"},
		{"pkg/y.py", "from .x import f\n"},
		{"pkg/z.py", "import os\n"},
	})
	got := (CircularImport{}).MatchGraph(context.Background(), tg)
	if len(got) != 2 {
		t.Fatalf("want 2 cycles (web/a <-> web/b, pkg/x <-> pkg/y), got %d: %+v", len(got), got)
	}
	var targets []string
	for _, f := range got {
		targets = append(targets, f.TargetURI)
	}
	joined := strings.Join(targets, " ")
	if !strings.Contains(joined, "/web/a/index.ts") && !strings.Contains(joined, "/web/b/index.ts") {
		t.Errorf("no TypeScript cycle reported: %v", targets)
	}
	if !strings.Contains(joined, "/pkg/x.py") && !strings.Contains(joined, "/pkg/y.py") {
		t.Errorf("no Python cycle reported: %v", targets)
	}
	if strings.Contains(joined, "/web/c/") || strings.Contains(joined, "/pkg/z.py") {
		t.Errorf("a file off the cycle was reported: %v", targets)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/analyze/rules/architecture/ -run TestCircularImport -count=1`
Expected: FAIL — `want 2 cycles (web/a <-> web/b, pkg/x <-> pkg/y), got 0: []`. The three existing `TestCircularImport_*` tests still pass.

- [ ] **Step 3: Implement**

Append to `internal/graph/model/node.go`:

```go
// IsExternal reports whether n is an extractor stub for something outside the scanned tree, such as an
// imported package, which extractors mark with attrs.external=true.
func IsExternal(n Node) bool {
	ext, _ := n.Attrs["external"].(bool)
	return ext
}
```

`internal/analyze/rules/architecture/circular_import.go`: the imports become `context, path, slices, sort, strings` plus `analyze` and `model`. Update the type doc and the edge loop:

```go
// CircularImport detects cycles in the IMPORTS graph. Extractors point every import at an external package
// node, so the rule resolves relative imports back to the scanned file they name; other external targets
// cannot close a cycle.
type CircularImport struct{}
```

```go
func (CircularImport) MatchGraph(_ context.Context, t analyze.GraphTarget) []analyze.Finding {
	nodes := indexNodesByID(t.Nodes)
	files := indexFilesByPath(t.Nodes)
	adj := map[string][]string{}
	for _, e := range t.Edges {
		if e.Kind != model.EdgeImports {
			continue
		}
		src, dst := nodes[e.Src], nodes[e.Dst]
		if src == nil || dst == nil {
			continue // both endpoints must be ingested nodes
		}
		target := e.Dst
		if model.IsExternal(*dst) {
			id, ok := files.resolve(src.SourceURI, dst.Label)
			if !ok {
				continue
			}
			target = id
		}
		if target != e.Src {
			adj[e.Src] = append(adj[e.Src], target)
		}
	}
	// ... the findCycles loop and findings are unchanged ...
}
```

Replace `isExternal` (lines 73-78) with the resolver:

```go
// fileIndex maps each scanned file's slash path (its file:// URI minus the scheme) to its node ID.
type fileIndex map[string]string

func indexFilesByPath(ns []model.Node) fileIndex {
	idx := fileIndex{}
	for _, n := range ns {
		if p, ok := strings.CutPrefix(n.SourceURI, "file://"); ok && n.Kind == model.KindFile {
			idx[p] = n.ID
		}
	}
	return idx
}

// jsImportExts are the extensions, in resolution order, a JS/TS relative import may omit.
var jsImportExts = []string{".ts", ".tsx", ".js", ".jsx"}

// resolve maps an import spec written in the file fromURI to the scanned file it names. It handles relative
// imports only: JS/TS "./x" and "../x" (with an omitted extension or index file) and Python ".x" / "..x.y".
// Go has none, and the Go toolchain already rejects import cycles.
func (idx fileIndex) resolve(fromURI, spec string) (string, bool) {
	from, ok := strings.CutPrefix(fromURI, "file://")
	if !ok {
		return "", false
	}
	dir := path.Dir(from)
	var candidates []string
	switch ext := analyze.FileExt(from); {
	case slices.Contains([]string{"js", "jsx", "ts", "tsx"}, ext) && (strings.HasPrefix(spec, "./") || strings.HasPrefix(spec, "../")):
		base := path.Join(dir, spec)
		candidates = append(candidates, base)
		for _, e := range jsImportExts {
			candidates = append(candidates, base+e, base+"/index"+e)
		}
	case ext == "py" && strings.HasPrefix(spec, "."):
		rest := strings.TrimLeft(spec, ".")
		for range len(spec) - len(rest) - 1 { // one dot is this package; each further dot goes up one
			dir = path.Dir(dir)
		}
		if rest == "" {
			candidates = append(candidates, dir+"/__init__.py")
		} else {
			base := path.Join(dir, strings.ReplaceAll(rest, ".", "/"))
			candidates = append(candidates, base+".py", base+"/__init__.py")
		}
	}
	for _, c := range candidates {
		if id, ok := idx[c]; ok {
			return id, true
		}
	}
	return "", false
}
```

`internal/analyze/rules/architecture/dead_symbol.go:43`: `if isExternal(n) {` becomes `if model.IsExternal(n) {`.

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/analyze/... ./internal/graph/model/ -count=1; go vet ./internal/analyze/rules/architecture/ ./internal/graph/model/`
Expected: PASS

- [ ] **Step 5: Docs**

`CHANGELOG.md` → `### Fixed`, add:

```markdown
- **`architecture/circular-import` can fire again.** Every extracted import points at an external package
  node, which the rule skipped, so it never reported anything on a real graph. It now resolves JS/TS and
  Python relative imports back to the scanned file they name. Go is excluded, because `go build` already
  rejects import cycles.
```

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/model/node.go internal/analyze/rules/architecture/circular_import.go internal/analyze/rules/architecture/dead_symbol.go internal/analyze/rules/architecture/cycle_test.go CHANGELOG.md
git commit -m "fix(rules): resolve relative imports so circular-import detects real cycles"
```

### Task D5: Emit graph findings and report members in a fixed order

**Finding:** Three places depend on map iteration order:
- `GodModule` ranges over the `containsOut` map (`internal/analyze/rules/architecture/god_module.go:46`).
- `findCycles` starts Tarjan from a map range (`circular_import.go:136-140`). Each SCC's members are sorted, but the order of the SCCs is not.
- The report's per-community member list is built from a map range (`internal/report/report.go:58-61`) and sorted only by betweenness then `SourceURI` (`:71-76`). Every symbol in one file shares a `SourceURI`, and most tie at betweenness 0.

The finding order in `chiton analyze` output, in hook `BlockingDetails`, and in `GRAPH_REPORT.md` therefore changes between runs.
**Priority:** P3   **Effort:** S   **Depends on:** D4 (same file, `circular_import.go`)
**Behaviour change:** Output order becomes stable. Report members that tie on betweenness are now ordered by label, then source URI, then ID, the same order as the god-node table.

**Files:**
- Modify: `internal/analyze/rules/architecture/god_module.go:3-8`, `:45-47`
- Modify: `internal/analyze/rules/architecture/circular_import.go` (imports; `findCycles` doc and tail, original `:88-89`, `:136-141`)
- Modify: `internal/report/report.go:58-76`
- Test: `internal/analyze/rules/architecture/order_test.go`, `internal/report/report_test.go` (new)

**Interfaces:**
- Consumes: none
- Produces: none

- [ ] **Step 1: Write the failing tests**

`internal/analyze/rules/architecture/order_test.go`:

```go
package architecture

import (
	"context"
	"reflect"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// targets lists each finding's TargetNodeID in order.
func targets(fs []analyze.Finding) []string {
	out := make([]string, len(fs))
	for i, f := range fs {
		out[i] = f.TargetNodeID
	}
	return out
}

// TestGraphRulesReportInStableOrder runs each map-driven rule repeatedly; Go randomises map iteration, so
// any order leak shows up as a mismatch.
func TestGraphRulesReportInStableOrder(t *testing.T) {
	var god analyze.GraphTarget
	for _, f := range []string{"F5", "F3", "F9", "F1", "F7"} {
		god.Nodes = append(god.Nodes, file(f, f+".go"))
		for i := 0; i <= GodModuleSymbolThreshold; i++ {
			id := f + "s" + itoa(i)
			god.Nodes = append(god.Nodes, sym(id, "x", "func"))
			god.Edges = append(god.Edges, edge(f, id, model.EdgeContains))
		}
	}
	var cyc analyze.GraphTarget
	for _, p := range [][2]string{{"P", "Q"}, {"C", "D"}, {"X", "Y"}, {"A", "B"}, {"M", "N"}} {
		cyc.Nodes = append(cyc.Nodes, file(p[0], p[0]+".ts"), file(p[1], p[1]+".ts"))
		cyc.Edges = append(cyc.Edges, edge(p[0], p[1], model.EdgeImports), edge(p[1], p[0], model.EdgeImports))
	}
	cases := []struct {
		name string
		run  func() []analyze.Finding
		want []string
	}{
		{"god-module", func() []analyze.Finding { return GodModule{}.MatchGraph(context.Background(), god) },
			[]string{"F1", "F3", "F5", "F7", "F9"}},
		{"circular-import", func() []analyze.Finding { return CircularImport{}.MatchGraph(context.Background(), cyc) },
			[]string{"A", "C", "M", "P", "X"}},
	}
	for _, tc := range cases {
		for range 20 {
			if got := targets(tc.run()); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("%s order = %v, want %v", tc.name, got, tc.want)
			}
		}
	}
}
```

`internal/report/report_test.go`:

```go
package report

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
)

// TestGenerateIsDeterministic builds communities whose members tie on betweenness and share a source file,
// the case the old SourceURI-only tie-break left to map order, and checks every run renders the same report.
func TestGenerateIsDeterministic(t *testing.T) {
	res := &cluster.Results{GeneratedAt: time.Unix(0, 0).UTC(), NodeCount: 24, CommunityCount: 2, Nodes: map[string]cluster.NodeResult{}}
	for i := range 24 {
		res.Nodes[fmt.Sprint("id", i)] = cluster.NodeResult{
			Label: fmt.Sprint("sym", i%7), SourceURI: "file:///x.go", CommunityID: i % 2, Degree: 1,
		}
	}
	opts := Options{MaxGodNodes: 5, MaxPerCommunity: 12}
	var first bytes.Buffer
	if err := Generate(&first, res, nil, opts); err != nil {
		t.Fatal(err)
	}
	for range 20 {
		var again bytes.Buffer
		if err := Generate(&again, res, nil, opts); err != nil {
			t.Fatal(err)
		}
		if again.String() != first.String() {
			t.Fatalf("report changed between runs:\n%s\n---\n%s", first.String(), again.String())
		}
	}
	if !strings.Contains(first.String(), "### Community 0 (12 nodes)\n\n- `sym0` (file:///x.go)\n- `sym0` (file:///x.go)\n- `sym1`") {
		t.Errorf("community members are not ranked by label:\n%s", first.String())
	}
}
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `go test ./internal/analyze/rules/architecture/ ./internal/report/ -run "TestGraphRulesReportInStableOrder|TestGenerateIsDeterministic" -count=1`
Expected: FAIL — `god-module order = [F5 F3 F9 F1 F7], want [F1 F3 F5 F7 F9]` (any non-sorted permutation), and `report changed between runs:`.

- [ ] **Step 3: Implement**

`god_module.go`: add `"maps"` and `"slices"` to the imports and iterate sorted IDs:

```go
	var out []analyze.Finding
	for _, id := range slices.Sorted(maps.Keys(containsOut)) { // sorted, so findings come out in a fixed order
		count := containsOut[id]
		if count <= GodModuleSymbolThreshold {
			continue
		}
		// ... unchanged ...
```

`circular_import.go`: add `"maps"` to the imports (D4 already added `slices`). Update the `findCycles` doc and replace its tail:

```go
// findCycles uses Tarjan's algorithm to return the members of each strongly-connected component in adj
// with more than one node. Members and components are sorted by ID, so the output is stable across runs.
```

```go
	for _, v := range slices.Sorted(maps.Keys(adj)) {
		if _, ok := indices[v]; !ok {
			strongconnect(v)
		}
	}
	// Components are disjoint, so their smallest (first) IDs differ.
	slices.SortFunc(sccs, func(a, b []string) int { return strings.Compare(a[0], b[0]) })
	return sccs
}
```

`report.go`: build the per-community lists from `all`, which is already sorted by `CompareByBetweenness` then ID (lines 41-44), and delete the per-community `sort.Slice` (lines 71-76):

```go
	// all is already in god-node order, so each community's members come out ranked the same way.
	byComm := map[int][]cluster.NodeResult{}
	for _, e := range all {
		byComm[e.r.CommunityID] = append(byComm[e.r.CommunityID], e.r)
	}
	commIDs := make([]int, 0, len(byComm))
	for id := range byComm {
		commIDs = append(commIDs, id)
	}
	sort.Ints(commIDs)

	fmt.Fprintf(w, "## Communities\n\n")
	for _, cid := range commIDs {
		members := byComm[cid]
		fmt.Fprintf(w, "### Community %d (%d nodes)\n\n", cid, len(members))
		shown := min(opts.MaxPerCommunity, len(members))
		for _, m := range members[:shown] {
			fmt.Fprintf(w, "- `%s` (%s)\n", m.Label, m.SourceURI)
		}
		// ... unchanged ...
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/analyze/rules/architecture/ ./internal/report/ -count=1; go vet ./internal/analyze/rules/architecture/ ./internal/report/`
Expected: PASS

- [ ] **Step 5: Docs**

`CHANGELOG.md` → `### Fixed`, add:

```markdown
- **Graph-rule findings and report community listings no longer reorder between runs.** `god-module` and
  `circular-import` iterated maps, and the report broke betweenness ties only by source URI, which every
  symbol in a file shares.
```

- [ ] **Step 6: Commit**

```powershell
git add internal/analyze/rules/architecture/god_module.go internal/analyze/rules/architecture/circular_import.go internal/analyze/rules/architecture/order_test.go internal/report/report.go internal/report/report_test.go CHANGELOG.md
git commit -m "fix(analyze,report): deterministic finding and community-member order"
```

### Task D6: Re-cluster only the touched region, read through the store

**Finding:** Every incremental re-cluster loads the whole graph: `recluster` calls `AllNodes` and `AllEdges` (`internal/ingest/engine.go:410-417`). `RunIncremental` then builds a map-of-maps adjacency for all of it (`internal/graph/cluster/incremental.go:25-29`, `:100-133`). The seeds are all of a file's extracted nodes, including shared `pkg://` stubs (`engine.go:168-171`), so the two-hop expansion (`incremental.go:32`) runs through hubs like `fmt` into much of the graph. `markSurvivorsTouched` does the same through `Neighbors(fileID, 2)` (`:464-475`). The sidecar is written as indented JSON (`sidecar.go:47`). Reproduced on a scratch GOROOT graph (118,290 nodes): a one-file edit took 1.8 to 2.0 s with `cluster.json` present, against the audit's 0.34 s without it. With this task the edit takes 0.55 s, and `cluster.json` goes from 28.8 MB to 22.9 MB.
**Priority:** P3   **Effort:** L   **Depends on:** D1 (prior findings), D4 (`model.IsExternal`)
**Behaviour change:** Only the region around changed files is re-clustered.
- The region is the nodes within 2 hops of what a change touched, read through `GraphStore.Neighbors`. External stubs are leaves, and those already in `cluster.json` keep their community and degree until the next full `chiton cluster`.
- Edges between two nodes both at the hop limit are not read.
- A new node no touched node reaches, such as a rule node with no findings yet, appears at the next full pass. Previously it got a singleton community.
- The sidecar is compact JSON. Older indented files still load.

**Files:**
- Modify: `internal/graph/cluster/incremental.go:1-205` (rewrite RunIncremental; add RegionDepth, Region, NeighborFunc, BuildRegion, ApplyRegion, compactResults, memNeighbors; delete buildAdjacency `:98-134`, neighborhood `:136-165`, compactByNodeOrder `:185-205`; keep Full, induce, buildResults)
- Modify: `internal/graph/cluster/incremental_test.go:178-193` (delete `TestNeighborhood`; `TestBuildRegion` replaces it)
- Modify: `internal/graph/cluster/sidecar.go:45-72`
- Modify: `internal/analyze/incremental.go` (add `PriorFindings.IDs`)
- Modify: `internal/ingest/engine.go:5-27`, `:73-105`, `:195`, `:387-427`, `:462-476`, `:509-536`
- Test: `internal/graph/cluster/region_test.go`, `internal/ingest/recluster_test.go` (new)

**Interfaces:**
- Consumes: D1 `(*analyze.PriorFindings)`; D4 `model.IsExternal(n model.Node) bool`
- Produces: `const cluster.RegionDepth = 2`; `type cluster.Region struct{ Nodes []model.Node; Edges []model.Edge; Degree map[string]int; Removed map[string]bool }`; `type cluster.NeighborFunc func(id string) (self *model.Node, edges []model.Edge, nbrs []model.Node, err error)`; `func cluster.BuildRegion(seeds []string, depth int, nb NeighborFunc) (Region, error)`; `func cluster.ApplyRegion(prev *Results, r Region, resolution float64, rng *rand.Rand) *Results`; `func (p *analyze.PriorFindings) IDs() []string`. **Sidecar encoding seam for Section E:** `func marshalSidecar(r *Results) ([]byte, error)` and `func unmarshalSidecar(data []byte) (*Results, error)` in `internal/graph/cluster/sidecar.go`. `WriteSidecar` and `LoadSidecar` are the only callers, so E wraps the bytes between `marshalSidecar` and `fsutil.WriteFileAtomic`, and between `os.ReadFile` and `unmarshalSidecar`.

- [ ] **Step 1: Write the failing tests**

`internal/graph/cluster/region_test.go`:

```go
package cluster

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// stub builds an external package node, as the extractors emit for an import.
func stub(id string) model.Node {
	n := makeNode(id, id)
	n.Attrs = map[string]any{"external": true}
	return n
}

// TestBuildRegion checks the hop limit, that an external stub is a leaf, and that an absent seed is Removed.
func TestBuildRegion(t *testing.T) {
	// Path a-b-c-d, plus a imports hub P, which 50 other files also import.
	nodes := []model.Node{makeNode("a", "a"), makeNode("b", "b"), makeNode("c", "c"), makeNode("d", "d"), stub("P")}
	edges := []model.Edge{makeEdge("a", "b"), makeEdge("b", "c"), makeEdge("c", "d"), makeEdge("a", "P")}
	for i := range 50 {
		id := fmt.Sprint("f", i)
		nodes = append(nodes, makeNode(id, id))
		edges = append(edges, makeEdge(id, "P"))
	}
	r, err := BuildRegion([]string{"a", "gone"}, 2, memNeighbors(nodes, edges))
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, n := range r.Nodes {
		got = append(got, n.ID)
	}
	if want := []string{"a", "b", "P", "c"}; !reflect.DeepEqual(got, want) {
		t.Errorf("region nodes = %v, want %v (no importer reached through P, d beyond 2 hops)", got, want)
	}
	if !r.Removed["gone"] || len(r.Removed) != 1 {
		t.Errorf("Removed = %v, want only the absent seed", r.Removed)
	}
	if r.Degree["a"] != 2 || r.Degree["b"] != 2 {
		t.Errorf("Degree = %v, want a:2 b:2 for the expanded nodes", r.Degree)
	}
	if _, ok := r.Degree["P"]; ok {
		t.Error("stub P is a leaf; its adjacency must not be read")
	}
	if len(r.Edges) != 3 { // a-b, a-P, b-c
		t.Errorf("region edges = %d, want 3", len(r.Edges))
	}
}

// TestApplyRegionPinsExistingStub checks a hub stub already in prev keeps its community (and its partners
// outside the region keep theirs) while the region is re-clustered and removed nodes drop out.
func TestApplyRegionPinsExistingStub(t *testing.T) {
	nodes, edges := twoCliques()
	nodes = append(nodes, stub("P"), makeNode("X", "X"))
	edges = append(edges, makeEdge("D", "P"), makeEdge("E", "P"), makeEdge("A", "P"))
	prev := Full(nodes, edges, 1.0, rand.New(rand.NewSource(1)))
	if !sameComm(prev, "P", "D") {
		t.Fatalf("setup: P should start with clique 2, got %+v", prev.Nodes)
	}

	r, err := BuildRegion([]string{"A", "X"}, 2, memNeighbors(nodes[:len(nodes)-1], edges)) // X was deleted
	if err != nil {
		t.Fatal(err)
	}
	res := ApplyRegion(prev, r, 1.0, rand.New(rand.NewSource(2)))
	if !sameComm(res, "P", "D") || !sameComm(res, "D", "E") {
		t.Error("pinned stub P and its untouched partners should keep one community")
	}
	if !sameComm(res, "A", "B") || sameComm(res, "A", "D") {
		t.Error("clique 1 should be re-clustered together and apart from clique 2")
	}
	if _, ok := res.Nodes["X"]; ok || res.NodeCount != len(nodes)-1 {
		t.Errorf("removed X should drop out: NodeCount=%d, X present=%v", res.NodeCount, ok)
	}
	validatePartition(t, res, nodes[:len(nodes)-1])
}

// TestSidecarIsCompactAndRoundTrips checks WriteSidecar emits unindented JSON that LoadSidecar reads back.
func TestSidecarIsCompactAndRoundTrips(t *testing.T) {
	nodes, edges := twoCliques()
	want := Full(nodes, edges, 1.0, nil)
	path := filepath.Join(t.TempDir(), "cluster.json")
	if err := WriteSidecar(path, want); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte("\n ")) {
		t.Errorf("sidecar is indented: %.80s", raw)
	}
	got, err := LoadSidecar(path)
	if err != nil {
		t.Fatal(err)
	}
	if !got.GeneratedAt.Equal(want.GeneratedAt) || !reflect.DeepEqual(got.Nodes, want.Nodes) || got.CommunityCount != want.CommunityCount {
		t.Errorf("round trip changed the results:\n got %+v\nwant %+v", got, want)
	}
}
```

`internal/ingest/recluster_test.go`:

```go
package ingest

import (
	"context"
	"errors"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

// noBulkStore fails the whole-graph reads, so a test proves a re-cluster never makes them.
type noBulkStore struct{ *badgerstore.Store }

func (noBulkStore) AllNodes(context.Context) ([]model.Node, error) {
	return nil, errors.New("AllNodes called")
}

func (noBulkStore) AllEdges(context.Context) ([]model.Edge, error) {
	return nil, errors.New("AllEdges called")
}

// seedSidecar writes a full cluster pass over the harness store, as `chiton cluster` would.
func seedSidecar(t *testing.T, h *harness) {
	t.Helper()
	ctx := context.Background()
	nodes, _ := h.gs.AllNodes(ctx)
	edges, _ := h.gs.AllEdges(ctx)
	if err := cluster.WriteSidecar(cluster.SidecarPath(h.eng.cfg.GraphPath), cluster.Full(nodes, edges, 1.0, nil)); err != nil {
		t.Fatal(err)
	}
}

// TestReclusterReadsOnlyRegion checks an edit re-clusters from the store's neighbour reads alone, drops the
// replaced symbol and the pruned finding, and adds the new symbol.
func TestReclusterReadsOnlyRegion(t *testing.T) {
	h := newHarness(t, false, true)
	ctx := context.Background()
	p := h.write(t, "a.go", "package src\n\nimport \"fmt\"\n\n// TODO: tidy\nfunc Old() { fmt.Println() }\n")
	h.write(t, "b.go", "package src\n\nimport \"fmt\"\n\nfunc B() { fmt.Println() }\n")
	if _, err := h.eng.SyncDir(ctx, h.dir, true); err != nil {
		t.Fatal(err)
	}
	seedSidecar(t, h)
	views, _ := analyze.List(ctx, h.gs, analyze.ListOptions{Scope: model.FileURI(p)})
	if len(views) == 0 {
		t.Fatal("setup: a.go should have a TODO finding")
	}
	todoID := views[0].ID

	eng := New(noBulkStore{h.gs}, h.cache, h.eng.cfg, nil, analyze.New(h.gs, nil), func(string, ...any) {})
	h.write(t, "a.go", "package src\n\nimport \"fmt\"\n\nfunc New() { fmt.Println() }\n")
	sum, err := eng.SyncChanges(ctx, []Change{{Path: p, Op: OpUpsert}})
	if err != nil || !sum.Reclustered {
		t.Fatalf("SyncChanges = %+v, %v; want a re-cluster without whole-graph reads", sum, err)
	}
	res, err := cluster.LoadSidecar(cluster.SidecarPath(h.eng.cfg.GraphPath))
	if err != nil {
		t.Fatal(err)
	}
	uri := model.FileURI(p)
	if _, ok := res.Nodes[model.NewNodeID(uri, "Old")]; ok {
		t.Error("replaced symbol Old is still in the sidecar")
	}
	if _, ok := res.Nodes[todoID]; ok {
		t.Error("pruned TODO finding is still in the sidecar")
	}
	if _, ok := res.Nodes[model.NewNodeID(uri, "New")]; !ok {
		t.Error("new symbol New is missing from the sidecar")
	}
	if res.NodeCount != len(res.Nodes) {
		t.Errorf("NodeCount %d != len(Nodes) %d", res.NodeCount, len(res.Nodes))
	}
}
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `go test ./internal/graph/cluster/ ./internal/ingest/ -run "TestBuildRegion|TestApplyRegion|TestSidecarIsCompact|TestReclusterReadsOnlyRegion" -count=1`
Expected: FAIL. cluster: build failed with `undefined: BuildRegion` and `undefined: memNeighbors`. ingest: `SyncChanges = &{... Reclustered:false}, <nil>; want a re-cluster without whole-graph reads`, because the re-cluster died on `AllNodes called`.

- [ ] **Step 3: Implement**

`internal/graph/cluster/incremental.go`: the imports become `maps, math/rand, slices, time`, `metrics` and `model`. Delete `buildAdjacency`, `neighborhood` and `compactByNodeOrder`. Keep `Full`, `induce` and `buildResults` as they are. Replace `RunIncremental` and add the region API:

```go
// RegionDepth is how many hops around the touched nodes an incremental pass re-clusters.
const RegionDepth = 2

// Region is the part of the graph an incremental pass re-clusters.
type Region struct {
	Nodes   []model.Node    // nodes to re-cluster, in first-reached order
	Edges   []model.Edge    // edges with both ends in Nodes
	Degree  map[string]int  // whole-graph degree of each node whose adjacency was read; others keep prev's
	Removed map[string]bool // seeds no longer in the graph, dropped from the results
}

// NeighborFunc returns id's node (nil when absent), its incident edges, and its distinct neighbours other
// than itself. ingest backs it with GraphStore.Neighbors, so a re-cluster reads only the region.
type NeighborFunc func(id string) (self *model.Node, edges []model.Edge, nbrs []model.Node, err error)

// BuildRegion collects the nodes within depth hops of seeds and the edges among them. Only nodes closer than
// depth are expanded, so edges between two nodes at the hop limit are not read. External package stubs are
// leaves, which keeps a hub like pkg://fmt from pulling in every importer. Absent seeds land in Removed.
func BuildRegion(seeds []string, depth int, nb NeighborFunc) (Region, error) {
	r := Region{Degree: map[string]int{}, Removed: map[string]bool{}}
	in := map[string]bool{}
	var level []string
	for _, id := range seeds {
		if !in[id] {
			in[id] = true
			level = append(level, id)
		}
	}
	seenEdge := map[string]bool{}
	var edges []model.Edge
	for d := 0; len(level) > 0; d++ {
		var next []string
		for _, id := range level {
			self, inc, nbrs, err := nb(id)
			if err != nil {
				return Region{}, err
			}
			if self == nil { // only a seed can be absent; deeper nodes came from a neighbour list
				delete(in, id)
				r.Removed[id] = true
				continue
			}
			if d == 0 {
				r.Nodes = append(r.Nodes, *self)
			}
			if d >= depth || model.IsExternal(*self) {
				continue
			}
			r.Degree[id] = len(nbrs)
			for _, e := range inc {
				if !seenEdge[e.ID] {
					seenEdge[e.ID] = true
					edges = append(edges, e)
				}
			}
			for _, n := range nbrs {
				if in[n.ID] {
					continue
				}
				in[n.ID] = true
				r.Nodes = append(r.Nodes, n)
				if d+1 < depth && !model.IsExternal(n) {
					next = append(next, n.ID)
				}
			}
		}
		level = next
	}
	for _, e := range edges {
		if e.Src != e.Dst && in[e.Src] && in[e.Dst] {
			r.Edges = append(r.Edges, e)
		}
	}
	return r, nil
}

// ApplyRegion re-clusters r and merges it into prev, keeping every other node's prev result. External stubs
// already in prev keep their community, since the region sees only a few of a hub's edges.
func ApplyRegion(prev *Results, r Region, resolution float64, rng *rand.Rand) *Results {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}
	var sub []model.Node
	keep := make(map[string]bool, len(r.Nodes))
	for _, n := range r.Nodes {
		if _, had := prev.Nodes[n.ID]; had && model.IsExternal(n) {
			continue
		}
		sub = append(sub, n)
		keep[n.ID] = true
	}
	_, subEdges := induce(sub, r.Edges, keep)
	local := Run(sub, subEdges, resolution, rng)

	out := make(map[string]NodeResult, len(prev.Nodes)+len(sub))
	base := 0
	for id, pr := range prev.Nodes {
		if r.Removed[id] || keep[id] {
			continue
		}
		out[id] = pr
		base = max(base, pr.CommunityID+1)
	}
	for _, n := range r.Nodes {
		pr, had := prev.Nodes[n.ID]
		nr := NodeResult{Label: n.Label, SourceURI: n.SourceURI, CommunityID: pr.CommunityID,
			Betweenness: pr.Betweenness, Degree: pr.Degree}
		if c, ok := local[n.ID]; ok {
			nr.CommunityID = base + c
		}
		if d, ok := r.Degree[n.ID]; ok || !had {
			nr.Degree = d
		}
		out[n.ID] = nr
	}
	return compactResults(out)
}

// compactResults renumbers the communities in nodes to 0..k-1, keeping their relative order.
func compactResults(nodes map[string]NodeResult) *Results {
	used := map[int]bool{}
	for _, nr := range nodes {
		used[nr.CommunityID] = true
	}
	ids := slices.Sorted(maps.Keys(used))
	remap := make(map[int]int, len(ids))
	for i, c := range ids {
		remap[c] = i
	}
	for id, nr := range nodes {
		nr.CommunityID = remap[nr.CommunityID]
		nodes[id] = nr
	}
	return &Results{GeneratedAt: time.Now(), NodeCount: len(nodes), CommunityCount: len(ids), Nodes: nodes}
}

// RunIncremental is ApplyRegion over an in-memory graph: it re-clusters the RegionDepth-hop region around
// touched plus any node new since prev, and drops prev nodes absent from nodes. A nil or empty prev runs a
// full pass. Betweenness is global, so it comes from prev (new nodes get 0).
func RunIncremental(prev *Results, nodes []model.Node, edges []model.Edge, touched map[string]bool, resolution float64, rng *rand.Rand) *Results {
	if len(nodes) == 0 {
		return &Results{GeneratedAt: time.Now(), Nodes: map[string]NodeResult{}}
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}
	// On a cold start or a lost sidecar, do the same full pass as `chiton cluster`.
	if prev == nil || len(prev.Nodes) == 0 {
		return Full(nodes, edges, resolution, rng)
	}

	var seeds []string
	for _, n := range nodes {
		if _, had := prev.Nodes[n.ID]; touched[n.ID] || !had {
			seeds = append(seeds, n.ID)
		}
	}
	r, _ := BuildRegion(seeds, RegionDepth, memNeighbors(nodes, edges)) // memNeighbors never fails
	present := make(map[string]bool, len(nodes))
	for _, n := range nodes {
		present[n.ID] = true
	}
	for id := range prev.Nodes {
		if !present[id] {
			r.Removed[id] = true
		}
	}
	return ApplyRegion(prev, r, resolution, rng)
}

// memNeighbors is a NeighborFunc over an in-memory graph, ignoring edges to IDs not in nodes.
func memNeighbors(nodes []model.Node, edges []model.Edge) NeighborFunc {
	byID := make(map[string]*model.Node, len(nodes))
	for i := range nodes {
		byID[nodes[i].ID] = &nodes[i]
	}
	incident := map[string][]model.Edge{}
	for _, e := range edges {
		if e.Src == e.Dst || byID[e.Src] == nil || byID[e.Dst] == nil {
			continue
		}
		incident[e.Src] = append(incident[e.Src], e)
		incident[e.Dst] = append(incident[e.Dst], e)
	}
	return func(id string) (*model.Node, []model.Edge, []model.Node, error) {
		self := byID[id]
		if self == nil {
			return nil, nil, nil, nil
		}
		seen := map[string]bool{}
		var nbrs []model.Node
		for _, e := range incident[id] {
			other := e.Dst
			if other == id {
				other = e.Src
			}
			if !seen[other] {
				seen[other] = true
				nbrs = append(nbrs, *byID[other])
			}
		}
		return self, incident[id], nbrs, nil
	}
}
```

Delete `TestNeighborhood` (`incremental_test.go:178-193`). `TestBuildRegion` covers the hop limit.

`internal/graph/cluster/sidecar.go`: replace `WriteSidecar` and `LoadSidecar` and add the encoding seam:

```go
// WriteSidecar atomically writes r to path as compact JSON, so a concurrent reader never sees a partial file.
func WriteSidecar(path string, r *Results) error {
	b, err := marshalSidecar(r)
	if err != nil {
		return err
	}
	if err := fsutil.WriteFileAtomic(path, b, 0o644); err != nil {
		return fmt.Errorf("write cluster sidecar %s: %w", path, err)
	}
	return nil
}

// LoadSidecar reads the cluster sidecar at path. Returns (nil, nil) when the
// file does not exist so callers can treat a missing sidecar as "not yet run".
func LoadSidecar(path string) (*Results, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read cluster sidecar %s: %w", path, err)
	}
	r, err := unmarshalSidecar(data)
	if err != nil {
		return nil, fmt.Errorf("parse cluster sidecar %s: %w", path, err)
	}
	return r, nil
}

// marshalSidecar is the sidecar's one encoding seam: compact JSON, which reads back older indented files too.
func marshalSidecar(r *Results) ([]byte, error) {
	b, err := json.Marshal(r)
	if err != nil {
		return nil, fmt.Errorf("marshal cluster results: %w", err)
	}
	return b, nil
}

// unmarshalSidecar decodes marshalSidecar's output.
func unmarshalSidecar(data []byte) (*Results, error) {
	var r Results
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, err
	}
	return &r, nil
}
```

`internal/analyze/incremental.go`, after the `PriorFindings` type from D1:

```go
// IDs returns the prior findings' node IDs; a nil p has none.
func (p *PriorFindings) IDs() []string {
	if p == nil {
		return nil
	}
	ids := make([]string, len(p.nodes))
	for i, n := range p.nodes {
		ids[i] = n.ID
	}
	return ids
}
```

`internal/ingest/engine.go`: add `"maps"` and `"slices"` to the imports. Change the touched-state fields and `New`:

```go
	touchedMu   sync.Mutex
	touched     map[string]bool // re-cluster seed IDs
	touchedURIs map[string]bool // files re-synced or deleted since the last re-cluster
	dirty       bool
}
```

```go
	return &Engine{
		store:       s,
		cache:       c,
		embedder:    embedder,
		analyzer:    analyzer,
		cfg:         cfg.withDefaults(),
		log:         log,
		touched:     map[string]bool{},
		touchedURIs: map[string]bool{},
	}
```

In `SyncFile`, replace `e.markTouched(ids)` (line 195) with:

```go
	// Prior findings are seeds too: the pruned ones drop out of the sidecar, the rest are re-clustered.
	e.markTouched(append(clusterSeeds(nodes), prior.IDs()...), uri)
	return out, nil
}

// clusterSeeds returns the IDs a re-cluster starts from. External package stubs are shared hubs (pkg://fmt
// has thousands of importers), so they are never seeds.
func clusterSeeds(nodes []model.Node) []string {
	ids := make([]string, 0, len(nodes))
	for _, n := range nodes {
		if !model.IsExternal(n) {
			ids = append(ids, n.ID)
		}
	}
	return ids
}
```

Replace `recluster`'s body after the sidecar load (from line 410 to the end of the function) and add the store-backed neighbour reader:

```go
	sidecarPath := cluster.SidecarPath(e.cfg.GraphPath)
	touched, uris, dirty := e.takeTouched()
	if !dirty {
		_, statErr := os.Stat(sidecarPath)
		return statErr == nil, nil
	}

	prev, err := cluster.LoadSidecar(sidecarPath)
	if err != nil {
		return false, fmt.Errorf("load cluster sidecar: %w", err)
	}
	if prev == nil {
		// With no prior full clustering, don't surprise the user with a full pass during sync.
		e.log("sync: skipping re-cluster (no cluster.json yet — run 'chiton cluster' once)")
		return false, nil
	}

	// A re-synced or deleted file's old nodes are seeds too, so any it no longer has drop out.
	for id, nr := range prev.Nodes {
		if uris[nr.SourceURI] {
			touched[id] = true
		}
	}
	region, err := cluster.BuildRegion(slices.Sorted(maps.Keys(touched)), cluster.RegionDepth, storeNeighbors(ctx, e.store))
	if err != nil {
		return false, fmt.Errorf("load re-cluster region: %w", err)
	}

	rng := rand.New(rand.NewSource(42))
	res := cluster.ApplyRegion(prev, region, e.cfg.Resolution, rng)
	if err := cluster.WriteSidecar(sidecarPath, res); err != nil {
		return false, err
	}
	e.log("sync: re-clustered %d node(s) around %d touched → %d communities over %d nodes",
		len(region.Nodes), len(touched), res.CommunityCount, res.NodeCount)
	return true, nil
}

// storeNeighbors backs cluster.BuildRegion with one gs.Neighbors read per expanded node.
func storeNeighbors(ctx context.Context, gs store.GraphStore) cluster.NeighborFunc {
	return func(id string) (*model.Node, []model.Edge, []model.Node, error) {
		sub, err := gs.Neighbors(ctx, id, 1, nil)
		if err != nil {
			return nil, nil, nil, err
		}
		if len(sub.Nodes) == 0 || sub.Nodes[0].ID != id {
			return nil, nil, nil, nil // Neighbors lists the node itself first only when it exists
		}
		return &sub.Nodes[0], sub.Edges, sub.Nodes[1:], nil
	}
}
```

Replace `markSurvivorsTouched`, `markTouched`, `markDirty` (deleted: it has no remaining caller) and `takeTouched`:

```go
// markSurvivorsTouched marks a file's nodes and their neighbors within 2 hops touched (external stubs are
// leaves, as in a re-cluster), and always marks uri and sets dirty.
func (e *Engine) markSurvivorsTouched(ctx context.Context, uri string) {
	r, err := cluster.BuildRegion([]string{model.NewNodeID(uri, uri)}, cluster.RegionDepth, storeNeighbors(ctx, e.store))
	if err != nil {
		e.markTouched(nil, uri)
		return
	}
	e.markTouched(clusterSeeds(r.Nodes), uri)
}
```

```go
// markTouched records re-cluster seeds and the file URI whose nodes changed, and sets dirty.
func (e *Engine) markTouched(ids []string, uri string) {
	e.touchedMu.Lock()
	defer e.touchedMu.Unlock()
	for _, id := range ids {
		e.touched[id] = true
	}
	e.touchedURIs[uri] = true
	e.dirty = true
}

// takeTouched atomically returns and resets the accumulated seeds, file URIs, and dirty flag.
func (e *Engine) takeTouched() (ids, uris map[string]bool, dirty bool) {
	e.touchedMu.Lock()
	defer e.touchedMu.Unlock()
	if !e.dirty {
		return nil, nil, false
	}
	ids, uris = e.touched, e.touchedURIs
	e.touched, e.touchedURIs = map[string]bool{}, map[string]bool{}
	e.dirty = false
	return ids, uris, true
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/graph/... ./internal/ingest/... ./internal/analyze/... ./internal/mcp/... -count=1; go vet ./internal/graph/cluster/ ./internal/ingest/`
Expected: PASS. The existing incremental tests (`TestIncrementalColdStart`, `…FreezesUntouched`, `…Deletion`, `…NewNode`), `TestIncrementalReclusterUpdatesSidecar` and `TestSyncReportsReclustered` all hold.

- [ ] **Step 5: Docs**

`README.md:327-329`: replace the two sentences starting "After a batch settles" with:

```markdown
After a batch settles, the 2-hop neighborhood of the changed files is
re-clustered, read straight from the store (shared package stubs are not
expanded); everything else keeps its community. Betweenness is a global metric
carried forward between full `chiton cluster` passes, which also pick up nodes
no change reached.
```

`CLAUDE.md:189`: change "the touched 2-hop neighborhood is incrementally re-clustered (run" to "the touched 2-hop neighborhood is incrementally re-clustered from store reads (run". Leave the rest of the sentence as it is.

`CHANGELOG.md` → `### Changed`, add:

```markdown
- **Incremental re-clustering reads only the touched region.** `sync` used to load every node and edge and
  expand two hops through shared package stubs such as `pkg://fmt` on each re-cluster; it now reads the
  region through the store's neighbour index and treats stubs as leaves. A one-file edit on a 118k-node
  graph went from ~1.9 s to ~0.55 s. `cluster.json` is now compact JSON (about 20% smaller); older indented
  files still load.
```

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/cluster/incremental.go internal/graph/cluster/incremental_test.go internal/graph/cluster/region_test.go internal/graph/cluster/sidecar.go internal/analyze/incremental.go internal/ingest/engine.go internal/ingest/recluster_test.go README.md CLAUDE.md CHANGELOG.md
git commit -m "perf(sync): re-cluster the touched region via store reads; compact cluster sidecar"
```

### Task D7: Parallelise Brandes betweenness across pivots

**Finding:** `betweenness` runs every Brandes source serially through a single buffer set (`internal/graph/metrics/metrics.go:96-104`). Above 5,000 nodes it samples 500 pivots, each a full BFS over the graph. On the 118k-node scratch graph, a full `chiton cluster` computes in 4.3 s, and betweenness dominates. The pivots are chosen deterministically already (`:48`, seed 42).
**Priority:** P3   **Effort:** S   **Depends on:** none
**Behaviour change:** none. The scores equal the serial ones within float rounding (≤1e-9), and for a given `GOMAXPROCS` they are identical across runs. Full cluster compute drops from 4.3 s to 0.95 s on this machine.

**Files:**
- Modify: `internal/graph/metrics/metrics.go:5-9` (imports), `:94-104`
- Test: `internal/graph/metrics/parallel_test.go` (new)

**Interfaces:**
- Consumes: none
- Produces: none (unexported `betweennessWorkers`)

- [ ] **Step 1: Write the failing test**

`internal/graph/metrics/parallel_test.go`:

```go
package metrics

import (
	"math"
	"math/rand"
	"testing"
)

// randomGraph returns a seeded n-node graph: a ring (so it is connected) plus extra random chords.
func randomGraph(n, chords int, seed int64) *intGraph {
	rng := rand.New(rand.NewSource(seed))
	adj := make([][]int, n)
	seen := map[[2]int]bool{}
	link := func(u, v int) {
		if u == v {
			return
		}
		key := [2]int{min(u, v), max(u, v)}
		if seen[key] {
			return
		}
		seen[key] = true
		adj[u] = append(adj[u], v)
		adj[v] = append(adj[v], u)
	}
	for i := range n {
		link(i, (i+1)%n)
	}
	for range chords {
		link(rng.Intn(n), rng.Intn(n))
	}
	return &intGraph{n: n, adj: adj}
}

// TestBetweennessParallelMatchesSerial checks splitting sources across workers changes no score beyond float
// rounding, in exact (every source) and sampled (scaled subset) mode, and that a worker count is repeatable.
func TestBetweennessParallelMatchesSerial(t *testing.T) {
	g := randomGraph(600, 1500, 7)
	all := make([]int, g.n)
	for i := range all {
		all[i] = i
	}
	sampled := rand.New(rand.NewSource(42)).Perm(g.n)[:50]
	cases := []struct {
		name    string
		sources []int
		scale   float64
	}{
		{"exact", all, 1},
		{"sampled", sampled, float64(g.n) / 50},
	}
	for _, tc := range cases {
		want := betweennessWorkers(g, tc.sources, tc.scale, 1)
		for _, workers := range []int{2, 3, 8, 64} {
			got := betweennessWorkers(g, tc.sources, tc.scale, workers)
			for i := range want {
				if math.Abs(got[i]-want[i]) > 1e-9 {
					t.Fatalf("%s, %d workers: node %d = %.15f, serial %.15f", tc.name, workers, i, got[i], want[i])
				}
			}
			again := betweennessWorkers(g, tc.sources, tc.scale, workers)
			for i := range got {
				if again[i] != got[i] {
					t.Fatalf("%s, %d workers: node %d differs between runs (%v vs %v)", tc.name, workers, i, got[i], again[i])
				}
			}
		}
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/graph/metrics/ -run TestBetweennessParallelMatchesSerial -count=1`
Expected: FAIL — build failed: `undefined: betweennessWorkers`.

- [ ] **Step 3: Implement**

`internal/graph/metrics/metrics.go`: add `"runtime"` and `"sync"` to the imports and replace `betweenness`:

```go
// maxWorkers caps betweenness's goroutines, since each holds five n-length buffers.
const maxWorkers = 8

// betweenness runs Brandes from each source on up to maxWorkers goroutines, scaling every contribution by
// scale, and normalizes the sums. It runs in O(len(sources)·m) total work.
func betweenness(g *intGraph, sources []int, scale float64) []float64 {
	return betweennessWorkers(g, sources, scale, min(runtime.GOMAXPROCS(0), maxWorkers))
}

// betweennessWorkers gives each worker a contiguous run of sources, its own buffers, and its own partial
// sums, then adds the partials in worker order, so a given worker count always yields the same floats.
func betweennessWorkers(g *intGraph, sources []int, scale float64, workers int) []float64 {
	workers = max(1, min(workers, len(sources)))
	chunk := (len(sources) + workers - 1) / workers
	partial := make([][]float64, workers)
	var wg sync.WaitGroup
	for w := range workers {
		partial[w] = make([]float64, g.n)
		lo, hi := min(w*chunk, len(sources)), min((w+1)*chunk, len(sources))
		wg.Go(func() {
			b := newBrandes(g)
			for _, s := range sources[lo:hi] {
				b.accumulate(s, partial[w], scale)
			}
		})
	}
	wg.Wait()
	bet := partial[0]
	for _, p := range partial[1:] {
		for i, v := range p {
			bet[i] += v
		}
	}
	normalize(bet, g.n)
	return bet
}
```

At 118k nodes and 8 workers the extra memory is about 38 MB of buffers. `sync.WaitGroup.Go` needs Go 1.25, which the module already requires (it is used in `internal/fsutil/fsutil_test.go:68`).

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/graph/... -count=1; go vet ./internal/graph/metrics/`
Expected: PASS. CI's Linux leg can also run `go test -race ./internal/graph/metrics/`. `-race` needs cgo, which the Windows toolchain here lacks.

- [ ] **Step 5: Docs**

`CHANGELOG.md` → `### Changed`, add:

```markdown
- **`chiton cluster` computes betweenness on up to 8 cores.** Brandes pivots are split across workers with
  private buffers and summed in a fixed order; a full pass over a 118k-node graph dropped from 4.3 s to
  0.95 s of compute.
```

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/metrics/metrics.go internal/graph/metrics/parallel_test.go CHANGELOG.md
git commit -m "perf(metrics): parallelise Brandes betweenness across pivots"
```

---

## Section E: Configuration, encryption and first-run UX

This section fixes how chiton finds its configuration and per-project state, how it guards the embedding endpoint, and how the encrypted store behaves on first run. E1 and E2 come first because later tasks build on them. E1 makes a bad `.chitonrc` an error, and E2 gives `.chiton/` one resolver anchored at `CHITON_PROJECT_DIR`. E3 extends E1's `LoadFrom`. E4, E5 and E6 reshape `openGraphStore` in that order, and each shows the whole function as it stands after the task. E6 also needs Task D6 (sidecar serialization seam) merged first. E7 and E8 are independent and can land at any point.

### Task E1: Fail on a malformed `.chitonrc` instead of ignoring it

**Finding:** `internal/config/config.go:86-87` discards every `ReadInConfig` error (`_ = v.ReadInConfig()`). A `.chitonrc` that exists but fails to parse is ignored, so `graph_path` falls back to `./chiton-out/graph.kuzu` and the command silently opens a different store than the file names. The test helper `isolate` (`internal/config/config_test.go:11-18`) also leaks the developer's real home config on Windows, because viper resolves `$HOME` from `HOMEDRIVE`+`HOMEPATH` before `USERPROFILE` (viper v1.21.0 `util.go:131-140`).
**Priority:** P2   **Effort:** S   **Depends on:** none
**Behaviour change:** A `.chitonrc` (project or `$HOME`) that exists but cannot be read or parsed now stops every command with `read config <path>: ...`. A missing file is still fine.

**Files:**
- Modify: `internal/config/config.go:3-9` (imports), `internal/config/config.go:86-87`
- Test: `internal/config/config_test.go:3-18` (imports, `isolate`), new test appended

**Interfaces:**
- Consumes: none
- Produces: `isolate(t *testing.T) string` (test helper now returns the scratch home; E3 uses it)

- [ ] **Step 1: Write the failing test**

Replace `isolate` in `internal/config/config_test.go:9-18` and add `"strings"` to the imports:

```go
// isolate points HOME at a scratch directory so a developer's real ~/.chitonrc can never leak into these tests.
// Viper reads HOMEDRIVE+HOMEPATH before USERPROFILE on Windows, so those are cleared too. It returns the home.
func isolate(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("HOMEDRIVE", "")
	t.Setenv("HOMEPATH", "")
	t.Setenv(ProjectDirEnv, "")
	t.Setenv("CHITON_GRAPH_PATH", "")
	return home
}
```

Append:

```go
// A .chitonrc that fails to parse must stop the command, because falling back to defaults silently opens a
// different store than the one the file names.
func TestMalformedChitonrcIsAnError(t *testing.T) {
	isolate(t)
	project := t.TempDir()
	rc := filepath.Join(project, ".chitonrc")
	if err := os.WriteFile(rc, []byte("graph_path = \"custom/store\"\n[embed\n"), 0o644); err != nil {
		t.Fatalf("write .chitonrc: %v", err)
	}
	t.Setenv(ProjectDirEnv, project)

	cfg, err := Load()
	if err == nil {
		t.Fatalf("Load succeeded with GraphPath %q, want a parse error", cfg.GraphPath)
	}
	if !strings.Contains(err.Error(), rc) {
		t.Errorf("error %q does not name the file %s", err, rc)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/config/ -run TestMalformedChitonrcIsAnError -count=1`
Expected: FAIL: `Load succeeded with GraphPath "...\chiton-out\graph.kuzu", want a parse error`

- [ ] **Step 3: Implement**

In `internal/config/config.go`, extend the imports:

```go
import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/viper"
)
```

Replace lines 86-87 (`// Config file is optional.` / `_ = v.ReadInConfig()`) inside `LoadFrom` with:

```go
	// The config file is optional, but one that exists and fails to read or parse must stop the command,
	// since falling back to defaults would silently open a different store than the file names.
	if err := v.ReadInConfig(); err != nil {
		var notFound viper.ConfigFileNotFoundError
		if !errors.As(err, &notFound) && !errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("read config %s: %w", v.ConfigFileUsed(), err)
		}
	}
```

(`ConfigFileUsed()` is set before the parse, see viper `viper.go:2051-2060`, so the error names the file.)

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/config/... -count=1; go vet ./internal/config/...`
Expected: PASS

- [ ] **Step 5: Docs**

- `README.md:358-363` (Configuration): append "A `.chitonrc` that exists but fails to parse is an error that names the file."
- `CHANGELOG.md` under `### Fixed`: "- **A malformed `.chitonrc` is now an error.** It used to be ignored, so a typo silently dropped `graph_path` back to the default and the command opened a different store."

- [ ] **Step 6: Commit**

```powershell
git add internal/config/config.go internal/config/config_test.go README.md CHANGELOG.md
git commit -m "fix(config): fail on an unreadable .chitonrc instead of ignoring it"
```

### Task E2: Resolve `.chiton/` state against `CHITON_PROJECT_DIR`

**Finding:** Per-project state uses a cwd-relative `.chiton`: `cmd/chiton/analyze.go:18` (`const chitonDir = ".chiton"`, used at `:114`, `:204`, `:237`, `:277`), the serve wiring at `cmd/chiton/main.go:941-942`, the keyfile fallback `ResolveKeyfilePath(".chiton")` at `internal/crypto/resolve.go:24`, and the defaults in `internal/mcp/server.go:58-61` and `internal/analyze/llm.go:106-108`. An assistant-launched `serve` may start in any directory, so `diff_findings` reads baselines from the wrong place, LLM-rule state lands in the wrong place, and a keyfile beside a custom `graph_path` is missed. `graph_path` and `.chitonrc` already honour `CHITON_PROJECT_DIR` (`config.go:11-22`).
**Priority:** P2   **Effort:** M   **Depends on:** none
**Behaviour change:** With `CHITON_PROJECT_DIR` set, baselines, `llm-analyzed.json` and the last keyfile candidate live under `$CHITON_PROJECT_DIR/.chiton`. With it unset, nothing moves (the path stays the relative `.chiton`).

**Files:**
- Modify: `internal/config/config.go:11-22` (add `StateDirName`, `StateDir` after `ProjectDir`)
- Modify: `internal/crypto/resolve.go:18-24`
- Modify: `cmd/chiton/analyze.go:18,114,204,237,277`
- Modify: `cmd/chiton/main.go:254,335,941-942,1176`
- Modify: `internal/mcp/server.go:58-61`, `internal/analyze/llm.go:97-108`
- Test: `internal/config/config_test.go`, `internal/crypto/crypto_test.go:175-203`
- Create: `cmd/chiton/store_test.go`

**Interfaces:**
- Consumes: none
- Produces: `config.StateDirName = ".chiton"`; `config.StateDir() string`; `crypto.ResolveKeyForGraph(graphPath, stateDir string) ([]byte, *KeyFile, error)`; test helper `testProject(t *testing.T) string` in `cmd/chiton/store_test.go` (E4, E5, E6 reuse it)

- [ ] **Step 1: Write the failing tests**

Append to `internal/config/config_test.go`:

```go
func TestStateDirFollowsProjectDir(t *testing.T) {
	isolate(t)
	if got := StateDir(); got != ".chiton" {
		t.Errorf("StateDir() = %q with no project dir, want %q", got, ".chiton")
	}
	project := t.TempDir()
	t.Setenv(ProjectDirEnv, project)
	if got, want := StateDir(), filepath.Join(project, ".chiton"); got != want {
		t.Errorf("StateDir() = %q, want %q", got, want)
	}
}
```

In `internal/crypto/crypto_test.go`, change the two existing calls (`:177`, `:193`) from `ResolveKeyForGraph(graph)` to `ResolveKeyForGraph(graph, filepath.Join(t.TempDir(), ".chiton"))`, and append:

```go
// A keyfile in the project's state dir must be found even when the graph lives elsewhere and the process
// runs from another directory, as an assistant-launched server does.
func TestResolveKeyForGraph_FindsKeyfileInStateDir(t *testing.T) {
	stateDir := filepath.Join(t.TempDir(), ".chiton")
	if err := os.MkdirAll(stateDir, 0o700); err != nil {
		t.Fatal(err)
	}
	kf, err := DefaultKeyFile()
	if err != nil {
		t.Fatal(err)
	}
	if err := kf.Save(ResolveKeyfilePath(stateDir), false); err != nil {
		t.Fatal(err)
	}
	t.Setenv(EnvVarPassphrase, "test-passphrase")

	graph := filepath.Join(t.TempDir(), "store", "graph.kuzu")
	key, got, err := ResolveKeyForGraph(graph, stateDir)
	if err != nil {
		t.Fatalf("ResolveKeyForGraph: %v", err)
	}
	if got == nil || len(key) != 32 {
		t.Fatalf("keyfile in the state dir was not used: kf=%v keylen=%d", got, len(key))
	}
}
```

Create `cmd/chiton/store_test.go`:

```go
package main

import (
	"context"
	"io"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/config"
	chitoncrypto "github.com/Eric-Fernald/chiton/internal/crypto"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

// testProject anchors chiton at a scratch project, as the plugin anchors an assistant-launched server, and
// returns it. The working directory is left alone to prove nothing depends on it.
func testProject(t *testing.T) string {
	t.Helper()
	project := t.TempDir()
	t.Setenv(config.ProjectDirEnv, project)
	t.Setenv("CHITON_GRAPH_PATH", "")
	t.Setenv(chitoncrypto.EnvVarPassphrase, "")
	return project
}

// diff_findings baselines and LLM-rule state must live under CHITON_PROJECT_DIR, not the server's cwd.
func TestServeStateFollowsProjectDir(t *testing.T) {
	project := testProject(t)
	gs, err := badgerstore.Open(filepath.Join(project, "chiton-out", "graph.kuzu"))
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()
	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())
	cmd.SetErr(io.Discard)
	cfg := &config.Config{Embed: config.EmbedConfig{Provider: "hash"}}

	deps := buildSemanticDeps(cmd, cfg, gs, nil)
	if want := filepath.Join(project, ".chiton", "baselines"); deps.BaselineDir != want {
		t.Errorf("BaselineDir = %q, want %q", deps.BaselineDir, want)
	}
}
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `go test ./internal/config/ ./internal/crypto/ ./cmd/chiton/ -run "TestStateDirFollowsProjectDir|TestResolveKeyForGraph|TestServeStateFollowsProjectDir" -count=1`
Expected: FAIL. `internal/config` and `internal/crypto` fail to build (`undefined: StateDir`, `too many arguments in call to ResolveKeyForGraph`); `cmd/chiton` fails with `BaselineDir = ".chiton\baselines", want "...\.chiton\baselines"`.

- [ ] **Step 3: Implement**

`internal/config/config.go`, after `ProjectDir` (line 22):

```go
// StateDirName is the per-project directory for the keyfile, finding baselines and LLM-rule state.
const StateDirName = ".chiton"

// StateDir returns ProjectDir's .chiton, so a server started in any directory finds the same state as the CLI.
// It stays the relative ".chiton" when CHITON_PROJECT_DIR is unset.
func StateDir() string { return filepath.Join(ProjectDir(), StateDirName) }
```

`internal/crypto/resolve.go:18-24` becomes:

```go
// ResolveKeyForGraph finds a keyfile beside graphPath, in a .chiton beside graphPath's directory, or in stateDir
// (normally config.StateDir()), then derives the key via GetPassphrase. With no keyfile it returns (nil, nil, nil).
func ResolveKeyForGraph(graphPath, stateDir string) ([]byte, *KeyFile, error) {
	candidates := []string{
		ResolveKeyfilePath(filepath.Dir(graphPath)),
		ResolveKeyfilePath(filepath.Join(filepath.Dir(graphPath), "..", ".chiton")),
		ResolveKeyfilePath(stateDir),
	}
```

(the rest of the function, lines 25-51, is unchanged).

`cmd/chiton/analyze.go`: delete line 18 (`const chitonDir = ".chiton"`) and replace each use:
- `:114` `baselineDir := analyze.BaselineDir(config.StateDir())`
- `:204` `svc := analyze.NewLLMService(gs, nil, cfg.Semantic.ConfidenceThreshold, config.StateDir())`
- `:237` `svc := analyze.NewLLMService(nil, nil, cfg.Semantic.ConfidenceThreshold, config.StateDir())`
- `:277` `svc := analyze.NewLLMService(gs, nil, cfg.Semantic.ConfidenceThreshold, config.StateDir())`

`cmd/chiton/main.go`:
- `:254` (init) `chitonDir := filepath.Join(dir, config.StateDirName)` (init keeps its explicit `--dir`)
- `:335` `key, _, err := chitoncrypto.ResolveKeyForGraph(graphPath, config.StateDir())`
- `:941-942` in `buildSemanticDeps`:

```go
	deps.LLM = analyze.NewLLMService(gs, nil, cfg.Semantic.ConfidenceThreshold, config.StateDir())
	deps.BaselineDir = analyze.BaselineDir(config.StateDir())
```

- `:1176` (hooks run keeps its `--dir`) `hooksYaml := filepath.Join(dir, config.StateDirName, "hooks.yaml")`

`internal/mcp/server.go:58-61` (add the `internal/config` import):

```go
	baselineDir := deps.BaselineDir
	if baselineDir == "" {
		baselineDir = analyze.BaselineDir(config.StateDir())
	}
```

`internal/analyze/llm.go:97-108` (add the `internal/config` import; one-line overlap with Section D's package, the fallback is only reached by callers that pass ""):

```go
// NewLLMService builds a Service. reg=nil uses the default registry;
// threshold<=0 uses 0.5; stateDir="" defaults to config.StateDir().
func NewLLMService(s store.GraphStore, reg *Registry, threshold float32, stateDir string) *LLMService {
	if reg == nil {
		reg = DefaultRegistry()
	}
	if threshold <= 0 {
		threshold = model.DefaultConfidenceThreshold
	}
	if stateDir == "" {
		stateDir = config.StateDir()
	}
	return &LLMService{store: s, registry: reg, threshold: threshold, stateDir: stateDir, now: time.Now}
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/config/... ./internal/crypto/... ./internal/analyze/... ./internal/mcp/... ./cmd/chiton/... -count=1; go vet ./internal/config/... ./internal/crypto/... ./internal/analyze/... ./internal/mcp/... ./cmd/chiton/...`
Expected: PASS

Then confirm no cwd-relative state path is left: `Get-ChildItem cmd\chiton\*.go -Exclude *_test.go | Select-String -Pattern '"\.chiton"'` prints nothing.

- [ ] **Step 5: Docs**

- `CLAUDE.md` (the `CHITON_PROJECT_DIR` paragraph under "Building the graph"): after "a relative `graph_path` resolves against it", add "and the `.chiton/` state directory (keyfile lookup, `analyze --diff` baselines, LLM-rule state) lives under it".
- `README.md:358-363`: same addition.
- `docs/INSTALL.md:269` (`CHITON_PROJECT_DIR` row): append "`.chiton/` state (baselines, LLM-rule state, keyfile lookup) resolves here too."
- `docs/SECURITY.md:278-281`: replace "and finally in `./.chiton/`" with "and finally in `$CHITON_PROJECT_DIR/.chiton/` (the working directory's `.chiton/` when unset)".
- `CHANGELOG.md` under `### Fixed`: "- **`.chiton/` state now follows `CHITON_PROJECT_DIR`.** An assistant-launched server started outside the workspace read `diff_findings` baselines and wrote LLM-rule state relative to its own working directory."

- [ ] **Step 6: Commit**

```powershell
git add internal/config/config.go internal/config/config_test.go internal/crypto/resolve.go internal/crypto/crypto_test.go cmd/chiton/analyze.go cmd/chiton/main.go cmd/chiton/store_test.go internal/mcp/server.go internal/analyze/llm.go CLAUDE.md README.md docs/INSTALL.md docs/SECURITY.md CHANGELOG.md
git commit -m "fix(config): resolve .chiton state against CHITON_PROJECT_DIR"
```

### Task E3: Refuse a remote embed endpoint chosen by the repository's `.chitonrc`

**Finding:** `.chitonrc` is read from the project directory first (`internal/config/config.go:60-64`), and `embed.endpoint` flows unchecked into `NewOllama` (`internal/embed/ollama.go:31-35`), which POSTs text to it (`ollama.go:77-78`). Cloning a hostile repo and running `chiton embed`, `chiton sync --embed`, `chiton search`, or the plugin's `serve --stdio` (whose `semantic_search` embeds each query, `cmd/chiton/main.go:948`) sends every doc section, ingested email and Slack text, and every query to the attacker. All embedder construction goes through `newEmbedder` (`cmd/chiton/main.go:76-91`; callers `semantic.go:54,226,350`, `sync.go:78`, `main.go:948`).
**Priority:** P2   **Effort:** M   **Depends on:** E1
**Behaviour change:** A non-loopback `embed.endpoint` that comes from the project `.chitonrc` is refused with an error naming `CHITON_EMBED_ENDPOINT`. `chiton embed`/`search` fail, `serve` and `sync --embed` log a warning and run without embeddings. Endpoints from `CHITON_EMBED_ENDPOINT` or `$HOME/.chitonrc` (read only when the project has no `.chitonrc`) are unaffected, as are loopback endpoints and `--provider hash`. When the project dir is `$HOME` itself, the file counts as the project's.

**Files:**
- Modify: `internal/config/config.go:24-31` (`EmbedConfig`), `LoadFrom` (after `Unmarshal`), new functions after `resolveAgainst`
- Modify: `cmd/chiton/main.go:74-91` (`newEmbedder`)
- Test: `internal/config/config_test.go` (`isolate` gains one line, new tests)
- Create: `cmd/chiton/embedder_test.go`

**Interfaces:**
- Consumes: `isolate(t) string` (E1)
- Produces: `type config.Origin string`; constants `config.OriginDefault`, `config.OriginEnv`, `config.OriginUserRC`, `config.OriginProjectRC`; field `config.EmbedConfig.EndpointOrigin Origin`; method `func (e config.EmbedConfig) CheckEndpoint() error`

- [ ] **Step 1: Write the failing tests**

In `isolate` (`internal/config/config_test.go`), add `t.Setenv("CHITON_EMBED_ENDPOINT", "")` before `return home`. Append:

```go
func writeRC(t *testing.T, dir, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, ".chitonrc"), []byte(body), 0o644); err != nil {
		t.Fatalf("write .chitonrc: %v", err)
	}
}

// A cloned repository's .chitonrc must not be able to send embedded text and queries to its own server.
func TestProjectRCRemoteEmbedEndpointRefused(t *testing.T) {
	isolate(t)
	project := t.TempDir()
	writeRC(t, project, "[embed]\nendpoint = \"https://embed.attacker.example\"\n")
	t.Setenv(ProjectDirEnv, project)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Embed.EndpointOrigin != OriginProjectRC {
		t.Errorf("EndpointOrigin = %q, want %q", cfg.Embed.EndpointOrigin, OriginProjectRC)
	}
	if err := cfg.Embed.CheckEndpoint(); err == nil || !strings.Contains(err.Error(), "CHITON_EMBED_ENDPOINT") {
		t.Errorf("CheckEndpoint() = %v, want a refusal naming CHITON_EMBED_ENDPOINT", err)
	}
}

func TestEmbedEndpointAllowedOrigins(t *testing.T) {
	remote := "[embed]\nendpoint = \"https://embed.attacker.example\"\n"
	cases := []struct {
		name, projectRC, homeRC, env string
		want                         Origin
	}{
		{name: "project loopback IP", projectRC: "[embed]\nendpoint = \"http://127.0.0.1:11434\"\n", want: OriginProjectRC},
		{name: "project localhost", projectRC: "[embed]\nendpoint = \"http://localhost:9999\"\n", want: OriginProjectRC},
		{name: "env overrides project", projectRC: remote, env: "https://embed.corp.example", want: OriginEnv},
		{name: "home config", homeRC: "[embed]\nendpoint = \"https://embed.corp.example\"\n", want: OriginUserRC},
		{name: "built-in default", want: OriginDefault},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			home := isolate(t)
			project := t.TempDir()
			if tc.projectRC != "" {
				writeRC(t, project, tc.projectRC)
			}
			if tc.homeRC != "" {
				writeRC(t, home, tc.homeRC)
			}
			t.Setenv(ProjectDirEnv, project)
			t.Setenv("CHITON_EMBED_ENDPOINT", tc.env)

			cfg, err := Load()
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			if cfg.Embed.EndpointOrigin != tc.want {
				t.Errorf("EndpointOrigin = %q, want %q", cfg.Embed.EndpointOrigin, tc.want)
			}
			if err := cfg.Embed.CheckEndpoint(); err != nil {
				t.Errorf("CheckEndpoint() = %v, want nil", err)
			}
		})
	}
}

func TestIsLoopbackURL(t *testing.T) {
	for raw, want := range map[string]bool{
		"http://localhost:11434":            true,
		"https://LOCALHOST":                 true,
		"http://127.0.0.1:11434":            true,
		"http://[::1]:11434":                true,
		"http://localhost.attacker.example": false,
		"http://127.0.0.1.attacker.example": false,
		"https://embed.attacker.example":    false,
		"localhost:11434":                   false, // no scheme: url.Parse reads "localhost" as the scheme
		"ftp://127.0.0.1":                   false,
	} {
		if got := isLoopbackURL(raw); got != want {
			t.Errorf("isLoopbackURL(%q) = %v, want %v", raw, got, want)
		}
	}
}
```

Create `cmd/chiton/embedder_test.go`:

```go
package main

import (
	"testing"

	"github.com/Eric-Fernald/chiton/internal/config"
)

func TestNewEmbedderRefusesRepoChosenRemoteEndpoint(t *testing.T) {
	cfg := &config.Config{Embed: config.EmbedConfig{
		Provider:       "ollama",
		Endpoint:       "https://embed.attacker.example",
		EndpointOrigin: config.OriginProjectRC,
	}}
	if _, err := newEmbedder(cfg, "", ""); err == nil {
		t.Fatal("newEmbedder accepted a remote endpoint chosen by the project .chitonrc")
	}
	// The hash provider never contacts the endpoint, so it stays available.
	if _, err := newEmbedder(cfg, "hash", ""); err != nil {
		t.Errorf("newEmbedder with provider hash: %v", err)
	}
}
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `go test ./internal/config/ ./cmd/chiton/ -run "Endpoint|IsLoopbackURL|NewEmbedder" -count=1`
Expected: FAIL, both packages fail to build: `undefined: OriginProjectRC`, `cfg.Embed.EndpointOrigin undefined`, `undefined: isLoopbackURL`.

- [ ] **Step 3: Implement**

`internal/config/config.go`: add `"net"` and `"net/url"` to the imports, then replace `EmbedConfig` (lines 24-31):

```go
// Origin names the layer that supplied a config value.
type Origin string

const (
	OriginDefault   Origin = "default"
	OriginEnv       Origin = "environment"
	OriginUserRC    Origin = "$HOME/.chitonrc"
	OriginProjectRC Origin = "project .chitonrc"
)

// embedEndpointEnv is the variable viper maps onto embed.endpoint.
const embedEndpointEnv = "CHITON_EMBED_ENDPOINT"

// EmbedConfig configures the embedding provider used by `chiton embed`,
// `chiton search`, and the semantic MCP tools.
type EmbedConfig struct {
	Provider string `mapstructure:"provider"` // ollama | hash
	Model    string `mapstructure:"model"`
	Endpoint string `mapstructure:"endpoint"`
	Dim      int    `mapstructure:"dim"`
	// EndpointOrigin records which layer set Endpoint, so CheckEndpoint can refuse a repository-chosen host.
	EndpointOrigin Origin `mapstructure:"-"`
}
```

In `LoadFrom`, after `cfg.GraphPath = resolveAgainst(dir, cfg.GraphPath)`:

```go
	cfg.Embed.EndpointOrigin = endpointOrigin(v, dir)
```

Add after `resolveAgainst`:

```go
// endpointOrigin reports which layer set embed.endpoint. Viper reads one .chitonrc, the project's before $HOME's.
func endpointOrigin(v *viper.Viper, dir string) Origin {
	switch {
	case os.Getenv(embedEndpointEnv) != "":
		return OriginEnv
	case !v.InConfig("embed.endpoint"):
		return OriginDefault
	case sameDir(filepath.Dir(v.ConfigFileUsed()), dir):
		return OriginProjectRC
	default:
		return OriginUserRC
	}
}

// sameDir reports whether a and b are one directory. A failed stat counts as a match, so a config file that
// cannot be placed gets the stricter project treatment.
func sameDir(a, b string) bool {
	ai, errA := os.Stat(a)
	bi, errB := os.Stat(b)
	return errA != nil || errB != nil || os.SameFile(ai, bi)
}

// CheckEndpoint refuses a non-loopback endpoint set by the project's .chitonrc, since a cloned repository must
// not choose where chiton sends document text and queries. The environment and $HOME/.chitonrc may point anywhere.
func (e EmbedConfig) CheckEndpoint() error {
	if e.EndpointOrigin != OriginProjectRC || isLoopbackURL(e.Endpoint) {
		return nil
	}
	return fmt.Errorf("embed.endpoint %q comes from the project's .chitonrc and is not a loopback address; "+
		"chiton will not send document text or queries off-device because a repository asked. "+
		"If you trust that server, set %s to it yourself", e.Endpoint, embedEndpointEnv)
}

// isLoopbackURL reports whether raw is an http(s) URL whose host is localhost or a loopback IP. Names are not
// resolved, so a DNS answer cannot turn a remote host into an allowed one.
func isLoopbackURL(raw string) bool {
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
		return false
	}
	host := u.Hostname()
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
```

`cmd/chiton/main.go:74-91` becomes:

```go
// newEmbedder constructs the configured embedding provider, with optional provider/model overrides from CLI flags
// (empty = use config). It refuses an Ollama endpoint that the project's .chitonrc points off-device.
func newEmbedder(cfg *config.Config, providerOverride, modelOverride string) (embed.Embedder, error) {
	provider := cfg.Embed.Provider
	if providerOverride != "" {
		provider = providerOverride
	}
	model := cfg.Embed.Model
	if modelOverride != "" {
		model = modelOverride
	}
	// Only the Ollama provider sends text to Endpoint; the hash provider stays usable offline.
	if p := strings.ToLower(strings.TrimSpace(provider)); p == "" || p == "ollama" {
		if err := cfg.Embed.CheckEndpoint(); err != nil {
			return nil, err
		}
	}
	return embed.New(embed.Config{
		Provider: provider,
		Model:    model,
		Endpoint: cfg.Embed.Endpoint,
		Dim:      cfg.Embed.Dim,
	})
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/config/... ./internal/embed/... ./cmd/chiton/... -count=1; go vet ./internal/config/... ./cmd/chiton/...`
Expected: PASS

- [ ] **Step 5: Docs**

- `README.md:371` (`embed.endpoint` row), Notes column: "Ollama endpoint. A non-loopback value in the project's `.chitonrc` is refused; set `CHITON_EMBED_ENDPOINT` (or `$HOME/.chitonrc`, read only when the project has none) to use a remote server."
- `docs/SECURITY.md:93-97`: after "can redirect it off-device", add "A non-loopback `embed.endpoint` from the project's `.chitonrc` is refused, so a cloned repository cannot redirect it; only `CHITON_EMBED_ENDPOINT` or `$HOME/.chitonrc` can."
- `docs/SECURITY.md:154-155`: add "A project `.chitonrc` cannot point `embed.endpoint` off-device (§1.1)."
- `CHANGELOG.md` under `### Security`: "- **A repository's `.chitonrc` can no longer send your text off-device.** A non-loopback `embed.endpoint` set by the project `.chitonrc` is refused. Set `CHITON_EMBED_ENDPOINT` yourself to use a remote embedder."

- [ ] **Step 6: Commit**

```powershell
git add internal/config/config.go internal/config/config_test.go cmd/chiton/main.go cmd/chiton/embedder_test.go README.md docs/SECURITY.md CHANGELOG.md
git commit -m "fix(config): refuse a remote embed endpoint set by the project .chitonrc"
```

### Task E4: Store a passphrase verifier in the keyfile

**Finding:** `KeyFile` holds salt and Argon2id params only (`internal/crypto/keyfile.go:33-44`). `init` saves the keyfile (`cmd/chiton/main.go:308`) and then derives a key it throws away (`main.go:313`). The first `build` prompts once without confirmation (`internal/crypto/resolve.go:42`, `confirm=false`), and Badger accepts any key for a store that does not exist yet, so a typo creates the store under the wrong key and the real passphrase can never open it. For an existing store Badger does reject a wrong key (`ErrEncryptionKeyMismatch`, badger v4 `key_registry.go:158`), but with an opaque message.
**Priority:** P3   **Effort:** M   **Depends on:** E2
**Behaviour change:** `init` writes `verifier` (hex HMAC-SHA256 of the derived key over a fixed label) into `keyfile.json`. Every derivation checks it and fails with `wrong passphrase: ...`. Keyfiles without a verifier keep working and gain one on the first open that succeeds. `KeyFileVersion` stays 1 (the field is optional, older binaries ignore it). Argon2id params are unchanged.

**Files:**
- Modify: `internal/crypto/keyfile.go:1-2,5-17,33-44,74-161`
- Modify: `cmd/chiton/main.go:296-319` (init keyfile branch), `cmd/chiton/main.go:332-348` (`openGraphStore`)
- Test: `internal/crypto/crypto_test.go`, `cmd/chiton/store_test.go`

**Interfaces:**
- Consumes: `crypto.ResolveKeyForGraph(graphPath, stateDir string)` (E2), `testProject` (E2)
- Produces: `crypto.ErrWrongPassphrase`; field `KeyFile.Verifier string` (`json:"verifier,omitempty"`); `func (kf *KeyFile) SetVerifier(key []byte)`; `func (kf *KeyFile) RecordVerifier(key []byte) error`; test helper `writeKeyfile(t *testing.T, project string) string` in `cmd/chiton/store_test.go`

- [ ] **Step 1: Write the failing tests**

In `internal/crypto/crypto_test.go`, add `"errors"` to the imports, add `"bad verifier": func(kf *KeyFile) { kf.Verifier = "zz" },` to the `cases` map in `TestDeriveKey_RejectsBadParams` (line 71-79), and append:

```go
// A mistyped passphrase must fail clearly instead of deriving a key that creates or opens a store.
func TestDeriveKey_VerifierRejectsWrongPassphrase(t *testing.T) {
	kf, err := DefaultKeyFile()
	if err != nil {
		t.Fatal(err)
	}
	key, err := kf.DeriveKey("right horse")
	if err != nil {
		t.Fatal(err)
	}
	kf.SetVerifier(key)
	if _, err := kf.DeriveKey("rihgt horse"); !errors.Is(err, ErrWrongPassphrase) {
		t.Errorf("DeriveKey(typo) err = %v, want ErrWrongPassphrase", err)
	}
	again, err := kf.DeriveKey("right horse")
	if err != nil || !bytes.Equal(again, key) {
		t.Errorf("DeriveKey(right) = %x, %v; want the original key", again, err)
	}
}

// Keyfiles written before verifiers existed have none and must keep accepting any passphrase.
func TestDeriveKey_LegacyKeyfileHasNoVerifier(t *testing.T) {
	kf, err := DefaultKeyFile()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := kf.DeriveKey("anything"); err != nil {
		t.Errorf("legacy keyfile rejected a passphrase: %v", err)
	}
}

func TestKeyFile_RecordVerifierUpgradesInPlace(t *testing.T) {
	path := filepath.Join(t.TempDir(), KeyfileName)
	kf, err := DefaultKeyFile()
	if err != nil {
		t.Fatal(err)
	}
	if err := kf.Save(path, false); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadKeyFile(path)
	if err != nil {
		t.Fatal(err)
	}
	key, err := loaded.DeriveKey("pw")
	if err != nil {
		t.Fatal(err)
	}
	if err := loaded.RecordVerifier(key); err != nil {
		t.Fatalf("RecordVerifier: %v", err)
	}
	reloaded, err := LoadKeyFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if reloaded.Verifier == "" {
		t.Fatal("verifier was not persisted")
	}
	if _, err := reloaded.DeriveKey("not pw"); !errors.Is(err, ErrWrongPassphrase) {
		t.Errorf("after the upgrade, a wrong passphrase gave %v, want ErrWrongPassphrase", err)
	}
}
```

In `cmd/chiton/store_test.go`, extend the imports with `"errors"` and `"os"`, and append:

```go
// writeKeyfile saves a keyfile with no verifier, as chiton wrote before verifiers, into project/.chiton.
func writeKeyfile(t *testing.T, project string) string {
	t.Helper()
	path := chitoncrypto.ResolveKeyfilePath(filepath.Join(project, config.StateDirName))
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	kf, err := chitoncrypto.DefaultKeyFile()
	if err != nil {
		t.Fatal(err)
	}
	if err := kf.Save(path, false); err != nil {
		t.Fatal(err)
	}
	return path
}

// init must record a verifier, so a later typo fails instead of creating a store under the wrong key.
func TestInitKeyfileRejectsMistypedPassphrase(t *testing.T) {
	project := testProject(t)
	t.Setenv(chitoncrypto.EnvVarPassphrase, "correct horse")
	initc := initCmd()
	initc.SetArgs([]string{"--dir", project})
	initc.SetOut(io.Discard)
	if err := initc.Execute(); err != nil {
		t.Fatalf("init: %v", err)
	}

	t.Setenv(chitoncrypto.EnvVarPassphrase, "correct hrose")
	graph := filepath.Join(project, "chiton-out", "graph.kuzu")
	gs, err := openGraphStore(graph, false)
	if !errors.Is(err, chitoncrypto.ErrWrongPassphrase) {
		if gs != nil {
			gs.Close()
		}
		t.Fatalf("openGraphStore with a typo: err = %v, want ErrWrongPassphrase", err)
	}
	if _, err := os.Stat(graph); !os.IsNotExist(err) {
		t.Errorf("a store was created under the mistyped passphrase (stat err %v)", err)
	}
}

// A keyfile from before verifiers existed gains one on the first open the store accepts.
func TestOpenGraphStoreUpgradesLegacyKeyfile(t *testing.T) {
	project := testProject(t)
	kfPath := writeKeyfile(t, project)
	t.Setenv(chitoncrypto.EnvVarPassphrase, "pw")

	gs, err := openGraphStore(filepath.Join(project, "chiton-out", "graph.kuzu"), false)
	if err != nil {
		t.Fatalf("openGraphStore: %v", err)
	}
	gs.Close()

	kf, err := chitoncrypto.LoadKeyFile(kfPath)
	if err != nil {
		t.Fatal(err)
	}
	if kf.Verifier == "" {
		t.Error("legacy keyfile was not upgraded with a verifier after a successful open")
	}
}
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `go test ./internal/crypto/ ./cmd/chiton/ -run "Verifier|Legacy|Mistyped" -count=1`
Expected: FAIL, both packages fail to build: `kf.SetVerifier undefined`, `undefined: ErrWrongPassphrase`, `kf.Verifier undefined`.

- [ ] **Step 3: Implement**

`internal/crypto/keyfile.go`. Package doc (lines 1-2):

```go
// Package crypto handles chiton's at-rest encryption. It derives an Argon2id key from a passphrase and seals
// files with AES-256-GCM. The keyfile holds salt, params and a verifier tag, never the key or passphrase.
```

Imports (lines 5-17):

```go
import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"

	"golang.org/x/crypto/argon2"

	"github.com/Eric-Fernald/chiton/internal/fsutil"
)
```

`KeyFile` (lines 33-44) plus the new declarations after it:

```go
// KeyFile is the on-disk metadata for deriving a chiton encryption key from a passphrase, which is never
// persisted. Defaults follow OWASP 2024 Argon2id guidance: 64 MiB, 3 iterations, parallelism 4.
type KeyFile struct {
	Version     int    `json:"version"`
	KDF         string `json:"kdf"`                // "argon2id"
	SaltHex     string `json:"salt_hex"`           // 16-byte salt, hex-encoded
	MemoryKiB   uint32 `json:"memory_kib"`         // Argon2id memory cost
	Iterations  uint32 `json:"iterations"`         // Argon2id time cost
	Parallelism uint8  `json:"parallelism"`        // Argon2id parallelism
	KeyLength   uint32 `json:"key_length"`         // bytes (32 for AES-256)
	Cipher      string `json:"cipher"`             // "AES-256-GCM"
	Verifier    string `json:"verifier,omitempty"` // hex HMAC-SHA256(key, verifierLabel); empty in pre-verifier keyfiles

	path string // where the keyfile was loaded from or saved to, for RecordVerifier
}

// ErrWrongPassphrase means the derived key does not match the keyfile's verifier.
var ErrWrongPassphrase = errors.New("wrong passphrase: it does not match the one chosen at 'chiton init'")

// verifierLabel is the HMAC message whose tag under the derived key is stored as the verifier.
const verifierLabel = "chiton-verifier"

func verifierMAC(key []byte) []byte {
	m := hmac.New(sha256.New, key)
	m.Write([]byte(verifierLabel))
	return m.Sum(nil)
}

// SetVerifier records key's verifier so later derivations detect a mistyped passphrase.
func (kf *KeyFile) SetVerifier(key []byte) { kf.Verifier = hex.EncodeToString(verifierMAC(key)) }

// RecordVerifier adds key's verifier to a keyfile saved before verifiers existed and rewrites it in place.
// It is a no-op when the keyfile already has one.
func (kf *KeyFile) RecordVerifier(key []byte) error {
	if kf.Verifier != "" {
		return nil
	}
	if kf.path == "" {
		return errors.New("keyfile: RecordVerifier on a keyfile that was never loaded or saved")
	}
	kf.SetVerifier(key)
	return kf.Save(kf.path, true)
}

// checkKey returns ErrWrongPassphrase when kf has a verifier that key does not match.
func (kf *KeyFile) checkKey(key []byte) error {
	if kf.Verifier == "" {
		return nil
	}
	want, _ := hex.DecodeString(kf.Verifier) // validate checked the format
	if !hmac.Equal(verifierMAC(key), want) {
		return ErrWrongPassphrase
	}
	return nil
}
```

`validate` (lines 74-100): add before the final `return nil`:

```go
	if kf.Verifier != "" {
		if b, err := hex.DecodeString(kf.Verifier); err != nil || len(b) != sha256.Size {
			return fmt.Errorf("verifier must be %d hex-encoded bytes", sha256.Size)
		}
	}
```

`DeriveKey` (lines 102-120):

```go
// DeriveKey runs Argon2id over the passphrase and the keyfile's parameters. It returns ErrWrongPassphrase when
// the keyfile's verifier does not match the derived key.
func (kf *KeyFile) DeriveKey(passphrase string) ([]byte, error) {
	if err := kf.validate(); err != nil {
		return nil, fmt.Errorf("keyfile: %w", err)
	}
	salt, err := kf.Salt()
	if err != nil {
		return nil, err
	}
	key := argon2.IDKey([]byte(passphrase), salt, kf.Iterations, kf.MemoryKiB, kf.Parallelism, kf.KeyLength)
	if err := kf.checkKey(key); err != nil {
		return nil, err
	}
	return key, nil
}
```

`Save` (lines 124-138): after the successful `fsutil.WriteFileAtomic` and before `return nil`, add `kf.path = path`.
`LoadKeyFile` (lines 142-161): before the final `return &kf, nil`, add `kf.path = path`.

`cmd/chiton/main.go:299-319`, the `else` branch of the init keyfile block:

```go
			} else {
				passphrase, err := chitoncrypto.GetPassphrase("Choose a chiton passphrase (used to encrypt the graph):", true)
				if err != nil {
					return fmt.Errorf("passphrase: %w", err)
				}
				kf, err := chitoncrypto.DefaultKeyFile()
				if err != nil {
					return err
				}
				// Derive before saving so the keyfile carries a verifier: a later typo then fails clearly instead
				// of creating the store under the wrong key. Argon2id at 64 MiB takes about a second.
				key, err := kf.DeriveKey(passphrase)
				if err != nil {
					return fmt.Errorf("derive key: %w", err)
				}
				kf.SetVerifier(key)
				if err := kf.Save(keyfilePath, false); err != nil {
					return err
				}
				fmt.Fprintf(out, "init: created %s\n", keyfilePath)
				fmt.Fprintln(out, "init: passphrase set. Subsequent commands read CHITON_PASSPHRASE from the environment or prompt interactively.")
				created++
			}
```

`openGraphStore` (`main.go:332-348`):

```go
// openGraphStore opens the badger store with encryption if a keyfile is
// detected next to it. readOnly chooses between Open and OpenReadOnly.
func openGraphStore(graphPath string, readOnly bool) (*badgerstore.Store, error) {
	key, kf, err := chitoncrypto.ResolveKeyForGraph(graphPath, config.StateDir())
	if err != nil {
		return nil, fmt.Errorf("open store: resolve encryption key: %w", err)
	}
	open := badgerstore.OpenWith
	if readOnly {
		open = badgerstore.OpenReadOnlyWith
	}
	gs, err := open(graphPath, badgerstore.Options{EncryptionKey: key})
	if err != nil {
		return nil, fmt.Errorf("open store: %w", err)
	}
	// Badger rejects a wrong key for an existing store, so a pre-verifier keyfile can now record one.
	if kf != nil {
		if err := kf.RecordVerifier(key); err != nil {
			fmt.Fprintf(os.Stderr, "chiton: warning: could not add a passphrase verifier to the keyfile: %v\n", err)
		}
	}
	return gs, nil
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/crypto/... ./internal/graph/store/badger/... ./cmd/chiton/... -count=1; go vet ./internal/crypto/... ./cmd/chiton/...`
Expected: PASS

- [ ] **Step 5: Docs**

- `docs/SECURITY.md:132` (`.chiton/keyfile.json` row): "Argon2id salt, KDF parameters, and a verifier tag (HMAC-SHA256 of the key); no key, no passphrase".
- `docs/SECURITY.md` §2 "Key derivation and handling" (after line 249): add "The keyfile also stores a verifier, an HMAC-SHA256 tag of the derived key over a fixed label, so a mistyped passphrase fails with `wrong passphrase` before any store is opened or created. It gives an attacker nothing new: guessing a passphrase against it costs the same Argon2id run as guessing against the encrypted store. Keyfiles from before this change gain a verifier on the first successful open."
- `CHANGELOG.md` under `### Fixed`: "- **A mistyped passphrase no longer creates a store you cannot reopen.** `chiton init` now stores a passphrase verifier in `keyfile.json`, and every command checks it. Older keyfiles are upgraded on their next successful open."

- [ ] **Step 6: Commit**

```powershell
git add internal/crypto/keyfile.go internal/crypto/crypto_test.go cmd/chiton/main.go cmd/chiton/store_test.go docs/SECURITY.md CHANGELOG.md
git commit -m "fix(crypto): store a passphrase verifier and reject wrong passphrases"
```

### Task E5: Make the missing-passphrase failure actionable for assistant-launched servers

**Finding:** `GetPassphrase` returns `ErrNoPassphrase` when stdin is not a TTY (`internal/crypto/passphrase.go:25-28`). `init` encrypts by default (`cmd/chiton/main.go:328`) and the plugin's `.mcp.json` sets only `CHITON_PROJECT_DIR`, so the plugin's `serve --stdio` exits at startup. `main.go:54-57` does print the error to stderr, but the text ("set CHITON_PASSPHRASE or run interactively") says nothing about where an assistant's environment comes from, and `init` never warns. The MCP client only reports a failed connection. `skills/setup/SKILL.md:97-99,153` also advises `chiton init --encrypt=false` as a fix, which does nothing once a keyfile exists (init skips the keyfile branch entirely, `main.go:289-294`).
**Priority:** P2   **Effort:** S   **Depends on:** E4
**Behaviour change:** The missing-passphrase error gains a second line naming `CHITON_PASSPHRASE` and where to set it. `init` prints a note that an assistant-launched `serve --stdio` needs it. DECISION NEEDED: OS keychain integration (Windows Credential Manager, macOS Keychain, libsecret) would remove the env var, and the recommendation is to defer it to a follow-up and not plan it here.

Option choice: (a) and (b) are both message-only changes on paths that already run. A degraded `serve` mode that starts without a key and answers every tool with the fix would put the message inside the assistant, but it adds a second server mode with its own tests, which is not worth it before keychain support decides the long-term answer.

**Files:**
- Modify: `cmd/chiton/main.go:3-17` (imports), `main.go:296-323` (init note), `main.go:332-348` (`openGraphStore`), new `passphraseHint` below it
- Test: `cmd/chiton/store_test.go`

**Interfaces:**
- Consumes: `writeKeyfile`, `testProject` (E2, E4); `openGraphStore` shape from E4
- Produces: `passphraseHint(err error) error` (E6 moves its call into `resolveStoreKey`)

- [ ] **Step 1: Write the failing tests**

Add `"bytes"` and `"strings"` to the `cmd/chiton/store_test.go` imports and append:

```go
// An assistant-launched server has no terminal, so its error must say where to supply the passphrase.
func TestOpenGraphStoreExplainsMissingPassphrase(t *testing.T) {
	project := testProject(t)
	writeKeyfile(t, project)

	gs, err := openGraphStore(filepath.Join(project, "chiton-out", "graph.kuzu"), true)
	if err == nil {
		gs.Close()
		t.Fatal("openGraphStore succeeded without a passphrase")
	}
	if !errors.Is(err, chitoncrypto.ErrNoPassphrase) {
		t.Errorf("err = %v, want it to wrap ErrNoPassphrase", err)
	}
	if !strings.Contains(err.Error(), "environment that starts your assistant") {
		t.Errorf("err = %q, want the assistant fix spelled out", err)
	}
}

func TestInitNotesAssistantPassphrase(t *testing.T) {
	project := testProject(t)
	t.Setenv(chitoncrypto.EnvVarPassphrase, "pw")
	var out bytes.Buffer
	initc := initCmd()
	initc.SetArgs([]string{"--dir", project})
	initc.SetOut(&out)
	if err := initc.Execute(); err != nil {
		t.Fatalf("init: %v", err)
	}
	if !strings.Contains(out.String(), "serve --stdio") {
		t.Errorf("init output lacks the assistant passphrase note:\n%s", out.String())
	}
}
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `go test ./cmd/chiton/ -run "ExplainsMissingPassphrase|NotesAssistantPassphrase" -count=1`
Expected: FAIL: `err = "open store: resolve encryption key: ...: no passphrase available: set CHITON_PASSPHRASE or run interactively", want the assistant fix spelled out`, and `init output lacks the assistant passphrase note`.

- [ ] **Step 3: Implement**

Add `"errors"` to the `cmd/chiton/main.go` imports. In `initCmd`, after the closing `}` of the keyfile `if/else` (line 319) and before `init: done`:

```go
			fmt.Fprintf(out, "init: note: an assistant-launched 'chiton serve --stdio' cannot prompt; set %s in the environment that starts your assistant or its MCP connection will fail\n", chitoncrypto.EnvVarPassphrase)
```

`openGraphStore` becomes (only the first error return changes from E4):

```go
// openGraphStore opens the badger store with encryption if a keyfile is
// detected next to it. readOnly chooses between Open and OpenReadOnly.
func openGraphStore(graphPath string, readOnly bool) (*badgerstore.Store, error) {
	key, kf, err := chitoncrypto.ResolveKeyForGraph(graphPath, config.StateDir())
	if err != nil {
		return nil, fmt.Errorf("open store: resolve encryption key: %w", passphraseHint(err))
	}
	open := badgerstore.OpenWith
	if readOnly {
		open = badgerstore.OpenReadOnlyWith
	}
	gs, err := open(graphPath, badgerstore.Options{EncryptionKey: key})
	if err != nil {
		return nil, fmt.Errorf("open store: %w", err)
	}
	// Badger rejects a wrong key for an existing store, so a pre-verifier keyfile can now record one.
	if kf != nil {
		if err := kf.RecordVerifier(key); err != nil {
			fmt.Fprintf(os.Stderr, "chiton: warning: could not add a passphrase verifier to the keyfile: %v\n", err)
		}
	}
	return gs, nil
}

// passphraseHint adds the fix for a missing passphrase, which an assistant-launched server hits because it has
// no terminal to prompt on. Other errors pass through unchanged.
func passphraseHint(err error) error {
	if !errors.Is(err, chitoncrypto.ErrNoPassphrase) {
		return err
	}
	return fmt.Errorf("%w\n  fix: set %s in the environment that starts your assistant (for example your shell profile), "+
		"then restart the assistant. Never put the passphrase in .mcp.json or any file in the repository", err, chitoncrypto.EnvVarPassphrase)
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./cmd/chiton/... -count=1; go vet ./cmd/chiton/...`
Expected: PASS

- [ ] **Step 5: Docs**

- `skills/setup/SKILL.md:88-102`: keep the quoted error (its first line is unchanged) and add "The error's second line names the fix." Replace the `--encrypt=false` bullet with: "Or, before the first `chiton init`, scaffold without encryption: `chiton init --encrypt=false`. Once a keyfile exists that flag changes nothing; turning encryption off then means removing `.chiton/keyfile.json` and `chiton-out/` and rebuilding, which discards ingested meetings, email and Slack." Add: "OS keychain support is not implemented; the environment variable is the only non-interactive path."
- `skills/setup/SKILL.md:153` (troubleshooting row) Fix column: "export `CHITON_PASSPHRASE` in the shell profile that starts the assistant, then restart it".
- `README.md:381-386`: add "An assistant-launched `chiton serve --stdio` cannot prompt, so export `CHITON_PASSPHRASE` in the environment that starts the assistant."
- `docs/INSTALL.md:271`: append "`chiton init` prints this reminder."
- `CHANGELOG.md` under `### Changed`: "- **The missing-passphrase error now says how to fix it for an assistant**, and `chiton init` warns that an assistant-launched `serve --stdio` needs `CHITON_PASSPHRASE`."

- [ ] **Step 6: Commit**

```powershell
git add cmd/chiton/main.go cmd/chiton/store_test.go skills/setup/SKILL.md README.md docs/INSTALL.md CHANGELOG.md
git commit -m "fix(cli): explain how to supply the passphrase to an assistant-launched server"
```

### Task E6: Close encryption gaps (plaintext notice, sealed cluster sidecar, `chiton-out/.gitignore`)

**Finding:** `openGraphStore` (`cmd/chiton/main.go:332-348`) opens a plaintext store without a word when no keyfile exists, so `chiton build .` without `init` never encrypts and never says so. `cluster.WriteSidecar` (`internal/graph/cluster/sidecar.go:45-55`) writes every node's label and `source_uri` as plaintext JSON (`0o644`) even for an encrypted store. For a graph with email, Slack, meetings or browser history, that means subjects, person names and URLs. `crypto.EncryptBytes`/`DecryptBytes` (`internal/crypto/file.go:19-63`) exist but only tests call them. `chiton-out/` gets no `.gitignore` (`docs/SECURITY.md:141-143`, `README.md:377-379`). `cache.json` (`internal/cache/cache.go`: `file://` URIs of repo files and their SHA-256) and `.chiton/baselines/*.json` (`internal/analyze/diff.go:20-26`: rule IDs and messages about repo files) hold only data derived from repository files that already sit in plaintext in the working tree, so they stay plaintext.
**Priority:** P2   **Effort:** L   **Depends on:** E2, E4, E5, D6
**Behaviour change:** Opening a store with no keyfile prints one stderr line per command, which `CHITON_PLAINTEXT_OK=1` silences. For an encrypted store, `cluster.json` is sealed with AES-256-GCM under an HKDF subkey of the store key. An existing plaintext `cluster.json` still loads and is sealed on its next write. A writable open writes `chiton-out/.gitignore` (`*`) when the store's directory is named `chiton-out` and has no `.gitignore`. DECISION NEEDED: notice only (recommended) vs. making `chiton build` refuse to create a plaintext store unless `chiton init --encrypt=false` ran, which would need `init` to write an opt-out marker.

Note: the brief names `cluster.SaveSidecar`, but the real functions are `cluster.WriteSidecar(path string, r *Results) error` and `cluster.LoadSidecar(path string) (*Results, error)`. Section D (Task D6) owns their serialization. This task only adds a `key []byte` parameter and one seal/unseal call between D6's encode and the file write, and between the file read and D6's decode. The code below shows today's `json.MarshalIndent` and `json.Unmarshal` calls. If D6 has landed, keep D6's encode and decode calls in those two places and add only the lines marked `E6`. D6's seam is `marshalSidecar(r *Results) ([]byte, error)` and `unmarshalSidecar(data []byte) (*Results, error)`, called only from `WriteSidecar` and `LoadSidecar`.

**Files:**
- Modify: `internal/crypto/file.go` (add `Seal`, `Unseal`, `FileKey`, `ErrSealedNoKey`)
- Modify: `internal/graph/cluster/sidecar.go:45-72`
- Modify: `internal/ingest/engine.go:47-55` (`Config`), `:400`, `:421`
- Modify: `cmd/chiton/main.go` (imports; `initCmd:289-294`; `openGraphStore:332-348`; new `storeKeyInfo`, `resolveStoreKey`, `loadClusterSidecar`, `writeClusterSidecar`, `plaintextNotice`, `ensureOutGitignore`; cluster `:827-835`; serve `:901`; report `:990`)
- Modify: `cmd/chiton/semantic.go:230` and imports, `cmd/chiton/dashboard.go:62`, `cmd/chiton/visualize.go:60` and imports, `cmd/chiton/sync.go:65-101`
- Create: `internal/graph/cluster/sidecar_test.go`
- Test: `internal/crypto/crypto_test.go`, `internal/ingest/engine_test.go:337,352,386` plus a new test, `cmd/chiton/store_test.go`

**Interfaces:**
- Consumes: D6's sidecar encode/decode seam (bytes in, bytes out); `passphraseHint` (E5); `KeyFile.RecordVerifier` (E4); `config.StateDir()` (E2)
- Produces: `crypto.Seal(key, data []byte) ([]byte, error)`; `crypto.Unseal(key, data []byte) ([]byte, error)`; `crypto.FileKey(storeKey []byte) ([]byte, error)`; `crypto.ErrSealedNoKey`; `cluster.WriteSidecar(path string, r *Results, key []byte) error`; `cluster.LoadSidecar(path string, key []byte) (*Results, error)`; field `ingest.Config.SidecarKey []byte`

- [ ] **Step 1: Write the failing tests**

Append to `internal/crypto/crypto_test.go`:

```go
func TestSealUnseal(t *testing.T) {
	key := bytes.Repeat([]byte{0x42}, 32)
	plain := []byte(`{"label":"Initech acquisition sync"}`)
	sealed, err := Seal(key, plain)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(sealed, []byte("Initech")) {
		t.Error("sealed output leaks plaintext")
	}
	if got, err := Unseal(key, sealed); err != nil || !bytes.Equal(got, plain) {
		t.Errorf("Unseal = %q, %v; want the plaintext", got, err)
	}
	if got, err := Seal(nil, plain); err != nil || !bytes.Equal(got, plain) {
		t.Errorf("Seal(nil) = %q, %v; a plaintext store must pass data through", got, err)
	}
	if got, err := Unseal(key, plain); err != nil || !bytes.Equal(got, plain) {
		t.Errorf("Unseal of a file written before encryption = %q, %v; want it unchanged", got, err)
	}
	if _, err := Unseal(nil, sealed); !errors.Is(err, ErrSealedNoKey) {
		t.Errorf("Unseal(nil, sealed) err = %v, want ErrSealedNoKey", err)
	}
}

func TestFileKeyIsDistinctFromStoreKey(t *testing.T) {
	storeKey := bytes.Repeat([]byte{0x42}, 32)
	a, err := FileKey(storeKey)
	if err != nil {
		t.Fatal(err)
	}
	b, err := FileKey(storeKey)
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 32 || !bytes.Equal(a, b) || bytes.Equal(a, storeKey) {
		t.Errorf("FileKey = %x, want a stable 32-byte key different from the store key", a)
	}
	if k, err := FileKey(nil); k != nil || err != nil {
		t.Errorf("FileKey(nil) = %x, %v; want nil, nil for a plaintext store", k, err)
	}
}
```

Create `internal/graph/cluster/sidecar_test.go`:

```go
package cluster

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"

	chitoncrypto "github.com/Eric-Fernald/chiton/internal/crypto"
)

func sampleResults() *Results {
	return &Results{NodeCount: 1, CommunityCount: 1, Nodes: map[string]NodeResult{
		"n1": {Label: "Re: acquisition of Initech", SourceURI: "email://msg-1", Degree: 1},
	}}
}

// An encrypted store's sidecar must not hold its labels and source URIs in plaintext.
func TestSidecarSealedWithKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cluster.json")
	key := bytes.Repeat([]byte{7}, 32)
	if err := WriteSidecar(path, sampleResults(), key); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte("Initech")) || bytes.Contains(raw, []byte("email://")) {
		t.Error("sealed sidecar leaks labels or source URIs")
	}
	if _, err := LoadSidecar(path, nil); !errors.Is(err, chitoncrypto.ErrSealedNoKey) {
		t.Errorf("LoadSidecar without a key err = %v, want ErrSealedNoKey", err)
	}
	got, err := LoadSidecar(path, key)
	if err != nil {
		t.Fatal(err)
	}
	if got.Nodes["n1"].Label != "Re: acquisition of Initech" {
		t.Errorf("round-trip label = %q", got.Nodes["n1"].Label)
	}
}

// A plaintext sidecar written before encryption stays readable with a key; its next write seals it.
func TestSidecarLegacyPlaintextReadWithKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cluster.json")
	if err := WriteSidecar(path, sampleResults(), nil); err != nil {
		t.Fatal(err)
	}
	got, err := LoadSidecar(path, bytes.Repeat([]byte{7}, 32))
	if err != nil || got.NodeCount != 1 {
		t.Fatalf("LoadSidecar(plaintext, key) = %+v, %v", got, err)
	}
}
```

In `internal/ingest/engine_test.go`, pass `nil` as the new last argument at `:337` (`cluster.WriteSidecar(sidecar, seed, nil)`), `:352` (`cluster.LoadSidecar(sidecar, nil)`) and `:386` (`cluster.WriteSidecar(..., cluster.Full(nodes, edges, 1.0, nil), nil)`). Add `"bytes"` and `"errors"` to its imports if missing and append:

```go
// Live sync must keep an encrypted store's cluster sidecar sealed when it re-clusters.
func TestReclusterKeepsSidecarSealed(t *testing.T) {
	h := newHarness(t, false, false)
	key := bytes.Repeat([]byte{9}, 32)
	h.eng.cfg.SidecarKey = key
	ctx := context.Background()
	pa := h.write(t, "a.go", "package src\nfunc A() {}\n")
	if _, err := h.eng.SyncFile(ctx, pa); err != nil {
		t.Fatal(err)
	}
	nodes, _ := h.gs.AllNodes(ctx)
	edges, _ := h.gs.AllEdges(ctx)
	sidecar := cluster.SidecarPath(h.eng.cfg.GraphPath)
	if err := cluster.WriteSidecar(sidecar, cluster.Full(nodes, edges, 1.0, nil), key); err != nil {
		t.Fatal(err)
	}

	pb := h.write(t, "b.go", "package src\nfunc B() {}\n")
	if _, err := h.eng.SyncChanges(ctx, []Change{{Path: pb, Op: OpUpsert}}); err != nil {
		t.Fatal(err)
	}
	if _, err := cluster.LoadSidecar(sidecar, nil); !errors.Is(err, chitoncrypto.ErrSealedNoKey) {
		t.Errorf("re-clustered sidecar is readable without the key (err %v); want it sealed", err)
	}
	if _, err := cluster.LoadSidecar(sidecar, key); err != nil {
		t.Errorf("LoadSidecar with the key: %v", err)
	}
}
```

(import `chitoncrypto "github.com/Eric-Fernald/chiton/internal/crypto"` in `engine_test.go`.)

In `cmd/chiton/store_test.go`, add `"github.com/Eric-Fernald/chiton/internal/graph/cluster"` to the imports, add `t.Setenv(plaintextOKEnv, "1")` to `testProject` before `return project`, and append:

```go
func TestOpenGraphStoreWritesOutGitignore(t *testing.T) {
	project := testProject(t)
	gs, err := openGraphStore(filepath.Join(project, "chiton-out", "graph.kuzu"), false)
	if err != nil {
		t.Fatal(err)
	}
	gs.Close()
	b, err := os.ReadFile(filepath.Join(project, "chiton-out", ".gitignore"))
	if err != nil || string(b) != "*\n" {
		t.Errorf("chiton-out/.gitignore = %q, %v; want \"*\\n\"", b, err)
	}

	// A custom store directory may hold the user's own tracked files, so it is left alone.
	gs, err = openGraphStore(filepath.Join(project, "graphs", "g"), false)
	if err != nil {
		t.Fatal(err)
	}
	gs.Close()
	if _, err := os.Stat(filepath.Join(project, "graphs", ".gitignore")); !os.IsNotExist(err) {
		t.Errorf("wrote a .gitignore outside chiton-out (stat err %v)", err)
	}
}

func TestPlaintextNotice(t *testing.T) {
	t.Setenv(plaintextOKEnv, "")
	if msg := plaintextNotice("chiton-out/graph.kuzu"); !strings.Contains(msg, "not encrypted") || !strings.Contains(msg, plaintextOKEnv) {
		t.Errorf("notice = %q, want it to say the store is not encrypted and name %s", msg, plaintextOKEnv)
	}
	t.Setenv(plaintextOKEnv, "1")
	if msg := plaintextNotice("chiton-out/graph.kuzu"); msg != "" {
		t.Errorf("notice with %s=1 = %q, want silence", plaintextOKEnv, msg)
	}
}

// An encrypted project's cluster sidecar is written sealed and read back through the same resolved key.
func TestClusterSidecarSealedForEncryptedProject(t *testing.T) {
	project := testProject(t)
	writeKeyfile(t, project)
	t.Setenv(chitoncrypto.EnvVarPassphrase, "pw")
	graph := filepath.Join(project, "chiton-out", "graph.kuzu")
	gs, err := openGraphStore(graph, false)
	if err != nil {
		t.Fatal(err)
	}
	defer gs.Close()

	r := &cluster.Results{NodeCount: 1, Nodes: map[string]cluster.NodeResult{"n": {Label: "Initech acquisition"}}}
	path, err := writeClusterSidecar(graph, r)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cluster.LoadSidecar(path, nil); !errors.Is(err, chitoncrypto.ErrSealedNoKey) {
		t.Errorf("sidecar readable without the key (err %v); want it sealed", err)
	}
	got, err := loadClusterSidecar(graph)
	if err != nil || got.Nodes["n"].Label != "Initech acquisition" {
		t.Errorf("loadClusterSidecar = %+v, %v", got, err)
	}
}
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `go test ./internal/crypto/ ./internal/graph/cluster/ ./internal/ingest/ ./cmd/chiton/ -count=1`
Expected: FAIL. All four packages fail to build: `undefined: Seal`, `too many arguments in call to WriteSidecar`, `h.eng.cfg.SidecarKey undefined`, `undefined: plaintextOKEnv`, `undefined: writeClusterSidecar`.

- [ ] **Step 3: Implement**

`internal/crypto/file.go`: extend the imports with `"bytes"`, `"crypto/hkdf"`, `"crypto/sha256"`, `"errors"`, and append:

```go
// sealMagic prefixes Seal's output so Unseal can tell a sealed file from one written before encryption was on.
var sealMagic = []byte("CHITON-SEALED-1\x00")

// ErrSealedNoKey means a file is sealed but no keyfile was found to open it.
var ErrSealedNoKey = errors.New("file is encrypted but no keyfile was found for this store")

// Seal returns data unchanged for a nil key (a plaintext store), else sealMagic followed by EncryptBytes' envelope.
func Seal(key, data []byte) ([]byte, error) {
	if key == nil {
		return data, nil
	}
	ct, err := EncryptBytes(key, data)
	if err != nil {
		return nil, err
	}
	out := make([]byte, 0, len(sealMagic)+len(ct))
	return append(append(out, sealMagic...), ct...), nil
}

// Unseal reverses Seal. Unsealed data passes through even with a key, so a sidecar written before encryption stays
// readable until its next write. Sealed data without a key is ErrSealedNoKey.
func Unseal(key, data []byte) ([]byte, error) {
	if !bytes.HasPrefix(data, sealMagic) {
		return data, nil
	}
	if key == nil {
		return nil, ErrSealedNoKey
	}
	plain, err := DecryptBytes(key, data[len(sealMagic):])
	if err != nil {
		return nil, fmt.Errorf("decrypt (wrong passphrase or corrupt file): %w", err)
	}
	return plain, nil
}

// FileKey derives the key that seals sidecar files from the store key via HKDF-SHA256, so Badger's master key
// never doubles as a GCM key. A nil store key (plaintext store) yields nil.
func FileKey(storeKey []byte) ([]byte, error) {
	if storeKey == nil {
		return nil, nil
	}
	return hkdf.Key(sha256.New, storeKey, nil, "chiton sidecar v1", keyLengthBytes)
}
```

`internal/graph/cluster/sidecar.go`: add the import `chitoncrypto "github.com/Eric-Fernald/chiton/internal/crypto"` and replace `WriteSidecar` and `LoadSidecar` (lines 45-72):

```go
// WriteSidecar atomically writes r to path, sealed with key when the store is encrypted (nil key = plaintext),
// so a concurrent reader never sees a partial file.
func WriteSidecar(path string, r *Results, key []byte) error {
	b, err := json.MarshalIndent(r, "", "  ") // D6's encode call replaces this line
	if err != nil {
		return fmt.Errorf("marshal cluster results: %w", err)
	}
	if b, err = chitoncrypto.Seal(key, b); err != nil { // E6
		return fmt.Errorf("seal cluster sidecar %s: %w", path, err)
	}
	if err := fsutil.WriteFileAtomic(path, b, 0o644); err != nil {
		return fmt.Errorf("write cluster sidecar %s: %w", path, err)
	}
	return nil
}

// LoadSidecar reads the cluster sidecar at path, unsealing it with key. Returns (nil, nil) when the
// file does not exist so callers can treat a missing sidecar as "not yet run".
func LoadSidecar(path string, key []byte) (*Results, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read cluster sidecar %s: %w", path, err)
	}
	if data, err = chitoncrypto.Unseal(key, data); err != nil { // E6
		return nil, fmt.Errorf("cluster sidecar %s: %w", path, err)
	}
	var r Results
	if err := json.Unmarshal(data, &r); err != nil { // D6's decode call replaces this block
		return nil, fmt.Errorf("parse cluster sidecar %s: %w", path, err)
	}
	return &r, nil
}
```

`internal/ingest/engine.go:47-55`, add the field to `Config`:

```go
	SidecarKey      []byte        // seals cluster.json for an encrypted store; nil writes plaintext
```

and in `recluster`: line 400 `prev, err := cluster.LoadSidecar(sidecarPath, e.cfg.SidecarKey)`, line 421 `if err := cluster.WriteSidecar(sidecarPath, res, e.cfg.SidecarKey); err != nil {`.

`cmd/chiton/main.go`: add `"sync"` to the imports. In `initCmd`, the `!encrypt` branch (lines 289-294) gains one line after the plaintext note:

```go
				fmt.Fprintf(out, "init: note: set %s=1 to hide the per-command plaintext notice\n", plaintextOKEnv)
```

Replace `openGraphStore` and add the helpers after it:

```go
// storeKeyInfo is a graph's resolved encryption; every field is nil for a plaintext store.
type storeKeyInfo struct {
	key     []byte                // Badger store key
	fileKey []byte                // seals sidecar files, derived from key
	kf      *chitoncrypto.KeyFile // the keyfile key came from
}

// storeKeys memoises resolveStoreKey by graph path.
var storeKeys sync.Map

// resolveStoreKey derives graphPath's keys once per process, so opening the store and its cluster sidecar
// prompts for the passphrase, and runs Argon2id, only once.
func resolveStoreKey(graphPath string) (storeKeyInfo, error) {
	if v, ok := storeKeys.Load(graphPath); ok {
		return v.(storeKeyInfo), nil
	}
	key, kf, err := chitoncrypto.ResolveKeyForGraph(graphPath, config.StateDir())
	if err != nil {
		return storeKeyInfo{}, passphraseHint(err)
	}
	fileKey, err := chitoncrypto.FileKey(key)
	if err != nil {
		return storeKeyInfo{}, err
	}
	info := storeKeyInfo{key: key, fileKey: fileKey, kf: kf}
	storeKeys.Store(graphPath, info)
	return info, nil
}

// openGraphStore opens the badger store with encryption if a keyfile is
// detected next to it. readOnly chooses between Open and OpenReadOnly.
func openGraphStore(graphPath string, readOnly bool) (*badgerstore.Store, error) {
	info, err := resolveStoreKey(graphPath)
	if err != nil {
		return nil, fmt.Errorf("open store: resolve encryption key: %w", err)
	}
	if info.key == nil {
		if msg := plaintextNotice(graphPath); msg != "" {
			fmt.Fprintln(os.Stderr, msg)
		}
	}
	open := badgerstore.OpenWith
	if readOnly {
		open = badgerstore.OpenReadOnlyWith
	}
	gs, err := open(graphPath, badgerstore.Options{EncryptionKey: info.key})
	if err != nil {
		return nil, fmt.Errorf("open store: %w", err)
	}
	// Badger rejects a wrong key for an existing store, so a pre-verifier keyfile can now record one.
	if info.kf != nil {
		if err := info.kf.RecordVerifier(info.key); err != nil {
			fmt.Fprintf(os.Stderr, "chiton: warning: could not add a passphrase verifier to the keyfile: %v\n", err)
		}
	}
	if !readOnly {
		if err := ensureOutGitignore(filepath.Dir(graphPath)); err != nil {
			gs.Close()
			return nil, fmt.Errorf("open store: %w", err)
		}
	}
	return gs, nil
}

// plaintextOKEnv silences the notice openGraphStore prints for a store with no keyfile.
const plaintextOKEnv = "CHITON_PLAINTEXT_OK"

// plaintextNotice returns the one-line notice for opening graphPath unencrypted, or "" when plaintextOKEnv is set.
func plaintextNotice(graphPath string) string {
	if os.Getenv(plaintextOKEnv) != "" {
		return ""
	}
	return fmt.Sprintf("chiton: note: graph store %s is not encrypted (no keyfile found); docs/SECURITY.md section 2 explains how to encrypt it. Set %s=1 to hide this note.", graphPath, plaintextOKEnv)
}

// ensureOutGitignore writes a "*" .gitignore into outDir when missing, so the graph is never committed. Only a
// directory named chiton-out is touched, since a custom graph_path may share its directory with tracked files.
func ensureOutGitignore(outDir string) error {
	if filepath.Base(outDir) != "chiton-out" {
		return nil
	}
	p := filepath.Join(outDir, ".gitignore")
	if _, err := os.Stat(p); err == nil {
		return nil
	}
	if err := os.WriteFile(p, []byte("*\n"), 0o644); err != nil {
		return fmt.Errorf("write %s: %w", p, err)
	}
	return nil
}

// loadClusterSidecar reads graphPath's cluster sidecar, unsealing it when the store is encrypted. A missing
// sidecar is (nil, nil).
func loadClusterSidecar(graphPath string) (*cluster.Results, error) {
	info, err := resolveStoreKey(graphPath)
	if err != nil {
		return nil, err
	}
	return cluster.LoadSidecar(cluster.SidecarPath(graphPath), info.fileKey)
}

// writeClusterSidecar writes r beside graphPath, sealed when the store is encrypted, and returns the sidecar path.
func writeClusterSidecar(graphPath string, r *cluster.Results) (string, error) {
	info, err := resolveStoreKey(graphPath)
	if err != nil {
		return "", err
	}
	path := cluster.SidecarPath(graphPath)
	return path, cluster.WriteSidecar(path, r, info.fileKey)
}
```

Call sites:
- `main.go:827-830` (cluster command) becomes:

```go
			sidecarPath, err := writeClusterSidecar(cfg.GraphPath, results)
			if err != nil {
				return err
			}
```

- `main.go:901` (serve) `clusterResults, err := loadClusterSidecar(cfg.GraphPath)`
- `main.go:990` (report) `clusterResults, err := loadClusterSidecar(cfg.GraphPath)`
- `semantic.go:230` `clusterResults, _ := loadClusterSidecar(cfg.GraphPath)`, then drop the now-unused `internal/graph/cluster` import from `semantic.go`.
- `dashboard.go:62` `overlay, err := loadClusterSidecar(cfg.GraphPath)` (the import stays, `dashboard.go:124` still uses `cluster.Results`).
- `visualize.go:60` `overlay, err := loadClusterSidecar(cfg.GraphPath)`, then drop the unused `internal/graph/cluster` import from `visualize.go`.

`cmd/chiton/sync.go:65-101`, `openSyncEngine`:

```go
// openSyncEngine opens the store read-write, loads the cache, and wires the
// optional embedder/analyzer per flags. The returned closer must be called.
func openSyncEngine(cmd *cobra.Command, f *syncEngineFlags) (*ingest.Engine, func() error, error) {
	cfg, gs, err := loadStore(false)
	if err != nil {
		return nil, nil, err
	}
	// loadStore already resolved the key, so this is a cache hit and never prompts again.
	keys, err := resolveStoreKey(cfg.GraphPath)
	if err != nil {
		gs.Close()
		return nil, nil, err
	}
	ch, err := loadBuildCache(cfg.GraphPath)
	if err != nil {
		gs.Close()
		return nil, nil, err
	}

	var embedder embed.Embedder
	if f.reembed {
		embedder, err = newEmbedder(cfg, f.provider, f.model)
		if err != nil {
			fmt.Fprintf(cmd.ErrOrStderr(), "sync: warning: --embed disabled: %v\n", err)
			embedder = nil
		}
	}

	var analyzer *analyze.Analyzer
	if f.reanalyze {
		analyzer = analyze.New(gs, nil)
	}

	eng := ingest.New(gs, ch, ingest.Config{
		GraphPath:       cfg.GraphPath,
		SidecarKey:      keys.fileKey,
		Resolution:      f.resolution,
		Reembed:         embedder != nil,
		Reanalyze:       analyzer != nil,
		Workers:         f.workers,
		Debounce:        time.Duration(f.debounceMS) * time.Millisecond,
		ClusterDebounce: time.Duration(f.clusterMS) * time.Millisecond,
	}, embedder, analyzer, errLogf(cmd))

	return eng, gs.Close, nil
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/crypto/... ./internal/graph/cluster/... ./internal/ingest/... ./cmd/chiton/... -count=1; go vet ./...; gofmt -l .`
Expected: PASS, and `gofmt -l .` prints nothing. Confirm every sidecar caller moved: `Get-ChildItem cmd\chiton\*.go -Exclude *_test.go | Select-String -Pattern 'cluster\.(LoadSidecar|WriteSidecar)\('` prints only the two lines inside `loadClusterSidecar` and `writeClusterSidecar` in `main.go`.

- [ ] **Step 5: Docs**

- `docs/SECURITY.md:131` (§1.3): split into two rows. `chiton-out/cluster.json` is "**Yes**, sealed with AES-256-GCM under a subkey of the store key when a keyfile is present". `chiton-out/cache.json` stays "**No**: `file://` paths of repository files and their SHA-256, nothing that is not already in the working tree".
- `docs/SECURITY.md:134` (baselines row): append "Plaintext on purpose: rule IDs and messages about repository files."
- `docs/SECURITY.md:141-143`: replace "**`chiton-out/` is not auto-ignored** ..." with "A writable open writes `chiton-out/.gitignore` (`*`), so the store is never committed. A custom `graph_path` directory is left alone."
- `docs/SECURITY.md:216-225` (scope caveat): rewrite as "The Badger store and, for an encrypted store, the cluster sidecar are encrypted. `cache.json`, `.chiton/baselines/`, `.chiton/llm-analyzed.json` and `chiton export` snapshots stay plaintext; the first three hold only data derived from repository files."
- `docs/SECURITY.md` §2 "Enabling and disabling": add "Opening a store with no keyfile prints a one-line notice on stderr; `CHITON_PLAINTEXT_OK=1` hides it. To encrypt an existing plaintext store, run `chiton export --out snapshot.json`, move `chiton-out/` aside, run `chiton init`, then `chiton merge snapshot.json` and `chiton cluster`, and delete the snapshot and the old store. Export before `init`: once a keyfile exists, chiton opens the old store with a key and Badger rejects it."
- `docs/SECURITY.md:492` (§6 row): Status becomes "**Partly mitigated:** `cluster.json` is sealed for encrypted stores; `cache.json`, baselines, `llm-analyzed.json` and `chiton export` output are plaintext."
- `README.md:375-379`: replace the "not auto-ignored" sentence with "chiton writes `chiton-out/.gitignore` (`*`) the first time it opens the store for writing." In `README.md:381-386` add the `CHITON_PLAINTEXT_OK` notice sentence.
- `docs/INSTALL.md:269-273` (Wiring and config table): add a row "`CHITON_PLAINTEXT_OK` | every command that opens a store | Set to `1` to hide the notice printed when a store has no keyfile."
- `CHANGELOG.md` under `### Security`: "- **The cluster sidecar is encrypted for encrypted stores.** `cluster.json` held every node label and source URI in plaintext; it is now sealed with a key derived from the store key. Existing plaintext sidecars load and are sealed on their next write." Under `### Changed`: "- **Opening an unencrypted store prints a one-line notice** (`CHITON_PLAINTEXT_OK=1` hides it), and **`chiton-out/` now gets its own `.gitignore`.**"

- [ ] **Step 6: Commit**

```powershell
git add internal/crypto/file.go internal/crypto/crypto_test.go internal/graph/cluster/sidecar.go internal/graph/cluster/sidecar_test.go internal/ingest/engine.go internal/ingest/engine_test.go cmd/chiton/main.go cmd/chiton/semantic.go cmd/chiton/dashboard.go cmd/chiton/visualize.go cmd/chiton/sync.go cmd/chiton/store_test.go docs/SECURITY.md README.md docs/INSTALL.md CHANGELOG.md
git commit -m "feat(crypto): seal the cluster sidecar, flag plaintext stores, gitignore chiton-out"
```

### Task E7: Create the MCP Unix socket owner-only

**Finding:** `serveUnix` binds with `net.Listen("unix", path)` (`cmd/chiton/serve_transport.go:36`) and never sets permissions, so the socket's mode is `0777 &^ umask`. Under umask `002` every group member can connect, and under `000` every local user can, and the socket serves the whole graph with no authentication (`docs/SECURITY.md:414`, "filesystem permissions only").
**Priority:** P3   **Effort:** S   **Depends on:** none
**Behaviour change:** The socket is created `0600`. Other local users (and group members) can no longer connect, whatever the umask.

**Files:**
- Modify: `cmd/chiton/serve_transport.go:36`
- Create: `cmd/chiton/unixsock_unix.go`, `cmd/chiton/unixsock_other.go`
- Test: `cmd/chiton/unixsock_unix_test.go`

**Interfaces:**
- Consumes: none
- Produces: `listenUnixPrivate(path string) (net.Listener, error)` (package `main`)

- [ ] **Step 1: Write the failing test**

Create `cmd/chiton/unixsock_unix_test.go`:

```go
//go:build unix

package main

import (
	"os"
	"path/filepath"
	"testing"
)

// Anyone who can connect to the socket reads the whole graph, so it must be owner-only whatever the umask.
func TestListenUnixPrivateIsOwnerOnly(t *testing.T) {
	// A short directory keeps the path under macOS's 104-byte socket path limit.
	dir, err := os.MkdirTemp("", "chs")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	path := filepath.Join(dir, "s.sock")

	ln, err := listenUnixPrivate(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o600 {
		t.Errorf("socket mode = %#o, want 0600", got)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

On Windows (compile check only): `$env:GOOS='linux'; go vet ./cmd/chiton/; Remove-Item Env:GOOS`
Expected: FAIL: `undefined: listenUnixPrivate`.
On Linux/macOS (CI or WSL): `go test ./cmd/chiton/ -run TestListenUnixPrivateIsOwnerOnly -count=1`
Expected: FAIL, same build error.

- [ ] **Step 3: Implement**

Create `cmd/chiton/unixsock_unix.go`:

```go
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
```

Create `cmd/chiton/unixsock_other.go`:

```go
//go:build !unix

package main

import "net"

// listenUnixPrivate is only reached on unix, since serveUnix rejects Windows before calling it.
func listenUnixPrivate(path string) (net.Listener, error) { return net.Listen("unix", path) }
```

In `cmd/chiton/serve_transport.go:36`, replace `ln, err := net.Listen("unix", path)` with `ln, err := listenUnixPrivate(path)`. The `net` import stays (`baseContext` uses `net.Listener`).

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./cmd/chiton/... -count=1; go vet ./cmd/chiton/...; $env:GOOS='linux'; go vet ./cmd/chiton/; $env:GOOS='darwin'; go vet ./cmd/chiton/; Remove-Item Env:GOOS`
Expected: PASS on Windows and a clean cross-vet. The unix test itself runs in CI (`.github/workflows/ci.yml` matrix: ubuntu-latest, macos-latest) and must PASS there.

- [ ] **Step 5: Docs**

- `docs/SECURITY.md:414`: Auth column becomes "filesystem permissions: the socket is created `0600`, owner-only".
- `docs/MCP.md:31` (`--unix` row): append "The socket is created owner-only (`0600`)."
- `CHANGELOG.md` under `### Security`: "- **`chiton serve --unix` creates its socket owner-only (`0600`).** Its mode used to follow the umask, which could let group members or every local user read the graph."

- [ ] **Step 6: Commit**

```powershell
git add cmd/chiton/serve_transport.go cmd/chiton/unixsock_unix.go cmd/chiton/unixsock_other.go cmd/chiton/unixsock_unix_test.go docs/SECURITY.md docs/MCP.md CHANGELOG.md
git commit -m "fix(serve): create the MCP unix socket owner-only"
```

### Task E8: Take webhook secrets from a file, not a flag value

**Finding:** `chiton slack serve-events --signing-secret` (`cmd/chiton/slack.go:130`, read at `:84-86`) and `chiton teams serve-webhook --client-state` (`cmd/chiton/teams.go:131`, read at `:85-87`) accept the secret as a flag value, which `ps` and shell history record. `docs/SECURITY.md:375-376` says every secret is read "from the environment only". `teams.go` has no build tag, so the flag exists in every build (only `internal/ingest/teams` is split by `msgraph`).
**Priority:** P3   **Effort:** S   **Depends on:** none
**Behaviour change:** New `--signing-secret-file` and `--client-state-file` read the secret from a file (whitespace trimmed). The env vars still work. The old flags stay registered but hidden, and passing one fails with a message naming the replacement. DECISION NEEDED: old flags fail with guidance (recommended, pre-1.0) vs. keep them working with a cobra `MarkDeprecated` warning for one release.

**Files:**
- Create: `cmd/chiton/secret.go`
- Test: `cmd/chiton/secret_test.go`
- Modify: `cmd/chiton/slack.go:3-13` (drop `"os"`), `slack.go:70-132` (`slackEventsCmd`)
- Modify: `cmd/chiton/teams.go:3-13` (drop `"os"`), `teams.go:71-133` (`teamsWebhookCmd`)

**Interfaces:**
- Consumes: none
- Produces: `readSecret(file, env string) (string, error)`; `secretFlagRemoved(cmd *cobra.Command, name, fileFlag, env string) error` (package `main`)

- [ ] **Step 1: Write the failing tests**

Create `cmd/chiton/secret_test.go`:

```go
package main

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestReadSecretPrefersFileAndTrims(t *testing.T) {
	t.Setenv("CHITON_TEST_SECRET", "from-env")
	p := filepath.Join(t.TempDir(), "secret")
	if err := os.WriteFile(p, []byte("  from-file\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got, err := readSecret(p, "CHITON_TEST_SECRET"); err != nil || got != "from-file" {
		t.Errorf("readSecret(file) = %q, %v; want from-file", got, err)
	}
	if got, err := readSecret("", "CHITON_TEST_SECRET"); err != nil || got != "from-env" {
		t.Errorf("readSecret(env) = %q, %v; want from-env", got, err)
	}
	empty := filepath.Join(t.TempDir(), "empty")
	if err := os.WriteFile(empty, []byte("\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readSecret(empty, "CHITON_TEST_SECRET"); err == nil {
		t.Error("readSecret accepted an empty secret file")
	}
}

// isolateWebhooks keeps these commands from reaching a real store or vendor API if a check regresses.
func isolateWebhooks(t *testing.T) {
	t.Helper()
	t.Setenv("CHITON_PROJECT_DIR", t.TempDir())
	for _, v := range []string{"CHITON_GRAPH_PATH", "CHITON_SLACK_TOKEN", "CHITON_SLACK_SIGNING_SECRET", "CHITON_TEAMS_CLIENT_STATE",
		"CHITON_TEAMS_TENANT_ID", "CHITON_TEAMS_CLIENT_ID", "CHITON_TEAMS_CLIENT_SECRET", "CHITON_TEAMS_USER_ID"} {
		t.Setenv(v, "")
	}
}

func execQuiet(cmd *cobra.Command, args ...string) error {
	cmd.SetArgs(args)
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	return cmd.Execute()
}

// A secret passed as a flag value lands in ps and shell history, so the old flags must refuse and say what to use.
func TestSecretFlagsRemoved(t *testing.T) {
	isolateWebhooks(t)
	if err := execQuiet(slackEventsCmd(), "--signing-secret", "x"); err == nil || !strings.Contains(err.Error(), "--signing-secret-file") {
		t.Errorf("slack serve-events --signing-secret: err = %v, want a pointer to --signing-secret-file", err)
	}
	if err := execQuiet(teamsWebhookCmd(), "--client-state", "x"); err == nil || !strings.Contains(err.Error(), "--client-state-file") {
		t.Errorf("teams serve-webhook --client-state: err = %v, want a pointer to --client-state-file", err)
	}
}

// The file flag must satisfy the secret check; the command then stops at the missing Slack token.
func TestSigningSecretFileIsUsed(t *testing.T) {
	isolateWebhooks(t)
	p := filepath.Join(t.TempDir(), "signing")
	if err := os.WriteFile(p, []byte("s3cret\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	err := execQuiet(slackEventsCmd(), "--signing-secret-file", p)
	if err == nil || strings.Contains(err.Error(), "signing secret is required") || strings.Contains(err.Error(), "unknown flag") {
		t.Errorf("err = %v, want the secret accepted and a later failure (missing CHITON_SLACK_TOKEN)", err)
	}
}
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `go test ./cmd/chiton/ -run "ReadSecret|SecretFlagsRemoved|SigningSecretFile" -count=1`
Expected: FAIL: `undefined: readSecret` (build failure).

- [ ] **Step 3: Implement**

Create `cmd/chiton/secret.go`:

```go
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

// readSecret returns the trimmed contents of file when it is set, else $env. Secrets never come from a flag
// value, which ps and shell history would record.
func readSecret(file, env string) (string, error) {
	if file == "" {
		return os.Getenv(env), nil
	}
	b, err := os.ReadFile(file)
	if err != nil {
		return "", fmt.Errorf("read secret file: %w", err)
	}
	s := strings.TrimSpace(string(b))
	if s == "" {
		return "", fmt.Errorf("secret file %s is empty", file)
	}
	return s, nil
}

// secretFlagRemoved errors when the removed secret-bearing flag name was passed, naming what replaced it.
func secretFlagRemoved(cmd *cobra.Command, name, fileFlag, env string) error {
	if !cmd.Flags().Changed(name) {
		return nil
	}
	return fmt.Errorf("--%s was removed because a secret on the command line shows up in ps and shell history; "+
		"use --%s <path> or set %s", name, fileFlag, env)
}
```

`cmd/chiton/slack.go`: remove `"os"` from the imports and replace `slackEventsCmd` (lines 70-132):

```go
func slackEventsCmd() *cobra.Command {
	var addr, secretFile string

	cmd := &cobra.Command{
		Use:   "serve-events",
		Short: "Run the Slack Events API receiver (auto-ingest new messages)",
		Long: `Run an HTTP receiver for the Slack Events API. On a new message it
re-ingests that message's thread (or channel) so the graph stays current.
Configure the app's Request URL to point at this endpoint; chiton performs the
url_verification handshake and verifies every request's X-Slack-Signature.

Set CHITON_SLACK_SIGNING_SECRET, or pass --signing-secret-file with the path of
a file that holds it. Bind to localhost behind a tunnel/reverse proxy; Slack
requires a public HTTPS Request URL.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if err := secretFlagRemoved(cmd, "signing-secret", "signing-secret-file", "CHITON_SLACK_SIGNING_SECRET"); err != nil {
				return err
			}
			signingSecret, err := readSecret(secretFile, "CHITON_SLACK_SIGNING_SECRET")
			if err != nil {
				return err
			}
			if signingSecret == "" {
				// Fail fast with a clear message rather than starting a server
				// that rejects every (unauthenticatable) request.
				return fmt.Errorf("a Slack signing secret is required: set CHITON_SLACK_SIGNING_SECRET or pass --signing-secret-file <path>")
			}
			warnIfNonLoopback(cmd.ErrOrStderr(), addr, "the Slack events receiver")
			src, err := slack.NewWebClientFromEnv()
			if err != nil {
				return err
			}
			_, gs, err := loadStore(false)
			if err != nil {
				return err
			}
			defer gs.Close()

			ing := slack.NewIngester(gs, src)
			out := cmd.OutOrStdout()
			handler := slack.NewEventsHandler(signingSecret, func(ctx context.Context, ev slack.MessageEvent) error {
				// A threaded reply re-ingests only its thread, and a top-level message the whole channel.
				// Re-ingest replaces the conversation by its slack:// URI, so this is idempotent.
				threadTS := ""
				if ev.ThreadTS != "" && ev.ThreadTS != ev.TS {
					threadTS = ev.ThreadTS
				}
				rep, err := ing.IngestConversation(ctx, ev.ChannelID, threadTS)
				if err != nil {
					return err
				}
				fmt.Fprintf(out, "slack events: ingested %s — %d message(s), %d turns\n",
					slackScope(rep), rep.Messages, rep.Turns)
				return nil
			}, errLogf(cmd))

			mux := http.NewServeMux()
			mux.Handle("/slack/events", handler)
			srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second, BaseContext: baseContext(cmd.Context())}

			fmt.Fprintf(out, "slack: events receiver listening on %s/slack/events (Ctrl-C to stop)\n", addr)
			return serveUntilDone(cmd.Context(), srv.ListenAndServe, srv.Shutdown)
		},
	}
	cmd.Flags().StringVar(&addr, "addr", "127.0.0.1:8445", "HTTP listen address (loopback by default; put a tunnel or reverse proxy in front)")
	cmd.Flags().StringVar(&secretFile, "signing-secret-file", "", "file holding the Slack app signing secret (default: CHITON_SLACK_SIGNING_SECRET)")
	cmd.Flags().String("signing-secret", "", "removed: use --signing-secret-file or CHITON_SLACK_SIGNING_SECRET")
	_ = cmd.Flags().MarkHidden("signing-secret")
	return cmd
}
```

`cmd/chiton/teams.go`: remove `"os"` from the imports and replace `teamsWebhookCmd` (lines 71-133):

```go
func teamsWebhookCmd() *cobra.Command {
	var addr, stateFile string

	cmd := &cobra.Command{
		Use:   "serve-webhook",
		Short: "Run the Graph change-notification webhook receiver (auto-ingest new transcripts)",
		Long: `Run an HTTP receiver for Microsoft Graph change notifications. On a
transcript-created notification it fetches and ingests that meeting's
transcripts. Register a Graph subscription pointing at this endpoint with a
clientState matching CHITON_TEAMS_CLIENT_STATE (or the contents of the file
named by --client-state-file).

Bind to localhost behind a tunnel/reverse proxy; Graph requires HTTPS and a
public URL for the subscription itself.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if err := secretFlagRemoved(cmd, "client-state", "client-state-file", "CHITON_TEAMS_CLIENT_STATE"); err != nil {
				return err
			}
			clientState, err := readSecret(stateFile, "CHITON_TEAMS_CLIENT_STATE")
			if err != nil {
				return err
			}
			if clientState == "" {
				// Fail fast rather than starting a receiver that cannot
				// authenticate anything. Matches `slack serve-events`.
				return fmt.Errorf("a Graph subscription clientState is required: set CHITON_TEAMS_CLIENT_STATE or pass --client-state-file <path>")
			}
			warnIfNonLoopback(cmd.ErrOrStderr(), addr, "the Teams webhook receiver")
			src, err := teams.NewGraphSourceFromEnv()
			if err != nil {
				return err
			}
			_, gs, err := loadStore(false)
			if err != nil {
				return err
			}
			defer gs.Close()

			ing := teams.NewIngester(gs, src)
			out := cmd.OutOrStdout()
			logf := errLogf(cmd)
			handler := teams.NewWebhookHandler(clientState, func(ctx context.Context, c teams.ChangeItem) error {
				meetingID, _, ok := teams.ParseTranscriptResource(c.Resource)
				if !ok {
					logf("teams webhook: ignoring non-transcript resource %q", c.Resource)
					return nil
				}
				rep, err := ing.IngestMeeting(ctx, meetingID)
				if err != nil {
					return err
				}
				fmt.Fprintf(out, "teams webhook: ingested meeting %s — %d transcript(s), %d turns\n",
					meetingID, rep.Transcripts, rep.Turns)
				return nil
			}, logf)

			mux := http.NewServeMux()
			mux.Handle("/notifications", handler)
			srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second, BaseContext: baseContext(cmd.Context())}

			fmt.Fprintf(out, "teams: webhook listening on %s/notifications (Ctrl-C to stop)\n", addr)
			return serveUntilDone(cmd.Context(), srv.ListenAndServe, srv.Shutdown)
		},
	}
	cmd.Flags().StringVar(&addr, "addr", "127.0.0.1:8444", "HTTP listen address (loopback by default; put a tunnel or reverse proxy in front)")
	cmd.Flags().StringVar(&stateFile, "client-state-file", "", "file holding the shared secret that validates notifications (default: CHITON_TEAMS_CLIENT_STATE)")
	cmd.Flags().String("client-state", "", "removed: use --client-state-file or CHITON_TEAMS_CLIENT_STATE")
	_ = cmd.Flags().MarkHidden("client-state")
	return cmd
}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./cmd/chiton/... -count=1; go vet ./cmd/chiton/...; go vet -tags msgraph ./cmd/chiton/...`
Expected: PASS

- [ ] **Step 5: Docs**

- `docs/SECURITY.md:375-376` (§4 intro): "Every secret chiton uses is read from the environment, or for the two webhook secrets from a file you name with `--signing-secret-file` / `--client-state-file`. No secret is accepted as a flag value, where `ps` and shell history would record it."
- `docs/SECURITY.md:382` and `:385` (table rows): add "(or `--signing-secret-file`)" and "(or `--client-state-file`)" to the Used-by column.
- `docs/INSTALL.md:280`: replace "(or `--signing-secret`)" with "(or a file named by `--signing-secret-file`)". `docs/INSTALL.md:292`: replace "(or `--client-state`)" with "(or a file named by `--client-state-file`)".
- `CHANGELOG.md` under `### Security`: "- **Webhook secrets no longer go on the command line.** `--signing-secret` and `--client-state` are removed and now fail with a pointer to `--signing-secret-file` / `--client-state-file`, which read the secret from a file. The `CHITON_SLACK_SIGNING_SECRET` and `CHITON_TEAMS_CLIENT_STATE` env vars work as before."

- [ ] **Step 6: Commit**

```powershell
git add cmd/chiton/secret.go cmd/chiton/secret_test.go cmd/chiton/slack.go cmd/chiton/teams.go docs/SECURITY.md docs/INSTALL.md CHANGELOG.md
git commit -m "fix(cli): read webhook secrets from a file instead of a flag value"
```

---

## Section F: Installer, ingest sources, embeddings and reports

This section hardens the places where chiton writes outside its own store (assistant config files, exported HTML) or takes in untrusted input (browser history, Slack and Teams webhooks), and makes stored embeddings self-describing so they can't silently go stale. F1, F2, F3 and F6 are independent of each other and of every other section. F4 changes the `v:` embedding value format in a backward-compatible way (legacy values still decode, so there is no migration and no dependency on Section B's schema-version work). F5 depends on F4, because pruning the vectors of newly non-embeddable stub nodes goes through F4's `embed.Plan`.

### Task F1: Keep config file modes and write through symlinks in the installer

**Finding:** `saveDoc` writes every MCP config through `fsutil.WriteFileAtomic(path, out, 0o644)` (`internal/install/fsutil.go:56`). That function creates a temp file with the given mode and renames it over `path` (`internal/fsutil/fsutil.go:14-16,49-53`), so every install or uninstall resets the config to 0644 minus the umask and swaps a symlink for a regular file. The affected files are `~/.codex/config.toml` (always user scope, `codex.go:21`), `~/.copilot/mcp-config.json` (`copilot_cli.go:40-46`), and `~/.gemini/config/mcp_config.json` (`antigravity.go:21`). Under `--user-scope` it also hits `~/.cursor/mcp.json` (`cursor.go:22-23`), `~/.gemini/settings.json` (`gemini_cli.go:19-20`) and `~/.config/opencode/opencode.json` (`opencode.go:19-20`). These files often hold other MCP servers' API keys, so a 0600 file becomes world-readable. The rules writer has the same problem (`rules_block.go:44,71`), and a `CLAUDE.md → AGENTS.md` symlink gets replaced. A second bug: when uninstall empties a symlinked config, `saveOrRemove` (`fsutil.go:63-71`) and `RemoveRulesBlock` (`rules_block.go:65-69`) call `os.Remove` on the **link**. That leaves the target file with the chiton entry still inside it. (The Claude Code installer is project-scope only, `claude.go:38-39`, but it shares the same helpers.)
**Priority:** P2   **Effort:** S   **Depends on:** none
**Behaviour change:** An existing config keeps its exact permissions. A symlinked config or rules file is updated in place: the link survives, and a dangling link is now an error instead of being silently replaced. Uninstalling from a symlinked file empties the target instead of deleting the link. New MCP configs are created 0600, and new rules files stay 0644. DECISION NEEDED: the plan creates **every** new MCP config 0600, including project-scope `.mcp.json`, `.cursor/mcp.json`, `.vscode/mcp.json`, `opencode.json` and `.gemini/settings.json` (git records no mode bits except exec). The alternative is 0600 only for paths under `$HOME` dot-directories and 0644 in the repo.

**Files:**
- Modify: `internal/install/fsutil.go:50-71` (saveDoc, saveOrRemove; add mcpConfigPerm, writeKeepingMode, resolveTarget, isSymlink)
- Modify: `internal/install/mcp_config.go:81,101`
- Modify: `internal/install/aider_config.go:26,50`
- Modify: `internal/install/rules_block.go:3-9,44,65,71`
- Create: `internal/install/fsutil_test.go`

**Interfaces:**
- Consumes: `fsutil.WriteFileAtomic(path string, data []byte, perm os.FileMode) error` (existing)
- Produces: none exported (package-internal `writeKeepingMode(path string, data []byte, newPerm os.FileMode) error`)

- [ ] **Step 1: Write the failing test**

Create `internal/install/fsutil_test.go`:

```go
package install

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestConfigWritesKeepExistingMode checks install and uninstall keep a config's permissions, since
// ~/.codex/config.toml and friends often hold other MCP servers' API keys.
func TestConfigWritesKeepExistingMode(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix permission bits are not meaningful on Windows")
	}
	dir := t.TempDir()
	cfg := filepath.Join(dir, "mcp.json")
	if err := os.WriteFile(cfg, []byte(`{"mcpServers":{"other":{"command":"x","env":{"API_KEY":"k"}}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(cfg, 0o640); err != nil { // explicit chmod, so the umask can't mask the check
		t.Fatal(err)
	}
	if _, _, err := mcpServersTable.write(cfg, ChitonServerEntry("/abs/chiton", "")); err != nil {
		t.Fatal(err)
	}
	if got := permOf(t, cfg); got != 0o640 {
		t.Errorf("after install mode = %v, want 0640 kept", got)
	}
	if _, err := mcpServersTable.remove(cfg); err != nil {
		t.Fatal(err)
	}
	if got := permOf(t, cfg); got != 0o640 {
		t.Errorf("after uninstall mode = %v, want 0640 kept", got)
	}

	rules := filepath.Join(dir, "AGENTS.md")
	if err := os.WriteFile(rules, []byte("# mine\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(rules, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := UpsertRulesBlock(rules, "hello"); err != nil {
		t.Fatal(err)
	}
	if got := permOf(t, rules); got != 0o600 {
		t.Errorf("rules file mode = %v, want 0600 kept", got)
	}
}

// A new MCP config starts owner-only.
func TestNewMCPConfigIsOwnerOnly(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix permission bits are not meaningful on Windows")
	}
	cfg := filepath.Join(t.TempDir(), ".codex", "config.toml")
	if _, _, err := codexTable.write(cfg, chitonCodexEntry("/abs/chiton", "")); err != nil {
		t.Fatal(err)
	}
	if got := permOf(t, cfg); got&0o077 != 0 {
		t.Errorf("new config mode = %v, want no group/other bits", got)
	}
}

// TestConfigWritesFollowSymlinks checks a symlinked config (a dotfiles manager's layout) is updated in place:
// the link survives and the file it names gets the change, on install and on an emptying uninstall.
func TestConfigWritesFollowSymlinks(t *testing.T) {
	dir := t.TempDir()
	dotfiles := filepath.Join(dir, "dotfiles")
	if err := os.MkdirAll(dotfiles, 0o755); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(dotfiles, "config.toml")
	if err := os.WriteFile(target, []byte("model = \"o3\"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "config.toml")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("cannot create symlinks here: %v", err)
	}
	if _, _, err := codexTable.write(link, chitonCodexEntry("/abs/chiton", "")); err != nil {
		t.Fatal(err)
	}
	assertSymlink(t, link)
	if has, _ := codexTable.has(target); !has {
		t.Fatal("chiton entry not written to the symlink target")
	}
	if readTOML(t, target)["model"] != "o3" {
		t.Error("the target's other settings were lost")
	}

	// A chiton-only file behind a link: uninstall must empty the target, not delete the link and strand it.
	only := filepath.Join(dotfiles, "mcp.json")
	if err := os.WriteFile(only, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	onlyLink := filepath.Join(dir, "mcp.json")
	if err := os.Symlink(only, onlyLink); err != nil {
		t.Fatal(err)
	}
	if _, _, err := mcpServersTable.write(onlyLink, ChitonServerEntry("/abs/chiton", "")); err != nil {
		t.Fatal(err)
	}
	if _, err := mcpServersTable.remove(onlyLink); err != nil {
		t.Fatal(err)
	}
	assertSymlink(t, onlyLink)
	if has, _ := mcpServersTable.has(only); has {
		t.Error("uninstall left the chiton entry in the symlink target")
	}

	// CLAUDE.md -> AGENTS.md is a common layout; the rules block must land in AGENTS.md.
	agents := filepath.Join(dir, "AGENTS.md")
	if err := os.WriteFile(agents, []byte("# shared\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	claude := filepath.Join(dir, "CLAUDE.md")
	if err := os.Symlink(agents, claude); err != nil {
		t.Fatal(err)
	}
	if _, _, err := UpsertRulesBlock(claude, "hello"); err != nil {
		t.Fatal(err)
	}
	assertSymlink(t, claude)
	if has, _ := RulesFileHasChiton(agents); !has {
		t.Error("rules block not written through the CLAUDE.md symlink")
	}
}

func TestConfigWriteRejectsDanglingSymlink(t *testing.T) {
	dir := t.TempDir()
	link := filepath.Join(dir, "mcp.json")
	if err := os.Symlink(filepath.Join(dir, "missing.json"), link); err != nil {
		t.Skipf("cannot create symlinks here: %v", err)
	}
	_, _, err := mcpServersTable.write(link, ChitonServerEntry("/abs/chiton", ""))
	if err == nil || !strings.Contains(err.Error(), "symlink") {
		t.Fatalf("write through a dangling symlink: err = %v, want a symlink error", err)
	}
	assertSymlink(t, link)
}

func permOf(t *testing.T, path string) os.FileMode {
	t.Helper()
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return fi.Mode().Perm()
}

func assertSymlink(t *testing.T, path string) {
	t.Helper()
	fi, err := os.Lstat(path)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("%s was replaced by a regular file", path)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/install/ -run 'TestConfigWrites|TestNewMCPConfig|TestConfigWriteRejects' -count=1`
Expected: FAIL on Linux/macOS (CI): `after install mode = -rw-r--r--, want 0640 kept`, `new config mode = -rw-r--r--`, `config.toml was replaced by a regular file`, and a nil error from the dangling-link write. On Windows without Developer Mode, every case skips, so check the failure in CI or WSL.

- [ ] **Step 3: Implement**

In `internal/install/fsutil.go`, replace `saveDoc` and `saveOrRemove` (lines 50-71) and add the helpers below them:

```go
// mcpConfigPerm is the mode a new MCP config gets. These files often hold other servers' API keys in env
// blocks, so they start owner-only.
const mcpConfigPerm os.FileMode = 0o600

// saveDoc marshals doc and atomically replaces path's contents with it (see writeKeepingMode).
func saveDoc(path string, doc map[string]any, marshal func(any) ([]byte, error), newPerm os.FileMode) error {
	out, err := marshal(doc)
	if err != nil {
		return fmt.Errorf("marshal %s: %w", path, err)
	}
	if err := writeKeepingMode(path, out, newPerm); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

// saveOrRemove saves doc to path, or deletes path instead when empty. A symlink is saved through rather than
// deleted, so the file it names never keeps a stale chiton entry.
func saveOrRemove(path string, doc map[string]any, empty bool, marshal func(any) ([]byte, error), newPerm os.FileMode) error {
	if !empty || isSymlink(path) {
		return saveDoc(path, doc, marshal, newPerm)
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove empty %s: %w", path, err)
	}
	return nil
}

// writeKeepingMode atomically replaces the file path names, following a symlink so the link survives. An
// existing file keeps its exact permissions; a new one gets newPerm minus the umask.
func writeKeepingMode(path string, data []byte, newPerm os.FileMode) error {
	target, err := resolveTarget(path)
	if err != nil {
		return err
	}
	fi, err := os.Stat(target)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	perm := newPerm
	if fi != nil {
		perm = fi.Mode().Perm()
	}
	if err := fsutil.WriteFileAtomic(target, data, perm); err != nil {
		return err
	}
	if fi == nil {
		return nil
	}
	return os.Chmod(target, perm) // WriteFileAtomic applied the umask; restore the file's own mode
}

// resolveTarget returns the file path names after following symlinks. A missing path resolves to itself; a
// dangling symlink is an error rather than a silent replacement.
func resolveTarget(path string) (string, error) {
	if !isSymlink(path) {
		return path, nil
	}
	target, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", fmt.Errorf("resolve symlink %s: %w", path, err)
	}
	return target, nil
}

// isSymlink reports whether path itself is a symbolic link.
func isSymlink(path string) bool {
	fi, err := os.Lstat(path)
	return err == nil && fi.Mode()&os.ModeSymlink != 0
}
```

`internal/install/mcp_config.go:81` becomes `if err := saveDoc(path, doc, t.marshal, mcpConfigPerm); err != nil {`, and line 101 becomes:

```go
	return true, saveOrRemove(path, doc, len(doc) == 0 || (t.shell != nil && t.shell(doc)), t.marshal, mcpConfigPerm)
```

`internal/install/aider_config.go:26` becomes `if err := saveDoc(path, doc, yaml.Marshal, 0o644); err != nil {`, and line 50 becomes `return true, saveOrRemove(path, doc, len(doc) == 0, yaml.Marshal, 0o644)`.

`internal/install/rules_block.go`: delete the `"github.com/Eric-Fernald/chiton/internal/fsutil"` import, then change lines 44, 65 and 71:

```go
	if err := writeKeepingMode(path, []byte(next), 0o644); err != nil { // line 44 (UpsertRulesBlock)
```

```go
	next := stripBlock(text)
	if strings.TrimSpace(next) == "" && !isSymlink(path) { // line 65: a symlinked rules file is emptied, not unlinked
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return true, fmt.Errorf("remove %s: %w", path, err)
		}
		return true, nil
	}
	if err := writeKeepingMode(path, []byte(next), 0o644); err != nil { // line 71
		return true, fmt.Errorf("write %s: %w", path, err)
	}
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/install/... -count=1; go vet ./internal/install/...`
Expected: PASS. The golden and installer tests don't depend on file modes.

- [ ] **Step 5: Docs**

- `docs/INSTALL.md`, "Per-assistant matrix" intro (lines 198-200): append "Config writes keep an existing file's permissions and write through a symlink, so the link survives and a dangling link is an error. A newly created MCP config is owner-only (`0600`) because these files often hold other servers' API keys."
- `CHANGELOG.md` → `### Security`: "`chiton install`/`uninstall` no longer reset MCP config files (e.g. `~/.codex/config.toml`) to 0644 or replace a symlinked config with a regular file. Existing modes are kept, symlinks are written through, and new MCP configs are created 0600."

- [ ] **Step 6: Commit**

```powershell
git add internal/install/fsutil.go internal/install/fsutil_test.go internal/install/mcp_config.go internal/install/aider_config.go internal/install/rules_block.go docs/INSTALL.md CHANGELOG.md
git commit -m "fix(install): keep config file modes and write through symlinks"
```

### Task F2: Strip secret-bearing URL parts from browser history

**Finding:** `IngestDB` stores each full history URL three ways: as `attrs.url`, inside `source_uri` (`"browser://" + v.URL`), and as the label when the title is empty (`internal/ingest/browser/browser.go:78-91`). The node ID also derives from it (`browser.go:99`), and `cluster.json` copies label and source_uri in plaintext (`internal/graph/cluster/sidecar.go:25-26`). That puts OAuth `?code=`, `#access_token=`, password-reset and magic-link tokens, presigned-URL signatures and Jupyter `?token=` into every MCP read tool.
**Priority:** P2   **Effort:** M   **Depends on:** none
**Behaviour change:** By default a stored page URL is `scheme://host/path` with no userinfo, query or fragment. `--keep-query` on `chiton browser ingest|watch` keeps the query and fragment, but userinfo is always dropped. History rows that now share a URL merge into one node: the newest row supplies the title and `last_visit`, and `visit_count` is the **sum** of all rows. Node IDs change for URLs that had a query. On an Ingester's first run without `--keep-query`, nodes an earlier chiton stored under an unredacted URL are deleted, so the secrets leave the store. `cluster.json` keeps their labels until the next `chiton cluster`. `--min-visits` still filters raw rows before they merge. DECISION NEEDED: redaction on by default plus an automatic one-time purge (recommended). The alternative is redaction on by default with the purge only via an explicit flag.

**Files:**
- Modify: `internal/ingest/browser/browser.go:3-12` (imports), `:28-52` (Ingester, Options, Report), `:54-117` (IngestDB), `:136-149` (isWebURL → parseWebURL; add redactURL, historyNodes, purgeUnredacted)
- Modify: `internal/graph/store/helpers.go` (add `DeleteNodes`)
- Modify: `cmd/chiton/browser.go:31-121`
- Test: `internal/ingest/browser/browser_test.go`, `internal/graph/store/helpers_test.go`

**Interfaces:**
- Consumes: `store.NodesMatching`, `store.NodeFilter`, `store.NodeDeleter` (existing)
- Produces: `browser.Options.KeepQuery bool`; `browser.Report.Purged int`; `store.DeleteNodes(ctx context.Context, gs store.GraphStore, nodes []model.Node) (int, error)`

- [ ] **Step 1: Write the failing test**

Append to `internal/ingest/browser/browser_test.go`, adding `"bytes"` and `"strings"` to its imports:

```go
// TestIngestDBRedactsAndMergesPages checks tokens in the query, fragment or userinfo never reach the store, and
// rows that then share a URL merge: newest row's title and last visit, summed visit counts.
func TestIngestDBRedactsAndMergesPages(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "History")
	recent := time.Date(2024, 1, 2, 3, 4, 5, 0, time.UTC)
	writeHistoryDB(t, dbPath, []Visit{
		{URL: "https://app.example.com/callback?code=SECRET1&state=x", Title: "Signed in", VisitCount: 1, LastVisit: recent},
		{URL: "https://app.example.com/callback?code=SECRET2", Title: "Old", VisitCount: 2, LastVisit: recent.Add(-time.Hour)},
		{URL: "https://user:pw@example.org/p#access_token=SECRET3", Title: "", VisitCount: 1, LastVisit: recent},
	})
	gs := newStore(t)
	ctx := context.Background()
	rep, err := New(gs).IngestDB(ctx, dbPath, "chrome", Options{})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Visits != 3 || rep.NodesUpserted != 2 {
		t.Errorf("report = %+v, want 3 rows merged into 2 pages", rep)
	}

	var snap bytes.Buffer
	if err := gs.Snapshot(ctx, &snap); err != nil {
		t.Fatal(err)
	}
	for _, leak := range []string{"SECRET1", "SECRET2", "SECRET3", "pw@", "code="} {
		if strings.Contains(snap.String(), leak) {
			t.Errorf("stored graph still contains %q", leak)
		}
	}

	nodes, _ := gs.AllNodes(ctx)
	byURI := map[string]model.Node{}
	for _, n := range nodes {
		byURI[n.SourceURI] = n
	}
	cb, ok := byURI["browser://https://app.example.com/callback"]
	if !ok {
		t.Fatalf("no merged callback node in %v", byURI)
	}
	if vc, _ := cb.Attrs["visit_count"].(float64); vc != 3 {
		t.Errorf("visit_count = %v, want 3 (summed across rows)", cb.Attrs["visit_count"])
	}
	if cb.Label != "Signed in" || cb.Attrs["last_visit"] != recent.Format(time.RFC3339) {
		t.Errorf("label=%q last_visit=%v, want the newest row's", cb.Label, cb.Attrs["last_visit"])
	}
	if p := byURI["browser://https://example.org/p"]; p.Label != "https://example.org/p" || p.Attrs["url"] != "https://example.org/p" {
		t.Errorf("untitled page label=%q url=%v, want the redacted URL", p.Label, p.Attrs["url"])
	}
}

func TestIngestDBKeepQuery(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "History")
	writeHistoryDB(t, dbPath, []Visit{
		{URL: "https://u:pw@example.com/search?q=go#r", Title: "Search", VisitCount: 1, LastVisit: time.Now().UTC()},
	})
	gs := newStore(t)
	if _, err := New(gs).IngestDB(context.Background(), dbPath, "chrome", Options{KeepQuery: true}); err != nil {
		t.Fatal(err)
	}
	nodes, _ := gs.AllNodes(context.Background())
	if len(nodes) != 1 || nodes[0].SourceURI != "browser://https://example.com/search?q=go#r" {
		t.Fatalf("nodes = %+v, want query and fragment kept, userinfo dropped", nodes)
	}
}

// TestIngestDBPurgesUnredactedNodes checks nodes an earlier chiton stored under a full URL are deleted, while
// clean history nodes and non-history url nodes stay.
func TestIngestDBPurgesUnredactedNodes(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	legacy := func(raw string) model.Node {
		uri := "browser://" + raw
		return model.Node{ID: model.NewNodeID(uri, raw), Kind: model.KindURL, Label: raw, SourceURI: uri,
			Attrs: map[string]any{"source": "browser", "url": raw}}
	}
	secret, clean := legacy("https://x.example/reset?token=SECRET"), legacy("https://x.example/docs")
	web := model.Node{ID: "web", Kind: model.KindURL, Label: "page", SourceURI: "https://x.example/p?q=1"}
	if err := gs.UpsertNodes(ctx, []model.Node{secret, clean, web}); err != nil {
		t.Fatal(err)
	}
	dbPath := filepath.Join(t.TempDir(), "History")
	writeHistoryDB(t, dbPath, nil)

	rep, err := New(gs).IngestDB(ctx, dbPath, "chrome", Options{})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Purged != 1 {
		t.Errorf("Purged = %d, want 1", rep.Purged)
	}
	for id, want := range map[string]bool{secret.ID: false, clean.ID: true, web.ID: true} {
		if n, _ := gs.GetNode(ctx, id); (n != nil) != want {
			t.Errorf("node %s present=%v, want %v", id, n != nil, want)
		}
	}
}
```

In the same file, rename `TestIsWebURL` (lines 207-220) to `TestParseWebURL`, and replace its two `isWebURL(u)` calls with `_, ok := parseWebURL(u)` checks:

```go
func TestParseWebURL(t *testing.T) {
	web := []string{"https://example.com", "http://x.org/path"}
	notWeb := []string{"chrome://settings", "about:blank", "file:///c:/x.txt", "chrome-extension://abc/page.html", "javascript:void(0)", ""}
	for _, u := range web {
		if _, ok := parseWebURL(u); !ok {
			t.Errorf("parseWebURL(%q) ok=false want true", u)
		}
	}
	for _, u := range notWeb {
		if _, ok := parseWebURL(u); ok {
			t.Errorf("parseWebURL(%q) ok=true want false", u)
		}
	}
}
```

Append to `internal/graph/store/helpers_test.go`:

```go
func TestDeleteNodesFallbackAgrees(t *testing.T) {
	ctx := context.Background()
	for name, wrap := range map[string]func(*badger.Store) store.GraphStore{
		"deleter":  func(s *badger.Store) store.GraphStore { return s },
		"fallback": func(s *badger.Store) store.GraphStore { return plain{s} },
	} {
		gs := openStore(t)
		nodes := []model.Node{{ID: "1", SourceURI: "u://1"}, {ID: "2", SourceURI: "u://2"}, {ID: "3", SourceURI: "u://3"}}
		if err := gs.UpsertNodes(ctx, nodes); err != nil {
			t.Fatal(err)
		}
		if n, err := store.DeleteNodes(ctx, wrap(gs), nodes[:2]); err != nil || n != 2 {
			t.Errorf("%s DeleteNodes = %d, %v; want 2, nil", name, n, err)
		}
		if left, _ := gs.AllNodes(ctx); len(left) != 1 || left[0].ID != "3" {
			t.Errorf("%s left %v, want only node 3", name, left)
		}
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/ingest/browser/ ./internal/graph/store/ -count=1`
Expected: FAIL. The build breaks with `unknown field KeepQuery in struct literal`, `rep.Purged undefined`, `undefined: parseWebURL` and `undefined: store.DeleteNodes`.

- [ ] **Step 3: Implement**

Append to `internal/graph/store/helpers.go`:

```go
// DeleteNodes removes nodes and their edges by ID via NodeDeleter when gs has it, else by each node's source_uri
// (which also removes any other node sharing that URI). It returns how many nodes were deleted.
func DeleteNodes(ctx context.Context, gs GraphStore, nodes []model.Node) (int, error) {
	if len(nodes) == 0 {
		return 0, nil
	}
	if d, ok := gs.(NodeDeleter); ok {
		ids := make([]string, len(nodes))
		for i, n := range nodes {
			ids[i] = n.ID
		}
		return d.DeleteByIDs(ctx, ids)
	}
	total := 0
	for _, n := range nodes {
		k, err := gs.DeleteBySourceURI(ctx, n.SourceURI)
		total += k
		if err != nil {
			return total, err
		}
	}
	return total, nil
}
```

In `internal/ingest/browser/browser.go`, add `"cmp"` and `"sync"` to the imports and replace the `Ingester`, `Options` and `Report` types (lines 28-52):

```go
// Ingester turns browser history into graph `url` nodes.
type Ingester struct {
	store store.GraphStore
	now   func() time.Time

	mu     sync.Mutex
	purged bool // purgeUnredacted has run for this Ingester
}

// Options bounds and filters an ingest run.
type Options struct {
	Limit     int       // max rows per DB; 0 → defaultLimit
	Since     time.Time // only visits at/after this time
	MinVisits int       // skip history rows visited fewer than this many times (applied before rows merge)
	KeepQuery bool      // keep query strings and #fragments, which can carry OAuth codes and signed-URL tokens
}

// Report summarises one History database's ingest.
type Report struct {
	Browser       string `json:"browser"`
	DB            string `json:"db"`
	Visits        int    `json:"visits"`           // web history rows read
	NodesUpserted int    `json:"nodes_upserted"`   // pages, after rows sharing a redacted URL merge
	Purged        int    `json:"purged,omitempty"` // nodes an earlier ingest stored under an unredacted URL
}
```

Replace `IngestDB` (lines 54-117):

```go
// IngestDB reads one History database and upserts a url node per visited web page, skipping non-web URLs.
// Unless opts.KeepQuery, URLs lose userinfo, query and fragment, and rows that then match merge into one node.
// IDs derive from that URL, so a re-run refreshes visit counts and titles in place without duplicates.
func (ing *Ingester) IngestDB(ctx context.Context, dbPath, browser string, opts Options) (Report, error) {
	if store.IsReadOnly(ing.store) {
		return Report{}, fmt.Errorf("browser: store is read-only — use the 'chiton browser ingest' CLI or restart 'chiton serve --writable'")
	}
	if opts.Limit <= 0 {
		opts.Limit = defaultLimit
	}
	visits, err := readHistory(ctx, dbPath, readOptions{
		limit:     opts.Limit,
		since:     opts.Since,
		minVisits: opts.MinVisits,
	})
	if err != nil {
		return Report{}, err
	}
	purged, err := ing.purgeUnredacted(ctx, opts.KeepQuery)
	if err != nil {
		return Report{}, fmt.Errorf("purge unredacted history nodes: %w", err)
	}
	nodes, rows := historyNodes(visits, browser, opts.KeepQuery, ing.now())
	if len(nodes) > 0 {
		if err := ing.store.UpsertNodes(ctx, nodes); err != nil {
			return Report{}, fmt.Errorf("upsert history nodes: %w", err)
		}
	}
	return Report{Browser: browser, DB: dbPath, Visits: rows, NodesUpserted: len(nodes), Purged: purged}, nil
}

// historyNodes builds one url node per page URL and counts the web rows. Rows arrive newest-first, so a page's
// first row supplies its title and last visit, and every row adds its visit count.
func historyNodes(visits []Visit, browser string, keepQuery bool, now time.Time) (nodes []model.Node, rows int) {
	index := map[string]int{}
	for _, v := range visits {
		u, ok := parseWebURL(v.URL)
		if !ok {
			continue
		}
		rows++
		page := redactURL(u, keepQuery)
		if i, seen := index[page]; seen {
			nodes[i].Attrs["visit_count"] = nodes[i].Attrs["visit_count"].(int) + v.VisitCount
			continue
		}
		index[page] = len(nodes)
		// Key on the stable URL, not the mutable title, so a re-ingest after a title change refreshes the same node.
		uri := historyURIScheme + page
		attrs := map[string]any{
			"kind":        "history",
			"source":      "browser",
			"url":         page,
			"title":       v.Title,
			"visit_count": v.VisitCount,
		}
		if browser != "" {
			attrs["browser"] = browser
		}
		if !v.LastVisit.IsZero() {
			attrs["last_visit"] = v.LastVisit.UTC().Format(time.RFC3339)
		}
		nodes = append(nodes, model.Node{
			ID:         model.NewNodeID(uri, page),
			Kind:       model.KindURL,
			Label:      cmp.Or(strings.TrimSpace(v.Title), page),
			SourceURI:  uri,
			Attrs:      attrs,
			Provenance: model.ProvExtracted,
			Confidence: 1.0,
			CreatedAt:  now,
			UpdatedAt:  now,
		})
	}
	return nodes, rows
}

// purgeUnredacted deletes, on this Ingester's first redacting run, history nodes an earlier chiton stored under a
// URL with userinfo, a query or a fragment, so their secrets leave the graph.
func (ing *Ingester) purgeUnredacted(ctx context.Context, keepQuery bool) (int, error) {
	ing.mu.Lock()
	defer ing.mu.Unlock()
	if keepQuery || ing.purged {
		return 0, nil
	}
	nodes, err := store.NodesMatching(ctx, ing.store, store.NodeFilter{
		Kinds: []model.NodeKind{model.KindURL},
		Attrs: map[string]string{"source": "browser"},
	})
	if err != nil {
		return 0, err
	}
	var stale []model.Node
	for _, n := range nodes {
		raw, ok := strings.CutPrefix(n.SourceURI, historyURIScheme)
		if !ok {
			continue
		}
		if u, ok := parseWebURL(raw); ok && redactURL(u, false) != raw {
			stale = append(stale, n)
		}
	}
	deleted, err := store.DeleteNodes(ctx, ing.store, stale)
	if err == nil {
		ing.purged = true
	}
	return deleted, err
}
```

Replace `isWebURL` (lines 136-149):

```go
// parseWebURL parses raw and reports whether it is an http(s) URL with a host, excluding the browser-internal
// schemes (chrome://, about:, file://, chrome-extension://...).
func parseWebURL(raw string) (*url.URL, bool) {
	u, err := url.Parse(raw)
	if err != nil {
		return nil, false
	}
	switch strings.ToLower(u.Scheme) {
	case "http", "https":
		return u, u.Host != ""
	default:
		return nil, false
	}
}

// redactURL renders u without userinfo and, unless keepQuery, without query or fragment. History URLs carry
// OAuth codes, reset tokens and signed-URL signatures there.
func redactURL(u *url.URL, keepQuery bool) string {
	c := *u
	c.User = nil
	if !keepQuery {
		c.RawQuery, c.ForceQuery, c.Fragment, c.RawFragment = "", false, "", ""
	}
	return c.String()
}
```

In `cmd/chiton/browser.go`:
- `browserIngestCmd` (lines 31-81): add `var keepQuery bool` after line 33 and replace lines 49-72 of `RunE` with:

```go
			ing := browser.New(gs)
			opts := browser.Options{Limit: limit, Since: sinceT, MinVisits: minVisits, KeepQuery: keepQuery}
			out := cmd.OutOrStdout()

			if dbPath != "" {
				rep, err := ing.IngestDB(cmd.Context(), dbPath, name, opts)
				if err != nil {
					return err
				}
				fmt.Fprintf(out, "browser: ingested %d pages from %s\n", rep.NodesUpserted, rep.DB)
				purgeNote(out, rep.Purged)
				return nil
			}

			reports, err := ing.IngestAuto(cmd.Context(), opts)
			if err != nil {
				return err
			}
			total, purged := 0, 0
			for _, r := range reports {
				fmt.Fprintf(out, "browser: %s — %d pages (%s)\n", r.Browser, r.NodesUpserted, r.DB)
				total += r.NodesUpserted
				purged += r.Purged
			}
			fmt.Fprintf(out, "browser: ingested %d pages from %d database(s)\n", total, len(reports))
			purgeNote(out, purged)
			return nil
```

  and register the flag after `--min-visits` (line 79):

```go
	cmd.Flags().BoolVar(&keepQuery, "keep-query", false, "Keep URL query strings and #fragments (may store OAuth codes, reset tokens, signed-URL signatures)")
```

- `browserWatchCmd` (lines 83-121): add `var keepQuery bool` after line 85, change line 110 to `opts := browser.Options{Limit: limit, MinVisits: minVisits, KeepQuery: keepQuery}`, and register the same `--keep-query` flag line after `--min-visits` (line 118). `Watch` reuses one `Ingester`, so the purge runs once per watch session.
- Add below `parseSince`:

```go
// purgeNote tells the user when an ingest deleted pages an older chiton stored with their query strings.
func purgeNote(out io.Writer, purged int) {
	if purged > 0 {
		fmt.Fprintf(out, "browser: removed %d page(s) an earlier ingest stored with query strings; run 'chiton cluster' to refresh cluster.json\n", purged)
	}
}
```

(add `"io"` to the imports).

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/ingest/browser/... ./internal/graph/store/... ./cmd/chiton/... -count=1; go vet ./internal/ingest/browser/... ./internal/graph/store/... ./cmd/chiton/...`
Expected: PASS. The existing `TestIngestDB`, `TestIngestDBTitleChangeNoDuplicate`, `TestIngestDBIdempotent` and `TestMinVisitsAndSinceFilter` still pass because their URLs have no query.

- [ ] **Step 5: Docs**

- `CLAUDE.md:122-125` (Browser history bullet): after "…no page fetch, no network." insert "URLs are stored without userinfo, query string or `#fragment` (they carry OAuth codes, reset tokens and signed-URL signatures), and rows that then match merge with summed visit counts; `--keep-query` keeps query and fragment." In the code block at line 114, add `chiton browser ingest --keep-query            # keep ?query/#fragment (may store tokens)`.
- `README.md:217`: add `--keep-query` to the flag list. `README.md:409-411`: append "URLs are stored without userinfo, query or fragment unless you pass `--keep-query`."
- `docs/SECURITY.md:368-369` (Browser history): add a paragraph saying that by default stored URLs are `scheme://host/path`, with userinfo, query string and fragment removed before anything is written. The first ingest after upgrading deletes nodes an older version stored with a query, and users should run `chiton cluster` afterwards so `cluster.json` drops their labels. Tokens embedded in the **path** (e.g. `/reset/<token>`) are not detected.
- `CHANGELOG.md` → `### Security`: "`chiton browser ingest|watch` strips userinfo, query strings and fragments from history URLs (they carried OAuth codes, reset tokens and presigned signatures into every MCP read tool). Rows that then share a URL merge, and pages stored unredacted by an earlier ingest are deleted on the next run. `--keep-query` restores query and fragment."

- [ ] **Step 6: Commit**

```powershell
git add internal/ingest/browser/browser.go internal/ingest/browser/browser_test.go internal/graph/store/helpers.go internal/graph/store/helpers_test.go cmd/chiton/browser.go CLAUDE.md README.md docs/SECURITY.md CHANGELOG.md
git commit -m "fix(browser): strip userinfo, query and fragment from history URLs"
```

### Task F3: Vendor a pinned vis-network into the HTML reports

**Finding:** Both report pages load `https://unpkg.com/vis-network/standalone/umd/vis-network.min.js` with no version and no SRI (`internal/report/visual.go:272`, `internal/report/dashboard.go:124`). Their CSP allowlists `script-src https://unpkg.com` (`visual.go:270`, `dashboard.go:122`). Whatever unpkg serves under "latest" therefore runs inside a page that inlines the whole graph, including email subjects and person names. `connect-src 'none'` blocks egress, but the script can still read everything, and opening the file tells the CDN you opened it (`docs/SECURITY.md:172-192`).
**Priority:** P3   **Effort:** S   **Depends on:** none
**Behaviour change:** Reports embed vis-network 9.1.9, which adds about 0.5–0.7 MB to each HTML file and to the binary. They open offline and make no network request, and the CSP drops `https://unpkg.com`. DECISION NEEDED: vendor (recommended). The alternative is to keep the CDN but pin `vis-network@9.1.9` with an `integrity="sha384-…"` attribute, which is smaller but still tells the CDN when a report is opened.

**Files:**
- Create: `internal/report/assets/vis-network.min.js` (downloaded)
- Create: `internal/report/assets/vis-network.min.js.sha256`
- Create: `internal/report/assets/LICENSE-vis-network-MIT.txt` (downloaded)
- Create: `internal/report/assets.go`
- Create: `internal/report/assets_test.go`
- Modify: `internal/report/visual.go:212-240,257-272`
- Modify: `internal/report/dashboard.go:109-124`
- Modify: `.gitattributes`, `NOTICE.md`

**Interfaces:**
- Consumes: none
- Produces: none exported (package-internal `visNetworkJS string`)

- [ ] **Step 1: Write the failing test**

Create `internal/report/assets_test.go`:

```go
package report

import (
	"bytes"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

//go:embed assets/vis-network.min.js.sha256
var visNetworkSHA256 string

// TestVendoredVisNetworkIntact pins the bundle to the bytes recorded at download and checks it can sit inside
// an inline <script> without ending it early.
func TestVendoredVisNetworkIntact(t *testing.T) {
	if len(visNetworkJS) < 100_000 {
		t.Fatalf("vendored vis-network is %d bytes; want the ~0.5-0.7 MB standalone build", len(visNetworkJS))
	}
	sum := sha256.Sum256([]byte(visNetworkJS))
	if got, want := hex.EncodeToString(sum[:]), strings.ToLower(strings.TrimSpace(visNetworkSHA256)); got != want {
		t.Fatalf("vis-network.min.js SHA-256 = %s, want %s (re-download it, or re-record the hash after a deliberate upgrade)", got, want)
	}
	lower := strings.ToLower(visNetworkJS)
	for _, bad := range []string{"</script", "<!--"} {
		if strings.Contains(lower, bad) {
			t.Fatalf("vendored bundle contains %q, which would break out of the inline <script>", bad)
		}
	}
}

// TestPagesLoadNoRemoteScript checks both pages inline vis-network and allow no remote script source.
func TestPagesLoadNoRemoteScript(t *testing.T) {
	nodes := []model.Node{sym("a", "A"), sym("b", "B")}
	edges := []model.Edge{{ID: "e1", Src: "a", Dst: "b", Kind: model.EdgeCalls}}
	var visual, dash bytes.Buffer
	if err := GenerateHTML(&visual, nodes, edges, nil, DefaultVisual); err != nil {
		t.Fatal(err)
	}
	if err := GenerateDashboard(&dash, DashboardData{Nodes: nodes, Edges: edges}); err != nil {
		t.Fatal(err)
	}
	for name, page := range map[string]string{"visualize": visual.String(), "dashboard": dash.String()} {
		for _, bad := range []string{"unpkg.com", "<script src", "__VIS_NETWORK__"} {
			if strings.Contains(page, bad) {
				t.Errorf("%s page contains %q", name, bad)
			}
		}
		if !strings.Contains(page, "script-src 'unsafe-inline';") {
			t.Errorf("%s page CSP still allows a remote script source", name)
		}
		if !strings.Contains(page, visNetworkJS[:200]) {
			t.Errorf("%s page does not inline the vendored vis-network", name)
		}
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/report/ -run 'TestVendoredVisNetworkIntact|TestPagesLoadNoRemoteScript' -count=1`
Expected: FAIL. The build breaks with `pattern assets/vis-network.min.js.sha256: no matching files found` and `undefined: visNetworkJS`.

- [ ] **Step 3: Implement**

Download and verify the bundle. It must be byte-identical from two independent CDNs, and the registry must report the dual licence:

```powershell
(Invoke-RestMethod https://registry.npmjs.org/vis-network/9.1.9).license   # expect: (Apache-2.0 OR MIT)
New-Item -ItemType Directory -Force internal/report/assets | Out-Null
Invoke-WebRequest -UseBasicParsing https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js -OutFile internal/report/assets/vis-network.min.js
Invoke-WebRequest -UseBasicParsing https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js -OutFile "$env:TEMP\vis-network-check.js"
(Get-FileHash internal/report/assets/vis-network.min.js).Hash -eq (Get-FileHash "$env:TEMP\vis-network-check.js").Hash   # must print True
(Get-FileHash internal/report/assets/vis-network.min.js -Algorithm SHA256).Hash.ToLower() | Set-Content -NoNewline -Encoding ascii internal/report/assets/vis-network.min.js.sha256
Invoke-WebRequest -UseBasicParsing https://raw.githubusercontent.com/visjs/vis-network/v9.1.9/LICENSE-MIT -OutFile internal/report/assets/LICENSE-vis-network-MIT.txt
```

If the registry call returns 404, the version doesn't exist. Run `(Invoke-RestMethod https://registry.npmjs.org/vis-network).versions.PSObject.Properties.Name` and use the newest `9.1.x` in every command above and in the comment below. If the licence URL returns 404, download the MIT licence file listed at the root of `https://github.com/visjs/vis-network/tree/v9.1.9`. If `TestVendoredVisNetworkIntact` then reports `</script` or `<!--`, choose another 9.1.x patch release; do not hand-edit the bundle.

Create `internal/report/assets.go`:

```go
package report

import _ "embed"

// visNetworkJS is vis-network 9.1.9's standalone build (Apache-2.0 OR MIT, see assets/LICENSE-vis-network-MIT.txt).
// Both pages inline it, so opening a report makes no network request; assets_test.go pins its SHA-256.
//
//go:embed assets/vis-network.min.js
var visNetworkJS string
```

In `internal/report/visual.go`, change the `GenerateHTML` comment (lines 212-213) to `// GenerateHTML writes a self-contained interactive graph page to w. The data and the vendored vis-network are` / `// inline, so the page loads nothing remote. Nodes are coloured by community and sized by degree.`. In `writePage` (lines 230-237), add the bundle to the base pairs:

```go
	// No placeholder is a prefix of another, and a Replacer never rescans what it inserts, so pair order can't
	// change the output and text inside the bundle is never substituted.
	pairs := append([]string{
		"__TITLE__", htmlTextEscape(title),
		"__NODE_COUNT__", strconv.Itoa(len(hn)),
		"__EDGE_COUNT__", strconv.Itoa(len(he)),
		"__NODES__", nodesJSON,
		"__EDGES__", edgesJSON,
		"__VIS_NETWORK__", visNetworkJS,
	}, extra...)
```

Change the `htmlTemplate` comment (lines 257-258) to `// htmlTemplate is the self-contained page GenerateHTML fills in through placeholders such as __NODES__.` / `// It loads nothing remote: vis-network is inlined at __VIS_NETWORK__.`, and replace template lines 264-272 with:

```html
<!-- Defence-in-depth CSP: graph data is already HTML-safe and the DOM is never built from innerHTML.
     vis-network is inlined, so no remote script source is allowed, and connect-src blocks all egress.
     'unsafe-inline' is required because the page's scripts and styles are inline. -->
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; object-src 'none'">
<title>__TITLE__</title>
<script>__VIS_NETWORK__</script>
```

In `internal/report/dashboard.go`, replace template lines 116-124 with the same four elements (comment, CSP meta, `<title>__TITLE__</title>`, `<script>__VIS_NETWORK__</script>`).

Append to `.gitattributes`:

```
# Vendored vis-network: byte-exact (a test pins its SHA-256), so no EOL conversion, and not our source.
internal/report/assets/vis-network.min.js binary linguist-vendored=true
```

In `NOTICE.md`, add under "### Third-Party Dependencies" (after line 56):

```markdown
The binary also bundles one third-party file that is not a Go module:
`internal/report/assets/vis-network.min.js`, the standalone build of
vis-network 9.1.9 (https://github.com/visjs/vis-network), dual-licensed
Apache-2.0 OR MIT and used here under the MIT licence
(`internal/report/assets/LICENSE-vis-network-MIT.txt`). Its licence banner is
kept at the top of the file.
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/report/... -count=1; go vet ./internal/report/...; go build ./...`
Expected: PASS. `TestGenerateHTMLEscapesScriptClose` still passes because the inlined bundle contains no `</script`.

- [ ] **Step 5: Docs**

- `docs/SECURITY.md:172-192` (§1.5): replace the paragraphs saying "Both load the vis-network library from `https://unpkg.com`" and the CDN caveats with: "Both inline a vendored, pinned vis-network 9.1.9 (Apache-2.0 OR MIT; SHA-256 pinned by `internal/report/assets_test.go`), so opening either file makes no network request." Update the quoted CSP to `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; object-src 'none'`. Add an "Upgrading the bundle" note listing the Step 3 download, cross-check and `Get-FileHash` commands.
- `README.md:337`: change the comment to `# interactive, pan/zoom/search (vis-network, bundled — works offline)`.
- `CHANGELOG.md` → `### Security`: "`chiton visualize --format html` and `chiton dashboard` inline a vendored, SHA-pinned vis-network 9.1.9 instead of loading an unpinned script from unpkg.com, and their CSP no longer allows any remote script."

- [ ] **Step 6: Commit**

```powershell
git add internal/report/assets internal/report/assets.go internal/report/assets_test.go internal/report/visual.go internal/report/dashboard.go .gitattributes NOTICE.md docs/SECURITY.md README.md CHANGELOG.md
git commit -m "fix(report): inline a pinned vis-network instead of an unpinned CDN script"
```

### Task F4: Record each embedding's model and text hash; re-embed stale vectors

**Finding:** A vector is stored as a bare little-endian `[]float32` under `v:{nodeID}` (`internal/graph/store/badger/store.go:32,676-688,718-737`), with no record of the model that produced it or the text it came from. `chiton build` updates nodes in place, so a vector outlives edits to its node's text. `chiton embed` only fills IDs that have no vector yet (`cmd/chiton/semantic.go:72-90`), so edits and model switches are never picked up. `serve` indexes every stored vector against the configured model (`cmd/chiton/main.go:955-969`). A different model with the same dimension then produces meaningless scores. A different dimension is silently skipped at query time (`internal/vector/flat.go:57-58`). `chiton embed --model X` with `Y` configured falls into the first case.
**Priority:** P2   **Effort:** M   **Depends on:** none (the value format stays backward compatible, so no store migration and no dependency on Section B)
**Behaviour change:** A `v:` value now begins with a header carrying the model id and the SHA-256 of the embedded text. Legacy bare vectors still decode, with model `""` (unknown). `chiton embed` re-embeds a node when it has no vector, its text changed, or its vector came from another model or an unknown one. It also deletes vectors whose node is gone or no longer embeddable. `serve` and `search` index only vectors from the configured model or of unknown model, keeping the most common dimension, and warn about anything excluded. Dimension is derived from the value length, not stored. DECISION NEEDED (1): on a mismatch, serve warns and excludes the vectors (recommended). The alternative is to refuse to enable `semantic_search` until `chiton embed` runs. DECISION NEEDED (2): the first `chiton embed` after upgrading re-embeds every legacy vector once (recommended; correct). The alternative is a `--adopt-legacy` flag that tags legacy vectors with the current model and text hash without re-embedding (fast, but it trusts them).

**Files:**
- Modify: `internal/graph/store/store.go:22-36` (Embedding, new EmbeddingMeta, EmbeddingMetaLister, EmbeddingDeleter)
- Modify: `internal/graph/store/helpers.go:116-130` (add EmbeddingMetas)
- Modify: `internal/graph/store/badger/store.go:5-23,32,42-50,674-737,750-751`
- Modify: `internal/embed/embed.go:5-12,69-97` (TextSHA, tagging, Plan, KeepModel)
- Modify: `cmd/chiton/semantic.go:3-20,31-136,140-145,196-231`
- Modify: `cmd/chiton/main.go:955-970`
- Test: `internal/graph/store/badger/embeddings_test.go`, `internal/graph/store/helpers_test.go`, `internal/embed/embed_test.go`

**Interfaces:**
- Consumes: `model.ContentSHA(b []byte) string` (existing)
- Produces:
  - `store.Embedding{NodeID string; Vec []float32; Model string; TextSHA string}`
  - `store.EmbeddingMeta{NodeID string; Model string; Dim int; TextSHA string}`
  - `store.EmbeddingMetaLister` with `EmbeddingMetas(ctx context.Context) ([]store.EmbeddingMeta, error)`; helper `store.EmbeddingMetas(ctx context.Context, es store.EmbeddingStore) ([]store.EmbeddingMeta, error)`
  - `store.EmbeddingDeleter` with `DeleteEmbeddings(ctx context.Context, ids []string) error`
  - `embed.TextSHA(text string) string`
  - `embed.Plan(nodes []model.Node, have []store.EmbeddingMeta, modelID string, all bool) (work []model.Node, stale []string)`
  - `embed.KeepModel(embs []store.Embedding, modelID string) ([]store.Embedding, embed.IndexReport)`; `embed.IndexReport{Legacy int; Other map[string]int; WrongDim int}`
  - Note for Section B: only the `v:` **value** format changes. No key changes, and `deleteNode` still deletes `embKey(id)`. A schema-version migration must not rewrite `v:` values.

- [ ] **Step 1: Write the failing test**

Append to `internal/graph/store/badger/embeddings_test.go`. Add `"encoding/json"`, `"slices"` and `badgerdb "github.com/dgraph-io/badger/v4"` to its imports:

```go
func openEmbStore(t *testing.T) *Store {
	t.Helper()
	gs, err := Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	return gs
}

// TestEmbeddingMetadataRoundTrip checks model and text hash survive storage, and EmbeddingMetas reports them with
// the dimension without decoding the vector.
func TestEmbeddingMetadataRoundTrip(t *testing.T) {
	gs, ctx := openEmbStore(t), context.Background()
	sha := model.ContentSHA([]byte("Retry"))
	want := store.Embedding{NodeID: "a", Vec: []float32{0.25, -1, 3}, Model: "nomic-embed-text", TextSHA: sha}
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{want}); err != nil {
		t.Fatal(err)
	}
	got, err := gs.AllEmbeddings(ctx)
	if err != nil || len(got) != 1 || got[0].Model != want.Model || got[0].TextSHA != sha || !slices.Equal(got[0].Vec, want.Vec) {
		t.Fatalf("AllEmbeddings = %+v, %v; want %+v", got, err, want)
	}
	metas, err := gs.EmbeddingMetas(ctx)
	if err != nil || !slices.Equal(metas, []store.EmbeddingMeta{{NodeID: "a", Model: "nomic-embed-text", Dim: 3, TextSHA: sha}}) {
		t.Fatalf("EmbeddingMetas = %+v, %v", metas, err)
	}
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "b", Vec: []float32{1}, Model: strings.Repeat("x", 256)}}); err == nil {
		t.Error("want an error for a model id over 255 bytes")
	}
}

// TestLegacyEmbeddingDecodesAsUnknownModel checks a bare vector written before metadata existed still reads.
func TestLegacyEmbeddingDecodesAsUnknownModel(t *testing.T) {
	gs, ctx := openEmbStore(t), context.Background()
	raw := make([]byte, 12)
	putVec(raw, []float32{1, 2, 3})
	if err := gs.db.Update(func(txn *badgerdb.Txn) error { return txn.Set(embKey("old"), raw) }); err != nil {
		t.Fatal(err)
	}
	got, err := gs.AllEmbeddings(ctx)
	if err != nil || len(got) != 1 || got[0].Model != "" || got[0].TextSHA != "" || !slices.Equal(got[0].Vec, []float32{1, 2, 3}) {
		t.Fatalf("legacy AllEmbeddings = %+v, %v", got, err)
	}
	metas, err := gs.EmbeddingMetas(ctx)
	if err != nil || len(metas) != 1 || metas[0].Model != "" || metas[0].Dim != 3 {
		t.Fatalf("legacy EmbeddingMetas = %+v, %v", metas, err)
	}
}

func TestDeleteEmbeddingsKeepsNodes(t *testing.T) {
	gs, ctx := openEmbStore(t), context.Background()
	if err := gs.UpsertNodes(ctx, []model.Node{{ID: "n1", Kind: model.KindSymbol, Label: "Foo", SourceURI: "file://x.go"}}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "n1", Vec: []float32{1}}, {NodeID: "n2", Vec: []float32{2}}}); err != nil {
		t.Fatal(err)
	}
	if err := gs.DeleteEmbeddings(ctx, []string{"n1", "missing"}); err != nil {
		t.Fatal(err)
	}
	if got, _ := gs.AllEmbeddings(ctx); len(got) != 1 || got[0].NodeID != "n2" {
		t.Errorf("embeddings after delete = %+v, want only n2", got)
	}
	if n, _ := gs.GetNode(ctx, "n1"); n == nil {
		t.Error("DeleteEmbeddings removed the node itself")
	}
}

func TestSnapshotDecodesTaggedEmbedding(t *testing.T) {
	gs, ctx := openEmbStore(t), context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{{ID: "n1", Kind: model.KindSymbol, Label: "Foo", SourceURI: "file://x.go"}})
	_ = gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "n1", Vec: []float32{0.5, -0.5}, Model: "m", TextSHA: model.ContentSHA([]byte("Foo"))}})
	var buf bytes.Buffer
	if err := gs.Snapshot(ctx, &buf); err != nil {
		t.Fatal(err)
	}
	var snap struct {
		Nodes []model.Node `json:"nodes"`
	}
	if err := json.Unmarshal(buf.Bytes(), &snap); err != nil {
		t.Fatal(err)
	}
	if len(snap.Nodes) != 1 || !slices.Equal(snap.Nodes[0].Embedding, []float32{0.5, -0.5}) {
		t.Fatalf("snapshot embedding = %+v, want [0.5 -0.5]", snap.Nodes)
	}
}
```

Append to `internal/graph/store/helpers_test.go`:

```go
func TestEmbeddingMetasFallbackAgrees(t *testing.T) {
	ctx := context.Background()
	gs := openStore(t)
	if err := gs.UpsertEmbeddings(ctx, []store.Embedding{{NodeID: "b", Vec: []float32{1}, Model: "m"}, {NodeID: "a", Vec: []float32{2, 3}}}); err != nil {
		t.Fatal(err)
	}
	want := []store.EmbeddingMeta{{NodeID: "a", Dim: 2}, {NodeID: "b", Model: "m", Dim: 1}}
	for name, es := range map[string]store.EmbeddingStore{"lister": gs, "fallback": plainEmb{gs}} {
		if got, err := store.EmbeddingMetas(ctx, es); err != nil || !slices.Equal(got, want) {
			t.Errorf("%s EmbeddingMetas = %+v, %v; want %+v", name, got, err, want)
		}
	}
}
```

Append to `internal/embed/embed_test.go`, adding `"github.com/Eric-Fernald/chiton/internal/graph/store"` to its imports:

```go
func TestNodeEmbeddingsTagsModelAndText(t *testing.T) {
	embs, _, err := NodeEmbeddings(context.Background(), NewHashEmbedder(16), []model.Node{{ID: "s", Kind: model.KindSymbol, Label: "Retry"}})
	if err != nil || len(embs) != 1 || embs[0].Model != "hash" || embs[0].TextSHA != TextSHA("Retry") {
		t.Fatalf("got %+v, %v; want model hash and the text's SHA", embs, err)
	}
}

func TestPlan(t *testing.T) {
	sym := func(id, label string) model.Node { return model.Node{ID: id, Kind: model.KindSymbol, Label: label} }
	nodes := []model.Node{
		sym("new", "New"),                                    // no vector yet
		sym("same", "Same"),                                  // current vector
		sym("edited", "Edited"),                              // text changed since its vector
		sym("othermodel", "Other"),                           // embedded by another model
		sym("legacy", "Legacy"),                              // stored before models were recorded
		{ID: "file", Kind: model.KindFile, Label: "main.go"}, // not embeddable, yet has a vector
	}
	have := []store.EmbeddingMeta{
		{NodeID: "same", Model: "m", Dim: 2, TextSHA: TextSHA("Same")},
		{NodeID: "edited", Model: "m", Dim: 2, TextSHA: TextSHA("Old text")},
		{NodeID: "othermodel", Model: "x", Dim: 2, TextSHA: TextSHA("Other")},
		{NodeID: "legacy", Dim: 2},
		{NodeID: "file", Model: "m", Dim: 2},
		{NodeID: "gone", Model: "m", Dim: 2}, // its node was deleted
	}
	work, stale := Plan(nodes, have, "m", false)
	if got := idsOf(work); !slices.Equal(got, []string{"new", "edited", "othermodel", "legacy"}) {
		t.Errorf("work = %v", got)
	}
	if !slices.Equal(stale, []string{"file", "gone"}) {
		t.Errorf("stale = %v, want [file gone]", stale)
	}
	if work, _ := Plan(nodes, have, "m", true); len(work) != 5 {
		t.Errorf("--all work = %v, want every embeddable node", idsOf(work))
	}
}

func TestKeepModel(t *testing.T) {
	embs := []store.Embedding{
		{NodeID: "a", Model: "m", Vec: []float32{1, 0}},
		{NodeID: "b", Vec: []float32{0, 1}},                // legacy, same dim: kept
		{NodeID: "c", Model: "other", Vec: []float32{1, 1}}, // another model: dropped
		{NodeID: "d", Model: "m", Vec: []float32{1, 0, 0}},  // minority dimension: dropped
		{NodeID: "e", Model: "m"},                           // empty: dropped
	}
	kept, rep := KeepModel(embs, "m")
	var ids []string
	for _, e := range kept {
		ids = append(ids, e.NodeID)
	}
	if !slices.Equal(ids, []string{"a", "b"}) {
		t.Errorf("kept = %v, want [a b]", ids)
	}
	if rep.Legacy != 1 || rep.WrongDim != 1 || len(rep.Other) != 1 || rep.Other["other"] != 1 {
		t.Errorf("report = %+v", rep)
	}
}

func idsOf(nodes []model.Node) []string {
	ids := make([]string, len(nodes))
	for i, n := range nodes {
		ids[i] = n.ID
	}
	return ids
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/graph/store/... ./internal/embed/ -count=1`
Expected: FAIL. The build breaks with `unknown field Model in struct literal of type store.Embedding`, `gs.EmbeddingMetas undefined`, `gs.DeleteEmbeddings undefined`, `undefined: putVec`, `undefined: store.EmbeddingMeta`, `undefined: TextSHA`, `undefined: Plan` and `undefined: KeepModel`.

- [ ] **Step 3: Implement**

`internal/graph/store/store.go`: replace `Embedding` (lines 22-26), and add after `EmbeddingStore` (line 36):

```go
// Embedding pairs a node ID with its embedding vector and what produced it.
type Embedding struct {
	NodeID  string    `json:"node_id"`
	Vec     []float32 `json:"vec"`
	Model   string    `json:"model,omitempty"`    // embedder Model(); "" for a vector stored before models were recorded
	TextSHA string    `json:"text_sha,omitempty"` // hex SHA-256 of the embedded text (embed.TextSHA)
}
```

```go
// EmbeddingMeta describes a stored embedding without its vector.
type EmbeddingMeta struct {
	NodeID  string
	Model   string // "" when unknown (stored before models were recorded)
	Dim     int
	TextSHA string
}

// EmbeddingMetaLister is an optional EmbeddingStore capability listing embedding metadata without decoding
// vectors. Callers use the EmbeddingMetas helper.
type EmbeddingMetaLister interface {
	EmbeddingMetas(ctx context.Context) ([]EmbeddingMeta, error)
}

// EmbeddingDeleter is an optional EmbeddingStore capability removing vectors while keeping their nodes.
type EmbeddingDeleter interface {
	// DeleteEmbeddings removes the embeddings of ids. Missing IDs are skipped.
	DeleteEmbeddings(ctx context.Context, ids []string) error
}
```

`internal/graph/store/helpers.go`: append after `EmbeddingIDs` (keep `EmbeddingIDs`; other code may still use it):

```go
// EmbeddingMetas lists es's stored embeddings without their vectors, via EmbeddingMetaLister when es has it.
func EmbeddingMetas(ctx context.Context, es EmbeddingStore) ([]EmbeddingMeta, error) {
	if l, ok := es.(EmbeddingMetaLister); ok {
		return l.EmbeddingMetas(ctx)
	}
	embs, err := es.AllEmbeddings(ctx)
	if err != nil {
		return nil, err
	}
	metas := make([]EmbeddingMeta, len(embs))
	for i, e := range embs {
		metas[i] = EmbeddingMeta{NodeID: e.NodeID, Model: e.Model, Dim: len(e.Vec), TextSHA: e.TextSHA}
	}
	return metas, nil
}
```

`internal/graph/store/badger/store.go`:
- imports: add `"encoding/hex"`.
- line 32: `pfxEmb  = "v:" // v:{nodeID}            → embMagic header + packed little-endian []float32 (legacy: bare vector)`
- var block (lines 42-50): add `_ store.EmbeddingMetaLister = (*Store)(nil)` and `_ store.EmbeddingDeleter = (*Store)(nil)`.
- Replace `UpsertEmbeddings` and `AllEmbeddings` (lines 674-688, 705-716) and `encodeVec` (lines 718-725). Keep `decodeVec`. Add the rest:

```go
// UpsertEmbeddings stores each vector with its model and text hash in the v: keyspace, out of the node JSON, so
// graph traversals stay lean. Implements store.EmbeddingStore.
func (s *Store) UpsertEmbeddings(_ context.Context, embs []store.Embedding) error {
	return s.batch(func(wb *badgerdb.WriteBatch) error {
		for _, e := range embs {
			if e.NodeID == "" {
				continue
			}
			val, err := encodeEmb(e)
			if err != nil {
				return err
			}
			if err := wb.Set(embKey(e.NodeID), val); err != nil {
				return err
			}
		}
		return nil
	})
}

// AllEmbeddings returns every stored embedding by scanning the v: keyspace. Implements store.EmbeddingStore.
func (s *Store) AllEmbeddings(_ context.Context) ([]store.Embedding, error) {
	var embs []store.Embedding
	err := s.db.View(func(txn *badgerdb.Txn) error {
		return scan(txn, pfxEmb, func(k, v []byte) error {
			embs = append(embs, decodeEmb(string(k), v))
			return nil
		})
	})
	return embs, err
}

// EmbeddingMetas lists every stored embedding's metadata in key order, parsing headers only.
// Implements store.EmbeddingMetaLister.
func (s *Store) EmbeddingMetas(_ context.Context) ([]store.EmbeddingMeta, error) {
	var metas []store.EmbeddingMeta
	err := s.db.View(func(txn *badgerdb.Txn) error {
		return scan(txn, pfxEmb, func(k, v []byte) error {
			m := store.EmbeddingMeta{NodeID: string(k)}
			if modelID, sha, off, ok := embHeader(v); ok {
				m.Model, m.TextSHA, m.Dim = modelID, sha, (len(v)-off)/4
			}
			metas = append(metas, m)
			return nil
		})
	})
	return metas, err
}

// DeleteEmbeddings removes the vectors of ids, keeping their nodes. Implements store.EmbeddingDeleter.
func (s *Store) DeleteEmbeddings(_ context.Context, ids []string) error {
	return s.batch(func(wb *badgerdb.WriteBatch) error {
		for _, id := range ids {
			if err := wb.Delete(embKey(id)); err != nil {
				return err
			}
		}
		return nil
	})
}

// embMagic opens a tagged embedding value. As a float32 it is a NaN, which no stored vector starts with, so a
// value without it is a legacy bare vector whose model is unknown.
const embMagic uint32 = 0x7FC04301

// encodeEmb packs e as embMagic, the length-prefixed model id and raw text SHA, zero padding to a 4-byte boundary,
// then the vector as little-endian float32s.
func encodeEmb(e store.Embedding) ([]byte, error) {
	sha, err := hex.DecodeString(e.TextSHA)
	if err != nil || len(e.Model) > 255 || len(sha) > 255 {
		return nil, fmt.Errorf("embedding %s: model id over 255 bytes or text SHA not short hex", e.NodeID)
	}
	hdr := (4 + 1 + len(e.Model) + 1 + len(sha) + 3) &^ 3
	b := make([]byte, hdr+4*len(e.Vec))
	binary.LittleEndian.PutUint32(b, embMagic)
	b[4] = byte(len(e.Model))
	n := 5 + copy(b[5:], e.Model)
	b[n] = byte(len(sha))
	copy(b[n+1:], sha)
	putVec(b[hdr:], e.Vec)
	return b, nil
}

// embHeader parses an embedding value's header: its model id, hex text SHA and the vector's byte offset. A legacy
// bare vector has none (off 0). ok is false for a malformed value.
func embHeader(b []byte) (modelID, textSHA string, off int, ok bool) {
	if len(b) < 4 || binary.LittleEndian.Uint32(b) != embMagic {
		return "", "", 0, len(b)%4 == 0
	}
	p := 4
	if p >= len(b) {
		return "", "", 0, false
	}
	ml := int(b[p])
	p++
	if p+ml >= len(b) {
		return "", "", 0, false
	}
	modelID = string(b[p : p+ml])
	p += ml
	sl := int(b[p])
	p++
	if p+sl > len(b) {
		return "", "", 0, false
	}
	textSHA = hex.EncodeToString(b[p : p+sl])
	off = (p + sl + 3) &^ 3
	if off > len(b) || (len(b)-off)%4 != 0 {
		return "", "", 0, false
	}
	return modelID, textSHA, off, true
}

// decodeEmb reverses encodeEmb and also reads legacy bare vectors (model ""). A malformed value yields a nil Vec.
func decodeEmb(id string, b []byte) store.Embedding {
	modelID, sha, off, ok := embHeader(b)
	if !ok {
		return store.Embedding{NodeID: id}
	}
	return store.Embedding{NodeID: id, Vec: decodeVec(b[off:]), Model: modelID, TextSHA: sha}
}

// putVec writes vec into b as little-endian float32s (4 bytes per value).
func putVec(b []byte, vec []float32) {
	for i, f := range vec {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
}
```

- `Snapshot` (line 751): `embByID[string(k)] = decodeEmb(string(k), v).Vec`.

`internal/embed/embed.go`: add `"slices"` to the imports. In `TextEmbeddings`, change the loop body (line 94) to:

```go
		embs[i] = store.Embedding{NodeID: ids[i], Vec: vecs[i], Model: e.Model(), TextSHA: TextSHA(texts[i])}
```

and append:

```go
// TextSHA is the hash stored with a vector to detect that its node's text changed since it was embedded.
func TextSHA(text string) string { return model.ContentSHA([]byte(text)) }

// Plan picks the nodes `chiton embed` must embed for modelID: those without a vector, with edited text, or with a
// vector from another or unknown model (all: every embeddable node). stale lists vectors whose node is gone or has no text.
func Plan(nodes []model.Node, have []store.EmbeddingMeta, modelID string, all bool) (work []model.Node, stale []string) {
	byID := make(map[string]store.EmbeddingMeta, len(have))
	for _, m := range have {
		byID[m.NodeID] = m
	}
	for _, n := range nodes {
		m, ok := byID[n.ID]
		delete(byID, n.ID)
		text := TextFor(n)
		switch {
		case text == "":
			if ok {
				stale = append(stale, n.ID)
			}
		case all || !ok || m.Model != modelID || m.TextSHA != TextSHA(text):
			work = append(work, n)
		}
	}
	for id := range byID {
		stale = append(stale, id)
	}
	slices.Sort(stale)
	return work, stale
}

// IndexReport says what KeepModel left out, and how many legacy vectors it kept on trust.
type IndexReport struct {
	Legacy   int            // kept: stored before models were recorded, so assumed to match
	Other    map[string]int // dropped: count per other model id
	WrongDim int            // dropped: dimension differs from the kept majority
}

// KeepModel filters embs in place to non-empty vectors made by modelID or of unknown model, then to their most
// common dimension, since cosine scores across models or dimensions are meaningless.
func KeepModel(embs []store.Embedding, modelID string) ([]store.Embedding, IndexReport) {
	rep := IndexReport{Other: map[string]int{}}
	dims := map[int]int{}
	kept := embs[:0]
	for _, e := range embs {
		switch {
		case len(e.Vec) == 0:
			continue
		case e.Model != "" && e.Model != modelID:
			rep.Other[e.Model]++
			continue
		}
		dims[len(e.Vec)]++
		kept = append(kept, e)
	}
	dim, most := 0, 0
	for d, n := range dims {
		if n > most || (n == most && d > dim) {
			dim, most = d, n
		}
	}
	out := kept[:0]
	for _, e := range kept {
		if len(e.Vec) != dim {
			rep.WrongDim++
			continue
		}
		if e.Model == "" {
			rep.Legacy++
		}
		out = append(out, e)
	}
	clear(embs[len(out):]) // drop the excluded vectors so the GC can free them
	return out, rep
}
```

`cmd/chiton/semantic.go`: add `"io"` and `"maps"` to the imports. In `embedCmd`, change the Long text's second sentence (lines 35-36) to "Nodes without a vector, or whose text or embedding model changed since their vector was written, are (re-)embedded; vectors of deleted or no-longer-embeddable nodes are pruned. --all re-embeds everything." Change the `--all` flag help (line 132) to "Re-embed every embeddable node, even those with a current vector". Replace lines 72-95 with:

```go
			metas, err := store.EmbeddingMetas(ctx, es)
			if err != nil {
				return fmt.Errorf("load existing embeddings: %w", err)
			}
			// work holds embeddable nodes with no current vector. Texts are built per batch, so only one batch of
			// doc-section text, which can be large, is live at a time.
			work, stale := embed.Plan(nodes, metas, emb.Model(), all)
			if err := pruneEmbeddings(ctx, es, stale, out); err != nil {
				return err
			}
			if len(work) == 0 {
				fmt.Fprintln(out, "embed: nothing to do (every embeddable node has a current vector; use --all to re-embed)")
				return nil
			}
			fmt.Fprintf(out, "embed: %d node(s) to embed\n", len(work))
```

Change line 118 to `n, err := linkSimilarNodes(ctx, gs, es, float32(similarThreshold), emb.Model())`. Replace the head of `linkSimilarNodes` (lines 138-145):

```go
// linkSimilarNodes builds a flat index over the embeddings made by modelID and adds a SIMILAR_TO edge (INFERRED,
// confidence=score) for each pair above threshold.
func linkSimilarNodes(ctx context.Context, gs store.GraphStore, es store.EmbeddingStore, threshold float32, modelID string) (int, error) {
	embs, err := es.AllEmbeddings(ctx)
	if err != nil {
		return 0, err
	}
	embs, _ = embed.KeepModel(embs, modelID) // never link vectors from different models
	idx := consumeIntoFlatIndex(slices.Clone(embs)) // clone so embs keeps raw vectors for the loop below
```

Add after `consumeIntoFlatIndex`:

```go
// pruneEmbeddings deletes the stale vectors embed.Plan found, when the store can delete vectors on their own.
func pruneEmbeddings(ctx context.Context, es store.EmbeddingStore, stale []string, out io.Writer) error {
	d, ok := es.(store.EmbeddingDeleter)
	if !ok || len(stale) == 0 {
		return nil
	}
	if err := d.DeleteEmbeddings(ctx, stale); err != nil {
		return fmt.Errorf("prune stale embeddings: %w", err)
	}
	fmt.Fprintf(out, "embed: pruned %d vector(s) of deleted or no-longer-embeddable nodes\n", len(stale))
	return nil
}

// indexEmbeddings indexes the embeddings made by modelID (see embed.KeepModel) and tells errOut, prefixed by who,
// what it left out. It returns the index and how many vectors it holds.
func indexEmbeddings(errOut io.Writer, who string, embs []store.Embedding, modelID string) (*vector.Flat, int) {
	kept, rep := embed.KeepModel(embs, modelID)
	for _, m := range slices.Sorted(maps.Keys(rep.Other)) {
		fmt.Fprintf(errOut, "%s: warning: skipped %d embedding(s) made with model %q, not the configured %q — run 'chiton embed' to re-embed\n",
			who, rep.Other[m], m, modelID)
	}
	if rep.WrongDim > 0 {
		fmt.Fprintf(errOut, "%s: warning: skipped %d embedding(s) whose dimension differs from the rest — run 'chiton embed'\n", who, rep.WrongDim)
	}
	if rep.Legacy > 0 {
		fmt.Fprintf(errOut, "%s: note: %d embedding(s) predate model tracking and are assumed to be %q — run 'chiton embed' to verify\n",
			who, rep.Legacy, modelID)
	}
	return consumeIntoFlatIndex(kept), len(kept)
}
```

In `searchCmd`, replace lines 222-231 (from `if len(embs) == 0 {` through `eng := retrieve.New(...)`):

```go
			emb, err := newEmbedder(cfg, "", "")
			if err != nil {
				return err
			}
			idx, n := indexEmbeddings(cmd.ErrOrStderr(), "search", embs, emb.Model())
			if n == 0 {
				fmt.Fprintln(out, "search: no embeddings for the configured model — run 'chiton embed' first")
				return nil
			}
			clusterResults, _ := cluster.LoadSidecar(cluster.SidecarPath(cfg.GraphPath))
			eng := retrieve.New(gs, idx, emb, clusterResults)
```

`cmd/chiton/main.go`, `buildSemanticDeps`: replace lines 964-970:

```go
	idx, n := indexEmbeddings(errOut, "serve", embs, emb.Model())
	if n == 0 {
		fmt.Fprintln(errOut, "serve: note: no embeddings for the configured model — semantic_search disabled (run 'chiton embed')")
		return deps
	}
	deps.Retriever = retrieve.New(gs, idx, emb, clusterResults)
	fmt.Fprintf(errOut, "serve: semantic_search enabled (%d embeddings, model %s)\n", n, emb.Model())
	return deps
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go build ./...; go test ./... -count=1; go vet ./...`
Expected: PASS. The `store.Embedding` change reaches `internal/ingest` (sync `--embed`), `internal/extract/subagent` and `internal/graph/portable`, so run the whole suite. Portable imports keep writing untagged (unknown-model) vectors by design.

- [ ] **Step 5: Docs**

- `README.md:203`: change to "Embed nodes for semantic search (`--provider`, `--all`, `--link-similar`). Re-embeds nodes whose text or embedding model changed and prunes vectors of deleted nodes. Takes no path — it always covers the whole store."
- `CLAUDE.md:152-153`: append "`chiton embed` re-embeds nodes whose text or model changed since their vector was written; `serve` leaves out (with a warning) vectors made by a different model than the configured one."
- `docs/MCP.md:62-64`: append to the `semantic_search` bullet: "Vectors made by a different embedding model than the server's are left out of the index, and serve logs a warning on stderr; re-run `chiton embed`."
- `skills/setup/SKILL.md:157`: add a troubleshooting row `| serve warns "made with model …" | embedding model changed | chiton embed, then restart the server |`.
- `CHANGELOG.md` → `### Changed`: "Stored embeddings record their model and a hash of the embedded text. `chiton embed` re-embeds nodes whose text or model changed and prunes vectors of deleted nodes, and `serve`/`search` exclude vectors from a different model with a warning. Existing vectors read as 'unknown model', so the first `chiton embed` after upgrading re-embeds them once."

- [ ] **Step 6: Commit**

```powershell
git add internal/graph/store/store.go internal/graph/store/helpers.go internal/graph/store/helpers_test.go internal/graph/store/badger/store.go internal/graph/store/badger/embeddings_test.go internal/embed/embed.go internal/embed/embed_test.go cmd/chiton/semantic.go cmd/chiton/main.go README.md CLAUDE.md docs/MCP.md skills/setup/SKILL.md CHANGELOG.md
git commit -m "feat(embed): record model and text hash per vector; re-embed stale vectors"
```

### Task F5: Stop embedding external stubs; store vectors in one slab

**Finding:** `TextFor` returns the label for every `KindSymbol` (`internal/embed/embed.go:53-56`). That includes the label-only `external` stubs the Go extractor creates for every imported package and cross-package call (`internal/extract/ast/extractor.go:329-339`, `golang.go:149-158`), and it explains most of the "94% of nodes embedded" and the serve footprint of 455 MB with 110,643 vectors against 24 MB without. `Flat` keeps each vector as its own slice (`internal/vector/flat.go:12-16,27-41`).
**Priority:** P3   **Effort:** S   **Depends on:** F4 (`embed.Plan` prunes the stub vectors that already exist)
**Behaviour change:** External package stubs (`pkg://…` symbols such as `fmt.Println`) are no longer embedded. `semantic_search` can't seed from them directly but still reaches them through graph expansion. The first `chiton embed` after this change deletes their vectors (F4's prune). An index now holds a single dimension: vectors whose length differs from the first vector added are skipped at `Add`, not at search. Honest sizing: most of the memory saving comes from skipping stubs. Per-vector slices already land on exact Go size classes (1 KB / 3 KB), so the slab itself saves only slice headers and GC-scanned pointers and gives a faster, cache-friendly scan. Measure serve RSS on a scratch copy of a large store before and after. DECISION NEEDED: int8 quantization (about 4× smaller, small recall loss). Recommended: defer, and revisit only if RSS is still a problem after this task.

**Files:**
- Modify: `internal/embed/embed.go:51-67` (TextFor)
- Modify: `internal/vector/flat.go:10-41,45-76,117-154`
- Test: `internal/embed/embed_test.go`, `internal/vector/flat_test.go`

**Interfaces:**
- Consumes: `embed.Plan` (F4)
- Produces: none (the `vector.Index` API is unchanged)

- [ ] **Step 1: Write the failing test**

In `internal/embed/embed_test.go`, add this case to the `cases` table in `TestTextFor` (after line 154):

```go
		{
			name: "external stub not embeddable",
			node: model.Node{Kind: model.KindSymbol, Label: "fmt.Println", Attrs: map[string]any{"external": true}},
			want: "",
		},
```

and append:

```go
func TestPlanPrunesExternalStubVectors(t *testing.T) {
	stub := model.Node{ID: "stub", Kind: model.KindSymbol, Label: "fmt.Println", Attrs: map[string]any{"external": true}}
	have := []store.EmbeddingMeta{{NodeID: "stub", Model: "m", Dim: 2, TextSHA: TextSHA("fmt.Println")}}
	if work, stale := Plan([]model.Node{stub}, have, "m", false); len(work) != 0 || !slices.Equal(stale, []string{"stub"}) {
		t.Fatalf("work=%v stale=%v, want the stub's vector pruned", idsOf(work), stale)
	}
}
```

In `internal/vector/flat_test.go`, add `"maps"` to the imports, then make these changes. Change `TestFlatSearchTopKMatchesFullSort` lines 27-30 to:

```go
	var want []Result
	q := normalized(query)
	for i := range f.ids {
		want = append(want, Result{ID: f.ids[i], Score: dot(q, f.row(i))})
	}
```

Change `TestNewFlatSizedMatchesNewFlat` lines 143-147 to:

```go
	if !slices.Equal(plain.vecs, sized.vecs) || !maps.Equal(plain.pos, sized.pos) {
		t.Fatal("slabs or positions differ")
	}
```

and append:

```go
// TestFlatMatchesReferenceAfterMutations checks the slab against a map of normalized vectors scored the same way,
// through adds, replacements and swap-with-last deletes.
func TestFlatMatchesReferenceAfterMutations(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	f := NewFlatSized(10)
	ref := map[string][]float32{}
	for i := range 500 {
		id := fmt.Sprintf("n%d", rng.Intn(120))
		if i%7 == 0 {
			f.Delete([]string{id})
			delete(ref, id)
			continue
		}
		v := []float32{rng.Float32() - 0.5, rng.Float32() - 0.5, rng.Float32() - 0.5, rng.Float32() - 0.5}
		f.Add([]Item{{ID: id, Vec: v}})
		ref[id] = normalized(v)
	}
	if f.Len() != len(ref) {
		t.Fatalf("Len = %d, want %d", f.Len(), len(ref))
	}
	q := []float32{0.2, -0.7, 0.1, 0.4}
	nq := normalized(q)
	var want []Result
	for id, v := range ref {
		want = append(want, Result{ID: id, Score: dot(nq, v)})
	}
	slices.SortFunc(want, rankOrder)
	for _, k := range []int{1, 5, len(want), len(want) + 3} {
		if got := f.Search(q, k); !slices.Equal(got, want[:min(k, len(want))]) {
			t.Fatalf("k=%d: got %v\nwant %v", k, got, want[:min(k, len(want))])
		}
	}
}

func TestFlatSkipsMismatchedDimOnAdd(t *testing.T) {
	f := NewFlat()
	f.Add([]Item{{ID: "a", Vec: []float32{1, 0}}, {ID: "b", Vec: []float32{1, 0, 0}}})
	if f.Len() != 1 {
		t.Fatalf("Len = %d, want 1 (the 3-dim vector skipped)", f.Len())
	}
	if got := f.Search([]float32{1, 0, 0}, 3); got != nil {
		t.Fatalf("mismatched-dim query = %v, want nil", got)
	}
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/embed/ ./internal/vector/ -count=1`
Expected: FAIL. `TestTextFor/external_stub_not_embeddable` reports `TextFor = "fmt.Println", want ""`, `TestPlanPrunesExternalStubVectors` reports the stub still in `work`, and the vector package build breaks with `f.row undefined`.

- [ ] **Step 3: Implement**

`internal/embed/embed.go`, `TextFor` (lines 51-56):

```go
// TextFor returns a node's embeddable text: a symbol's label and docstring, a doc section's label and text, or a
// concept's label and description. It returns "" for external package stubs and any other node kind.
func TextFor(n model.Node) string {
	switch n.Kind {
	case model.KindSymbol:
		if ext, _ := n.Attrs["external"].(bool); ext {
			return "" // a label-only stub for an import or cross-package call; graph expansion still reaches it
		}
		return join(n.Label, model.AttrString(n.Attrs, "docstring"))
```

(the rest of the switch is unchanged).

`internal/vector/flat.go`: replace the `Flat` type, `NewFlatSized` and `Add` (lines 10-41):

```go
// Flat is an exact cosine index. It L2-normalizes vectors on insert, so a search is a dot product. Vectors live in
// one slab of dim-wide rows. Not safe for concurrent mutation; build once, then read.
type Flat struct {
	dim  int            // row width, fixed by the first vector added
	ids  []string
	vecs []float32      // row i, normalized, is vecs[i*dim:(i+1)*dim], parallel to ids
	pos  map[string]int // id → row
}

// NewFlat returns an empty Flat index.
func NewFlat() *Flat { return NewFlatSized(0) }

// NewFlatSized returns an empty Flat index presized for n vectors, so adding n of them never regrows it.
func NewFlatSized(n int) *Flat {
	return &Flat{ids: make([]string, 0, n), pos: make(map[string]int, n)}
}

// Add inserts or replaces vectors by ID. It skips zero-length vectors and any whose length differs from the first
// vector added, since a cosine score across dimensions is meaningless.
func (f *Flat) Add(items []Item) {
	for _, it := range items {
		if len(it.Vec) == 0 {
			continue
		}
		if f.dim == 0 {
			f.dim = len(it.Vec)
			f.vecs = make([]float32, 0, cap(f.ids)*f.dim)
		}
		if len(it.Vec) != f.dim {
			continue
		}
		idx, ok := f.pos[it.ID]
		if !ok {
			idx = len(f.ids)
			f.pos[it.ID] = idx
			f.ids = append(f.ids, it.ID)
			f.vecs = slices.Grow(f.vecs, f.dim)[:len(f.vecs)+f.dim]
		}
		normalizeInto(f.row(idx), it.Vec)
	}
}

// row returns vector i, capped so an append can't spill into row i+1.
func (f *Flat) row(i int) []float32 { return f.vecs[i*f.dim : (i+1)*f.dim : (i+1)*f.dim] }
```

In `Search`, replace lines 49-69 (from `q := normalized(query)` through the end of the scan loop):

```go
	q := normalized(query)
	if len(q) != f.dim || isZero(q) {
		return nil // an empty, zero or dimension-mismatched query
	}

	// top is a min-heap of the best k hits seen so far, rooted at the worst one kept, so a scan is O(n log k).
	top := make([]Result, 0, min(k, len(f.ids)))
	for i, id := range f.ids {
		r := Result{ID: id, Score: dot(q, f.row(i))}
		switch {
		case len(top) < k:
			top = append(top, r)
			siftUp(top, len(top)-1)
		case worse(top[0], r):
			top[0] = r
			siftDown(top, 0)
		}
	}
```

Update the `Search` doc comment (lines 43-44) so its second sentence reads "A zero or dimension-mismatched query, or k <= 0, yields no results." (unchanged meaning). Replace the body of `Delete`'s loop (lines 120-133):

```go
		idx, ok := f.pos[id]
		if !ok {
			continue
		}
		last := len(f.ids) - 1
		if idx != last {
			f.ids[idx] = f.ids[last]
			copy(f.row(idx), f.row(last))
			f.pos[f.ids[idx]] = idx
		}
		f.ids = f.ids[:last]
		f.vecs = f.vecs[:last*f.dim]
		delete(f.pos, id)
```

Replace `normalized` (lines 139-154):

```go
// normalized returns a unit-L2 copy of v. A zero vector returns a zero copy.
func normalized(v []float32) []float32 {
	out := make([]float32, len(v))
	normalizeInto(out, v)
	return out
}

// normalizeInto writes v scaled to unit L2 length into dst (len(v) long), or zeros for a zero vector.
func normalizeInto(dst, v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		clear(dst)
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i, x := range v {
		dst[i] = x * inv
	}
}
```

`consumeIntoFlatIndex` (`cmd/chiton/semantic.go:185-194`) needs no change. It already presizes with `NewFlatSized(len(embs))`, so the slab is allocated once at its final size.

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/embed/... ./internal/vector/... ./internal/graph/retrieve/... ./cmd/chiton/... -count=1; go vet ./internal/embed/... ./internal/vector/...`
Expected: PASS. `TestFlatEdgeCases` still holds: a dimension-mismatched query returns nil.

- [ ] **Step 5: Docs**

- `docs/MCP.md` `semantic_search` section (around line 371-374): add "External package stubs (`pkg://` symbols such as `fmt.Println`) are not embedded; results reach them through graph expansion."
- `CHANGELOG.md` → `### Changed`: "`chiton embed` no longer embeds label-only external package stubs, and the next run prunes their vectors; this cuts `serve` memory on large Go graphs. The in-memory vector index stores all vectors in one contiguous slab."

- [ ] **Step 6: Commit**

```powershell
git add internal/embed/embed.go internal/embed/embed_test.go internal/vector/flat.go internal/vector/flat_test.go docs/MCP.md CHANGELOG.md
git commit -m "perf(embed): skip external stubs and store vectors in one slab"
```

### Task F6: Ack Slack and Teams webhooks before ingesting; handle 429s

**Finding:** The Slack receiver runs the whole `onMessage` re-ingest (a paged `conversations.history` fetch plus store writes) before it writes the 200 (`internal/ingest/slack/events.go:101-115`). Slack retries anything not acked within 3 s, so a slow ingest triggers retries and duplicate ingests, and nothing de-duplicates redeliveries by `event_id` (the envelope at `events.go:166-172` doesn't even decode it). The Teams webhook has the same problem, and its own comment says so (`internal/ingest/teams/webhook.go:78-91`: "The 202 is sent only after every onChange returns"). `webhook.go` has no build tag, since only `graph_client.go` is `msgraph`-gated, so it builds and tests by default. Outbound, `apiGet` (`internal/ingest/slack/web_client.go:221-240`) goes through `bearer.Get` (`internal/ingest/bearer/bearer.go:13-28`), which never retries a 429 or reads `Retry-After`. A rate-limited page therefore aborts the ingest, and the same applies to Gmail (`internal/ingest/email/gmail.go:104`).
**Priority:** P2   **Effort:** M   **Depends on:** none
**Behaviour change:** Both receivers ack immediately and hand ingestion to a single-worker `jobs.Queue` that coalesces bursts per conversation (Slack, 1 s debounce) or per meeting (Teams, 1 s debounce). A Slack redelivery whose `event_id` was already accepted is acked and dropped; the last 1,024 IDs are remembered. On shutdown the server stops first, and then queued ingests finish before the store closes. `bearer.Get`, shared by Slack and Gmail, retries a 429 up to 3 times. It waits for `Retry-After` (seconds or an HTTP date) or, without one, 1 s/2 s/4 s backoff, caps each wait at 1 min, and aborts the wait on context cancellation.

**Files:**
- Modify: `internal/ingest/bearer/bearer.go:1-28`
- Modify: `internal/ingest/slack/events.go:1-58,94-115,165-172`
- Modify: `internal/ingest/teams/webhook.go:1-47,78-91`
- Modify: `cmd/chiton/slack.go:103-119`, `cmd/chiton/teams.go:104-120`
- Test: `internal/ingest/bearer/bearer_test.go`, `internal/ingest/slack/slack_test.go:145-250`, `internal/ingest/teams/teams_test.go:85-146`

**Interfaces:**
- Consumes: `jobs.New(handler jobs.Handler, opts jobs.Options) *jobs.Queue`, `(*jobs.Queue).Start/Enqueue/Drain` (existing, `internal/jobs/queue.go`)
- Produces: `(slack.MessageEvent).Thread() string`; `(*slack.EventsHandler).Close()`; `(*teams.WebhookHandler).Close()`. `bearer.Get` keeps its signature.

- [ ] **Step 1: Write the failing test**

Append to `internal/ingest/bearer/bearer_test.go`, adding `"slices"`, `"sync/atomic"` and `"time"` to its imports:

```go
func stubSleep(t *testing.T, waits *[]time.Duration, err error) {
	t.Helper()
	prev := sleep
	sleep = func(_ context.Context, d time.Duration) error {
		if waits != nil {
			*waits = append(*waits, d)
		}
		return err
	}
	t.Cleanup(func() { sleep = prev })
}

func TestGetRetriesTooManyRequests(t *testing.T) {
	var waits []time.Duration
	stubSleep(t, &waits, nil)
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "7")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	status, body, err := Get(context.Background(), srv.Client(), srv.URL, "tok")
	if err != nil || status != http.StatusOK || string(body) != "ok" {
		t.Fatalf("got %d %q %v, want 200 ok after one retry", status, body, err)
	}
	if !slices.Equal(waits, []time.Duration{7 * time.Second}) {
		t.Errorf("waits = %v, want [7s] from Retry-After", waits)
	}
}

func TestGetGivesUpAfterMaxRetries(t *testing.T) {
	var waits []time.Duration
	stubSleep(t, &waits, nil)
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()

	status, _, err := Get(context.Background(), srv.Client(), srv.URL, "tok")
	if err != nil || status != http.StatusTooManyRequests || calls.Load() != 4 {
		t.Fatalf("status=%d err=%v calls=%d, want 429 after 1+3 attempts", status, err, calls.Load())
	}
	if !slices.Equal(waits, []time.Duration{time.Second, 2 * time.Second, 4 * time.Second}) {
		t.Errorf("waits = %v, want backoff 1s 2s 4s without Retry-After", waits)
	}
}

func TestGetStopsWaitingWhenCancelled(t *testing.T) {
	stubSleep(t, nil, context.Canceled)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()
	if _, _, err := Get(context.Background(), srv.Client(), srv.URL, "tok"); err != context.Canceled {
		t.Fatalf("err = %v, want context.Canceled from the interrupted wait", err)
	}
}

func TestRetryDelay(t *testing.T) {
	for _, tc := range []struct {
		header  string
		attempt int
		want    time.Duration
	}{
		{"7", 0, 7 * time.Second},
		{"", 2, 4 * time.Second},
		{"999999", 0, maxRetryWait},
		{"junk", 0, time.Second},
	} {
		if got := retryDelay(tc.header, tc.attempt); got != tc.want {
			t.Errorf("retryDelay(%q, %d) = %v, want %v", tc.header, tc.attempt, got, tc.want)
		}
	}
}
```

In `internal/ingest/slack/slack_test.go`, dispatch is now asynchronous. Call `h.Close()` after `h.ServeHTTP(rec, req)` and before the assertions in `TestEventsDispatchesMessage` (line 178), `TestEventsRejectsBadSignature` (line 199), `TestEventsRejectsStaleTimestamp` (line 221) and `TestEventsFailsClosedWithoutSecret` (line 242), and add `defer h.Close()` after construction in `TestEventsURLVerification` (line 147). For example, `TestEventsDispatchesMessage` lines 177-185 become:

```go
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	h.Close() // runs the queued re-ingest

	if rec.Code != http.StatusOK {
		t.Errorf("code=%d want 200", rec.Code)
	}
	if len(got) != 1 || got[0].ChannelID != "C1" || got[0].Text != "ship it" {
		t.Fatalf("dispatched=%+v want 1 message for C1", got)
	}
```

Then append:

```go
// postEvent signs body as Slack would at 1735689600 and serves it, returning the status code.
func postEvent(t *testing.T, h *EventsHandler, secret, body, retryNum string) int {
	t.Helper()
	const ts = "1735689600"
	req := httptest.NewRequest(http.MethodPost, "/slack/events", strings.NewReader(body))
	req.Header.Set("X-Slack-Request-Timestamp", ts)
	req.Header.Set("X-Slack-Signature", signSlack(secret, ts, []byte(body)))
	if retryNum != "" {
		req.Header.Set("X-Slack-Retry-Num", retryNum)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec.Code
}

// TestEventsAcksBeforeIngest checks the 200 goes out while the ingest is still blocked, well inside Slack's 3s.
func TestEventsAcksBeforeIngest(t *testing.T) {
	const secret = "topsecret"
	release := make(chan struct{})
	h := NewEventsHandler(secret, func(context.Context, MessageEvent) error { <-release; return nil }, nil)
	h.now = func() time.Time { return time.Unix(1735689600, 0) }
	body := `{"type":"event_callback","event_id":"Ev1","event":{"type":"message","channel":"C1","user":"U1","text":"hi","ts":"1735689600.000100"}}`

	done := make(chan int, 1)
	go func() { done <- postEvent(t, h, secret, body, "") }()
	select {
	case code := <-done:
		if code != http.StatusOK {
			t.Errorf("code=%d want 200", code)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("ServeHTTP blocked on the ingest; Slack retries anything not acked within 3s")
	}
	close(release)
	h.Close()
}

// TestEventsCoalescesConversationBurst checks a burst in one thread becomes a single re-ingest of the newest event.
func TestEventsCoalescesConversationBurst(t *testing.T) {
	prev := eventDebounce
	eventDebounce = time.Minute // only Close's drain runs the queue, so the burst surely coalesces
	t.Cleanup(func() { eventDebounce = prev })
	const secret = "topsecret"
	var mu sync.Mutex
	var got []MessageEvent
	h := NewEventsHandler(secret, func(_ context.Context, ev MessageEvent) error {
		mu.Lock()
		defer mu.Unlock()
		got = append(got, ev)
		return nil
	}, nil)
	h.now = func() time.Time { return time.Unix(1735689600, 0) }
	for i, ev := range []string{
		`{"type":"message","channel":"C1","user":"U1","text":"a","ts":"1735689600.000200","thread_ts":"1735689600.000100"}`,
		`{"type":"message","channel":"C1","user":"U2","text":"b","ts":"1735689600.000300","thread_ts":"1735689600.000100"}`,
		`{"type":"message","channel":"C1","user":"U1","text":"c","ts":"1735689600.000400","thread_ts":"1735689600.000100"}`,
		`{"type":"message","channel":"C2","user":"U1","text":"d","ts":"1735689600.000500"}`,
	} {
		body := `{"type":"event_callback","event_id":"Ev` + strconv.Itoa(i) + `","event":` + ev + `}`
		if code := postEvent(t, h, secret, body, ""); code != http.StatusOK {
			t.Fatalf("event %d: code=%d", i, code)
		}
	}
	h.Close()

	if len(got) != 2 {
		t.Fatalf("re-ingests = %+v, want 2 (one per conversation)", got)
	}
	for _, ev := range got {
		if ev.ChannelID == "C1" && (ev.Text != "c" || ev.Thread() != "1735689600.000100") {
			t.Errorf("thread re-ingest used %+v, want the newest event of the thread", ev)
		}
	}
}

func TestEventsDropsRedeliveredEvent(t *testing.T) {
	prev := eventDebounce
	eventDebounce = time.Millisecond
	t.Cleanup(func() { eventDebounce = prev })
	const secret = "topsecret"
	var calls atomic.Int32
	ran := make(chan struct{}, 4)
	h := NewEventsHandler(secret, func(context.Context, MessageEvent) error {
		calls.Add(1)
		ran <- struct{}{}
		return nil
	}, nil)
	h.now = func() time.Time { return time.Unix(1735689600, 0) }
	body := `{"type":"event_callback","event_id":"Ev1","event":{"type":"message","channel":"C1","user":"U1","text":"hi","ts":"1735689600.000100"}}`

	postEvent(t, h, secret, body, "")
	select {
	case <-ran:
	case <-time.After(2 * time.Second):
		t.Fatal("first delivery was never ingested")
	}
	if code := postEvent(t, h, secret, body, "1"); code != http.StatusOK {
		t.Fatalf("retry code=%d want 200 (ack so Slack stops retrying)", code)
	}
	h.Close()
	if n := calls.Load(); n != 1 {
		t.Errorf("onMessage ran %d times, want 1 (the redelivery of Ev1 must be dropped)", n)
	}
}

func TestRecentIDsEvictsOldest(t *testing.T) {
	r := newRecentIDs(2)
	if !r.add("a") || !r.add("b") || r.add("a") {
		t.Fatal("a and b must be new once, then a seen")
	}
	if !r.add("c") || !r.add("a") {
		t.Fatal("adding c must evict a, the oldest")
	}
	if !r.add("") || !r.add("") {
		t.Fatal("an empty id is always new")
	}
}
```

In `internal/ingest/teams/teams_test.go`, call `h.Close()` after `h.ServeHTTP(rec, req)` in `TestWebhookWithoutClientStateFailsClosed` (line 112) and `TestWebhookDispatchesValidNotifications` (line 134), and add `defer h.Close()` after construction in `TestWebhookValidationHandshake` (line 86). The file already imports `time`. Append:

```go
// TestWebhookAcksBeforeIngest checks the 202 goes out while the ingest is still blocked, inside Graph's deadline.
func TestWebhookAcksBeforeIngest(t *testing.T) {
	release := make(chan struct{})
	h := NewWebhookHandler("s3cr3t", func(context.Context, ChangeItem) error { <-release; return nil }, nil)
	body := `{"value":[{"subscriptionId":"s","changeType":"created","clientState":"s3cr3t",
	  "resource":"communications/onlineMeetings('M1')/transcripts('T1')"}]}`
	done := make(chan int, 1)
	go func() {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/hook", strings.NewReader(body)))
		done <- rec.Code
	}()
	select {
	case code := <-done:
		if code != http.StatusAccepted {
			t.Errorf("code=%d want 202", code)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("ServeHTTP blocked on the ingest; Graph needs the ack within 3s")
	}
	close(release)
	h.Close()
}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `go test ./internal/ingest/bearer/ ./internal/ingest/slack/ ./internal/ingest/teams/ -count=1`
Expected: FAIL. The build breaks with `undefined: sleep`, `undefined: maxRetryWait`, `undefined: retryDelay`, `h.Close undefined (type *EventsHandler has no field or method Close)`, `undefined: eventDebounce`, `undefined: newRecentIDs` and `h.Close undefined (type *WebhookHandler …)`.

- [ ] **Step 3: Implement**

Replace `internal/ingest/bearer/bearer.go` below the package clause:

```go
import (
	"context"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// maxBodyBytes caps how much of a response body Get reads.
const maxBodyBytes = 32 << 20

// maxRetries bounds how many times Get retries a 429 before returning it to the caller.
const maxRetries = 3

// maxRetryWait caps one wait, so a confused or hostile server can't stall an ingest.
const maxRetryWait = time.Minute

// sleep waits d or until ctx ends. A var so tests skip real waits.
var sleep = func(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// Get sends an "Authorization: Bearer <token>" GET to reqURL, returning the status code and up to 32 MiB of the
// body (a read error truncates it). It retries a 429 up to maxRetries times; callers check the final status.
func Get(ctx context.Context, client *http.Client, reqURL, token string) (status int, body []byte, err error) {
	for attempt := 0; ; attempt++ {
		var wait string
		status, body, wait, err = get(ctx, client, reqURL, token)
		if err != nil || status != http.StatusTooManyRequests || attempt == maxRetries {
			return status, body, err
		}
		if err := sleep(ctx, retryDelay(wait, attempt)); err != nil {
			return status, body, err
		}
	}
}

// get performs one request and also returns the response's Retry-After header.
func get(ctx context.Context, client *http.Client, reqURL, token string) (int, []byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return 0, nil, "", err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	return resp.StatusCode, body, resp.Header.Get("Retry-After"), nil
}

// retryDelay is the wait before retry attempt+1: Retry-After (seconds or an HTTP date) when present, else backoff
// doubling from 1s, capped at maxRetryWait.
func retryDelay(header string, attempt int) time.Duration {
	d := time.Second << attempt
	if secs, err := strconv.Atoi(strings.TrimSpace(header)); err == nil && secs >= 0 {
		d = time.Duration(secs) * time.Second
	} else if t, err := http.ParseTime(header); err == nil {
		d = time.Until(t)
	}
	return max(0, min(d, maxRetryWait))
}
```

`internal/ingest/slack/events.go`: add `"sync"` and `"github.com/Eric-Fernald/chiton/internal/jobs"` to the imports. Add after `replayWindow` (line 21):

```go
// eventDebounce coalesces a burst of messages in one conversation into a single re-ingest. A var so tests can change it.
var eventDebounce = time.Second

// seenEventsCap bounds how many recent event_ids the handler remembers to drop Slack's redeliveries.
const seenEventsCap = 1024
```

Add after `MessageEvent` (line 30):

```go
// Thread returns the thread ts whose conversation ev belongs to, or "" for a top-level channel message.
func (ev MessageEvent) Thread() string {
	if ev.ThreadTS != "" && ev.ThreadTS != ev.TS {
		return ev.ThreadTS
	}
	return ""
}
```

Replace `EventsHandler` and `NewEventsHandler` (lines 32-58):

```go
// EventsHandler receives Slack Events API callbacks. It verifies each request's X-Slack-Signature (HMAC over the raw
// body, within a replay window), answers url_verification, and queues message events for onMessage after acking.
type EventsHandler struct {
	signingSecret string
	onMessage     func(ctx context.Context, ev MessageEvent) error
	log           func(format string, args ...any)
	now           func() time.Time

	queue   *jobs.Queue
	seen    *recentIDs
	mu      sync.Mutex
	pending map[string]MessageEvent // newest event per conversation, taken by the queue worker
}

// NewEventsHandler builds a handler that authenticates every request with the Slack signing secret. An empty secret
// fails closed. onMessage handles one verified message event, later, on a single background worker. Call Close when done.
func NewEventsHandler(signingSecret string, onMessage func(ctx context.Context, ev MessageEvent) error, log func(string, ...any)) *EventsHandler {
	if log == nil {
		log = func(string, ...any) {}
	}
	if signingSecret == "" {
		// Every Slack app has a signing secret, so an empty one is a misconfiguration. Fail closed rather than
		// authenticate forged requests.
		log("slack events: WARNING — no signing secret configured; ALL requests will be rejected (set CHITON_SLACK_SIGNING_SECRET)")
	}
	h := &EventsHandler{
		signingSecret: signingSecret,
		onMessage:     onMessage,
		log:           log,
		now:           time.Now,
		seen:          newRecentIDs(seenEventsCap),
		pending:       map[string]MessageEvent{},
	}
	// One worker, since every ingest writes the shared store; a background ctx, since a request's ends at the ack.
	h.queue = jobs.New(h.ingest, jobs.Options{
		Workers:  1,
		Debounce: eventDebounce,
		OnError:  func(key string, err error) { log("slack events: re-ingest of %s failed: %v", key, err) },
	})
	h.queue.Start(context.Background())
	return h
}

// Close stops accepting events and finishes every queued re-ingest. Call it after the server stops, before the store closes.
func (h *EventsHandler) Close() { h.queue.Drain() }

// enqueue records ev as its conversation's newest event and schedules a coalesced re-ingest.
func (h *EventsHandler) enqueue(ev MessageEvent) {
	key := ConversationURI(ev.ChannelID, ev.Thread())
	h.mu.Lock()
	h.pending[key] = ev
	h.mu.Unlock()
	h.queue.Enqueue(key)
}

// ingest runs onMessage for key's newest pending event. It is the queue's handler.
func (h *EventsHandler) ingest(ctx context.Context, key string) error {
	h.mu.Lock()
	ev, ok := h.pending[key]
	delete(h.pending, key)
	h.mu.Unlock()
	if !ok || h.onMessage == nil {
		return nil
	}
	return h.onMessage(ctx, ev)
}
```

Replace the `event_callback` case (lines 94-115):

```go
	case "event_callback":
		// Only cheap work happens here; the re-ingest runs on the queue, so the ack always beats Slack's 3s deadline.
		if !h.seen.add(env.EventID) {
			h.log("slack events: dropping redelivered event %s (retry %s: %s)", env.EventID,
				r.Header.Get("X-Slack-Retry-Num"), r.Header.Get("X-Slack-Retry-Reason"))
			w.WriteHeader(http.StatusOK)
			return
		}
		var ev slackInnerEvent
		if err := json.Unmarshal(env.Event, &ev); err != nil {
			h.log("slack events: invalid inner event: %v", err)
			w.WriteHeader(http.StatusOK) // ack so Slack does not retry junk
			return
		}
		if h.shouldDispatch(ev) && h.onMessage != nil {
			h.enqueue(MessageEvent{ChannelID: ev.Channel, User: ev.User, Text: ev.Text, TS: ev.TS, ThreadTS: ev.ThreadTS})
		}
		w.WriteHeader(http.StatusOK)
```

Add `EventID string `json:"event_id"`` to `eventEnvelope` (after `TeamID`, line 170), and append:

```go
// recentIDs remembers the last n IDs added, evicting the oldest first.
type recentIDs struct {
	mu   sync.Mutex
	set  map[string]struct{}
	ring []string
	next int
}

func newRecentIDs(n int) *recentIDs {
	return &recentIDs{set: make(map[string]struct{}, n), ring: make([]string, n)}
}

// add records id and reports whether it was new. An empty id is always new.
func (r *recentIDs) add(id string) bool {
	if id == "" {
		return true
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.set[id]; ok {
		return false
	}
	if old := r.ring[r.next]; old != "" {
		delete(r.set, old)
	}
	r.ring[r.next] = id
	r.next = (r.next + 1) % len(r.ring)
	r.set[id] = struct{}{}
	return true
}
```

`internal/ingest/teams/webhook.go`: add `"sync"`, `"time"` and `"github.com/Eric-Fernald/chiton/internal/jobs"` to the imports. Replace `WebhookHandler` and `NewWebhookHandler` (lines 27-47):

```go
// notificationDebounce coalesces repeated notifications for one meeting into a single fetch. A var for tests.
var notificationDebounce = time.Second

// WebhookHandler receives Graph change notifications for Teams transcripts. It echoes ?validationToken=… handshakes
// and queues changes whose clientState matches for onChange, after acking.
type WebhookHandler struct {
	clientState string
	onChange    func(ctx context.Context, c ChangeItem) error
	log         func(format string, args ...any)

	queue   *jobs.Queue
	mu      sync.Mutex
	pending map[string]ChangeItem // newest change per meeting, taken by the queue worker
}

// NewWebhookHandler builds a handler. clientState is the shared secret from the Graph subscription, and the handler
// rejects notifications that don't match it as forged. onChange runs on one background worker; call Close when done.
func NewWebhookHandler(clientState string, onChange func(ctx context.Context, c ChangeItem) error, log func(string, ...any)) *WebhookHandler {
	if log == nil {
		log = func(string, ...any) {}
	}
	if clientState == "" {
		// With no clientState the handler fails closed and rejects every notification. Callers should refuse
		// to start in that case, and this check is the library's fallback.
		log("teams webhook: WARNING — clientState is empty; every notification will be rejected")
	}
	h := &WebhookHandler{clientState: clientState, onChange: onChange, log: log, pending: map[string]ChangeItem{}}
	h.queue = jobs.New(h.dispatch, jobs.Options{
		Workers:  1,
		Debounce: notificationDebounce,
		OnError:  func(key string, err error) { log("teams webhook: onChange failed for %s: %v", key, err) },
	})
	h.queue.Start(context.Background())
	return h
}

// Close stops accepting notifications and finishes every queued ingest. Call it after the server stops.
func (h *WebhookHandler) Close() { h.queue.Drain() }

// changeKey groups notifications by meeting, since one fetch ingests all of a meeting's transcripts.
func changeKey(c ChangeItem) string {
	if meetingID, _, ok := ParseTranscriptResource(c.Resource); ok {
		return "meeting:" + meetingID
	}
	return "resource:" + c.Resource
}

// enqueue records c as its meeting's newest change and schedules a coalesced dispatch.
func (h *WebhookHandler) enqueue(c ChangeItem) {
	key := changeKey(c)
	h.mu.Lock()
	h.pending[key] = c
	h.mu.Unlock()
	h.queue.Enqueue(key)
}

// dispatch runs onChange for key's newest pending change. It is the queue's handler.
func (h *WebhookHandler) dispatch(ctx context.Context, key string) error {
	h.mu.Lock()
	c, ok := h.pending[key]
	delete(h.pending, key)
	h.mu.Unlock()
	if !ok {
		return nil
	}
	return h.onChange(ctx, c)
}
```

Replace the notification loop (lines 78-91):

```go
	// Drop mismatched clientStates without an error response, so a forged callback can't learn valid states.
	// Valid changes only queue here, so the 202 always beats Graph's 3s ack deadline.
	for _, c := range note.Value {
		if !h.validState(c.ClientState) {
			h.log("teams webhook: dropping notification with bad clientState (sub %s)", c.SubscriptionID)
			continue
		}
		if h.onChange != nil {
			h.enqueue(c)
		}
	}
	w.WriteHeader(http.StatusAccepted)
```

`cmd/chiton/slack.go` lines 105-119: use the event's own scope and drain on shutdown:

```go
			handler := slack.NewEventsHandler(signingSecret, func(ctx context.Context, ev slack.MessageEvent) error {
				// A threaded reply re-ingests only its thread, and a top-level message the whole channel. Re-ingest
				// replaces the conversation by its slack:// URI, so this is idempotent.
				rep, err := ing.IngestConversation(ctx, ev.ChannelID, ev.Thread())
				if err != nil {
					return err
				}
				fmt.Fprintf(out, "slack events: ingested %s — %d message(s), %d turns\n",
					slackScope(rep), rep.Messages, rep.Turns)
				return nil
			}, errLogf(cmd))
			defer handler.Close() // runs before the deferred gs.Close, so queued re-ingests finish first
```

`cmd/chiton/teams.go`: after the `teams.NewWebhookHandler(...)` call ends (line 120), add:

```go
			defer handler.Close() // runs before the deferred gs.Close, so queued ingests finish first
```

- [ ] **Step 4: Run the package tests and vet**

Run: `go test ./internal/ingest/... ./internal/jobs/... ./cmd/chiton/... -count=1 -race; go vet ./internal/ingest/... ./cmd/chiton/...; go vet -tags msgraph ./internal/ingest/teams/... ./cmd/chiton/...`
Expected: PASS with no race reports. `-race` needs CGO; on a Windows machine without a C toolchain, drop `-race` and rely on CI. The `msgraph` vet run confirms the tag-gated build still compiles against the new handler. It needs the Graph SDK modules in the module cache; `make check-msgraph` covers the same ground in CI.

- [ ] **Step 5: Docs**

- `README.md:214,216`: append to each receiver row "It acks immediately and ingests on one background worker, coalescing bursts." For Slack also add "and ignores Slack's redeliveries of an already-seen `event_id`".
- `docs/INSTALL.md:282-284` and `:294-295`: add "The receiver acks each request immediately and ingests on a single background worker, so Slack's and Graph's 3-second ack deadlines are always met; on Ctrl-C it finishes queued ingests before exiting."
- `docs/SECURITY.md:84` (Slack egress row): append "rate-limited calls (HTTP 429) are retried up to 3 times, honouring `Retry-After`, capped at 1 minute per wait". Add the same note to the Gmail row (line 85).
- `CHANGELOG.md` → `### Fixed`: "`chiton slack serve-events` and `chiton teams serve-webhook` ack before ingesting, so slow ingests no longer trigger Slack/Graph retries and duplicate re-ingests. Bursts coalesce per conversation or meeting, and Slack redeliveries are dropped by `event_id`. The Slack and Gmail clients retry HTTP 429 using `Retry-After`."

- [ ] **Step 6: Commit**

```powershell
git add internal/ingest/bearer internal/ingest/slack/events.go internal/ingest/slack/slack_test.go internal/ingest/teams/webhook.go internal/ingest/teams/teams_test.go cmd/chiton/slack.go cmd/chiton/teams.go README.md docs/INSTALL.md docs/SECURITY.md CHANGELOG.md
git commit -m "fix(ingest): ack webhooks before ingesting, dedup Slack retries, honour 429 Retry-After"
```
