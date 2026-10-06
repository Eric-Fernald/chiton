package url

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

// fakeFetcher returns a canned payload, bypassing the network.
type fakeFetcher struct{ res *fetchResult }

func (f *fakeFetcher) fetch(_ context.Context, _ string) (*fetchResult, error) {
	return f.res, nil
}

func newIngester(t *testing.T, fr *fetchResult) (*Ingester, *badgerstore.Store) {
	t.Helper()
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	return &Ingester{store: gs, fetcher: &fakeFetcher{res: fr}, now: time.Now}, gs
}

func TestAddHTML(t *testing.T) {
	html := `<html><head><title>Retry Policies</title></head>
	<body><h1>Backoff</h1><p>Use exponential backoff.</p>
	<script>var x=1;</script><p>Cap retries at three.</p></body></html>`
	ing, gs := newIngester(t, &fetchResult{
		canonicalURL: "https://example.com/retry",
		contentType:  "text/html",
		ext:          "html",
		body:         []byte(html),
	})
	rep, err := ing.Add(context.Background(), "https://example.com/retry", AddOptions{Author: "Jane"})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Title != "Retry Policies" {
		t.Errorf("title=%q want Retry Policies", rep.Title)
	}
	if rep.Kind != "article" {
		t.Errorf("kind=%q want article", rep.Kind)
	}

	nodes, _ := gs.AllNodes(context.Background())
	var urlNode *model.Node
	docSections := 0
	for i := range nodes {
		switch nodes[i].Kind {
		case model.KindURL:
			urlNode = &nodes[i]
		case model.KindDocSection:
			docSections++
		}
	}
	if urlNode == nil {
		t.Fatal("no url node created")
	}
	if urlNode.Attrs["author"] != "Jane" {
		t.Errorf("author=%v want Jane", urlNode.Attrs["author"])
	}
	if docSections < 1 {
		t.Errorf("doc sections=%d want >=1", docSections)
	}
	// The script content must not leak into extracted text.
	for _, n := range nodes {
		if n.Kind == model.KindDocSection {
			if txt, _ := n.Attrs["text"].(string); strings.Contains(txt, "var x=1") {
				t.Errorf("script content leaked into doc_section text: %q", txt)
			}
		}
	}
}

func TestAddIdempotentReplace(t *testing.T) {
	ing, gs := newIngester(t, &fetchResult{
		canonicalURL: "https://example.com/p",
		contentType:  "text/plain",
		body:         []byte("first version of the text"),
	})
	ctx := context.Background()
	if _, err := ing.Add(ctx, "https://example.com/p", AddOptions{}); err != nil {
		t.Fatal(err)
	}
	// Re-add with new content; node count for this URI must not double.
	ing.fetcher = &fakeFetcher{res: &fetchResult{
		canonicalURL: "https://example.com/p",
		contentType:  "text/plain",
		body:         []byte("second version of the text entirely"),
	}}
	if _, err := ing.Add(ctx, "https://example.com/p", AddOptions{}); err != nil {
		t.Fatal(err)
	}
	nodes, _ := gs.AllNodes(ctx)
	urls := 0
	for _, n := range nodes {
		if n.Kind == model.KindURL {
			urls++
		}
	}
	if urls != 1 {
		t.Errorf("url nodes=%d want 1 (re-add must replace)", urls)
	}
}

func TestTranscriptTextDispatch(t *testing.T) {
	vtt := "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n<v Host>Welcome to the show.</v>\n"
	text, _ := extractText(&fetchResult{contentType: "text/vtt", ext: "vtt", body: []byte(vtt)})
	if !strings.Contains(text, "Host: Welcome to the show.") {
		t.Errorf("transcript text=%q missing speaker line", text)
	}
	if strings.Contains(text, "WEBVTT") || strings.Contains(text, "-->") {
		t.Errorf("transcript text leaked VTT markup: %q", text)
	}
}

func TestChunkText(t *testing.T) {
	long := ""
	for range 10 {
		long += "paragraph number with some words here\n\n"
	}
	chunks := chunkText(long, 60, 5)
	if len(chunks) < 2 {
		t.Errorf("expected multiple chunks, got %d", len(chunks))
	}
	if len(chunks) > 5 {
		t.Errorf("chunks=%d exceeds maxChunks 5", len(chunks))
	}
}

func TestHostAllowedBlocksPrivate(t *testing.T) {
	// Literal private/loopback/link-local addresses must be refused (SSRF guard).
	for _, h := range []string{"127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1", "169.254.169.254", "0.0.0.0", "::1"} {
		if err := hostAllowed(h); err == nil {
			t.Errorf("hostAllowed(%q) = nil; want blocked", h)
		}
	}
	// A public literal IP is allowed (no DNS needed, keeps the test hermetic).
	if err := hostAllowed("93.184.216.34"); err != nil {
		t.Errorf("hostAllowed(public IP) = %v; want allowed", err)
	}
}

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

func TestHostAllowedBlocksSpecialPurposeRanges(t *testing.T) {
	for _, h := range []string{
		"100.64.0.1", "100.127.255.254", // carrier-grade NAT
		"198.18.0.1", "198.19.255.254", // benchmarking
		"0.1.2.3", "240.0.0.1", "255.255.255.255", "192.0.0.170", "224.0.0.251",
		"::ffff:127.0.0.1", "::ffff:169.254.169.254", // IPv4-mapped
		"64:ff9b::a9fe:a9fe", "2002:a9fe:a9fe::1", "2001:0:4136:e378:8000:63bf:3fff:fdd2", // embed IPv4
		"::ffff:0:7f00:1", "::ffff:0:a9fe:a9fe", // SIIT IPv4-translated
		"::", "::1", "::7f00:1", "::a9fe:a9fe", // unspecified, loopback, IPv4-compatible
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
