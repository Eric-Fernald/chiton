package url

import (
	"context"
	"path/filepath"
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
			if txt, _ := n.Attrs["text"].(string); contains(txt, "var x=1") {
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
	if !contains(text, "Host: Welcome to the show.") {
		t.Errorf("transcript text=%q missing speaker line", text)
	}
	if contains(text, "WEBVTT") || contains(text, "-->") {
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

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
