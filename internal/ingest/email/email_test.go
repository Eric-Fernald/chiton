package email

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

const multipartEML = `From: Alice <alice@example.com>
To: Bob <bob@example.com>, carol@example.com
Cc: Dave <dave@example.com>
Subject: =?utf-8?B?SGVsbG8gd29ybGQ=?=
Message-Id: <msg1@example.com>
Date: Mon, 02 Jan 2024 03:04:05 +0000
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="BND"

--BND
Content-Type: text/plain; charset=utf-8

This is the plain text body about retry policies.
--BND
Content-Type: text/html; charset=utf-8

<html><body><script>var leak=1;</script><p>HTML version</p></body></html>
--BND--
`

func newStore(t *testing.T) *badgerstore.Store {
	t.Helper()
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	return gs
}

func writeFile(t *testing.T, name, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestIngestLocalEML(t *testing.T) {
	path := writeFile(t, "mail.eml", multipartEML)
	gs := newStore(t)
	ctx := context.Background()

	rep, err := NewIngester(gs).Ingest(ctx, NewLocalSource(path), Query{})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Messages != 1 {
		t.Fatalf("messages=%d want 1", rep.Messages)
	}

	nodes, _ := gs.AllNodes(ctx)
	var emailNode *model.Node
	docSections, persons := 0, 0
	for i := range nodes {
		switch nodes[i].Kind {
		case model.KindEmail:
			emailNode = &nodes[i]
		case model.KindDocSection:
			docSections++
		case model.KindPerson:
			persons++
		}
	}
	if emailNode == nil {
		t.Fatal("no email node created")
	}
	if emailNode.Label != "Hello world" { // RFC 2047 subject decoded
		t.Errorf("subject=%q want 'Hello world'", emailNode.Label)
	}
	if emailNode.Attrs["from"] != "alice@example.com" {
		t.Errorf("from=%v want alice@example.com", emailNode.Attrs["from"])
	}
	if docSections < 1 {
		t.Errorf("doc sections=%d want >=1", docSections)
	}
	if persons != 4 { // Alice, Bob, carol, Dave
		t.Errorf("persons=%d want 4", persons)
	}
	// text/plain alternative is preferred, and script text must not leak.
	for _, n := range nodes {
		if n.Kind == model.KindDocSection {
			txt, _ := n.Attrs["text"].(string)
			if !strings.Contains(txt, "retry policies") {
				t.Errorf("doc_section text=%q missing plain body", txt)
			}
			if strings.Contains(txt, "leak") {
				t.Errorf("script content leaked into body: %q", txt)
			}
		}
		if n.Kind == model.KindPerson && n.Attrs["pii"] != true {
			t.Errorf("person %q missing pii tag", n.Label)
		}
	}
}

func TestIngestHTMLOnlyStripsScript(t *testing.T) {
	const htmlOnly = `From: a@x.com
To: b@x.com
Subject: HTML only
Message-Id: <h1@x.com>
Content-Type: text/html; charset=utf-8

<html><body><style>.x{color:red}</style><script>steal()</script><p>Visible text here.</p></body></html>
`
	path := writeFile(t, "html.eml", htmlOnly)
	gs := newStore(t)
	ctx := context.Background()
	if _, err := NewIngester(gs).Ingest(ctx, NewLocalSource(path), Query{}); err != nil {
		t.Fatal(err)
	}
	nodes, _ := gs.AllNodes(ctx)
	found := false
	for _, n := range nodes {
		if n.Kind == model.KindDocSection {
			txt, _ := n.Attrs["text"].(string)
			if !strings.Contains(txt, "Visible text here") {
				t.Errorf("missing visible text: %q", txt)
			}
			if strings.Contains(txt, "steal") || strings.Contains(txt, "color:red") {
				t.Errorf("script/style leaked: %q", txt)
			}
			found = true
		}
	}
	if !found {
		t.Error("no doc_section created from HTML-only body")
	}
}

const mboxReplyThread = `From alice@example.com Mon Jan 02 03:04:05 2024
From: Alice <alice@example.com>
To: Bob <bob@example.com>
Subject: Original
Message-Id: <orig@example.com>
Date: Mon, 02 Jan 2024 03:04:05 +0000

Original body about the database migration plan.

From bob@example.com Mon Jan 02 04:04:05 2024
From: Bob <bob@example.com>
To: Alice <alice@example.com>
Subject: Re: Original
Message-Id: <reply@example.com>
In-Reply-To: <orig@example.com>
Date: Mon, 02 Jan 2024 04:04:05 +0000

Reply body agreeing with the migration plan.
`

func TestIngestMboxThreadLinking(t *testing.T) {
	path := writeFile(t, "thread.mbox", mboxReplyThread)
	gs := newStore(t)
	ctx := context.Background()

	rep, err := NewIngester(gs).Ingest(ctx, NewLocalSource(path), Query{})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Messages != 2 {
		t.Fatalf("messages=%d want 2", rep.Messages)
	}
	if rep.References != 1 {
		t.Fatalf("references=%d want 1 (reply -> original)", rep.References)
	}

	// The REFERENCES edge must connect two email nodes (reply -> original).
	edges, _ := gs.AllEdges(ctx)
	var ref *model.Edge
	for i := range edges {
		if edges[i].Kind == model.EdgeReferences {
			ref = &edges[i]
		}
	}
	if ref == nil {
		t.Fatal("no REFERENCES edge created")
	}
	src, _ := gs.GetNode(ctx, ref.Src)
	dst, _ := gs.GetNode(ctx, ref.Dst)
	if src == nil || dst == nil || src.Kind != model.KindEmail || dst.Kind != model.KindEmail {
		t.Fatalf("REFERENCES endpoints not both email nodes: src=%v dst=%v", src, dst)
	}
	if dst.Attrs["message_id"] != "<orig@example.com>" {
		t.Errorf("reply should reference the original; dst message_id=%v", dst.Attrs["message_id"])
	}
}

func TestIngestIdempotentReplace(t *testing.T) {
	path := writeFile(t, "mail.eml", multipartEML)
	gs := newStore(t)
	ctx := context.Background()
	ing := NewIngester(gs)
	if _, err := ing.Ingest(ctx, NewLocalSource(path), Query{}); err != nil {
		t.Fatal(err)
	}
	if _, err := ing.Ingest(ctx, NewLocalSource(path), Query{}); err != nil {
		t.Fatal(err)
	}
	nodes, _ := gs.AllNodes(ctx)
	emails := 0
	for _, n := range nodes {
		if n.Kind == model.KindEmail {
			emails++
		}
	}
	if emails != 1 {
		t.Errorf("email nodes=%d want 1 (re-ingest must replace)", emails)
	}
}

func TestMessageURIStability(t *testing.T) {
	m := Message{MessageID: "<abc@host>", Subject: "x", Source: "gmail"}
	if got := MessageURI(m); got != "email://message/abc@host" {
		t.Errorf("MessageURI=%q want email://message/abc@host", got)
	}
	// No Message-ID → content-hashed, deterministic.
	n := Message{Subject: "x", From: Address{Email: "a@b.com"}, Source: "local", Date: time.Unix(0, 0)}
	if a, b := MessageURI(n), MessageURI(n); a != b {
		t.Errorf("MessageURI not deterministic without Message-ID: %q vs %q", a, b)
	}
	if !strings.HasPrefix(MessageURI(n), "email://local/") {
		t.Errorf("fallback URI=%q want email://local/ prefix", MessageURI(n))
	}
}

func TestAddressNameDecodedOnce(t *testing.T) {
	// An encoded-word display name is decoded once (by mail.ParseAddressList).
	encoded := "From: =?utf-8?B?SsO2cmc=?= <jorg@example.com>\r\nTo: b@x.com\r\nSubject: x\r\nMessage-Id: <a@x>\r\n\r\nBody.\r\n"
	m, err := parseMessage([]byte(encoded))
	if err != nil {
		t.Fatal(err)
	}
	if m.From.Name != "Jörg" {
		t.Errorf("From.Name=%q want Jörg (decoded once)", m.From.Name)
	}
	// A quoted literal that merely *looks* like an encoded-word must be preserved
	// verbatim — a second decode pass would corrupt it.
	literal := "From: \"=?utf-8?B?SGVsbG8=?=\" <x@example.com>\r\nTo: b@x.com\r\nSubject: x\r\nMessage-Id: <b@x>\r\n\r\nBody.\r\n"
	m2, err := parseMessage([]byte(literal))
	if err != nil {
		t.Fatal(err)
	}
	if m2.From.Name != "=?utf-8?B?SGVsbG8=?=" {
		t.Errorf("From.Name=%q want the literal encoded-word preserved", m2.From.Name)
	}
}

func TestChunkTextHardCap(t *testing.T) {
	// A single unbroken block longer than size (no paragraph breaks) must still
	// be split so no chunk exceeds the budget.
	one := strings.Repeat("x", 100)
	chunks := chunkText(one, 10)
	if len(chunks) != 10 {
		t.Errorf("chunks=%d want 10", len(chunks))
	}
	for i, c := range chunks {
		if len(c) > 10 {
			t.Errorf("chunk %d len=%d exceeds size 10", i, len(c))
		}
	}
	if strings.Join(chunks, "") != one {
		t.Error("split chunks do not reconstitute the original text")
	}
}

// TestBuildGraphKeepsWholeLongBody checks a long body becomes many sections within the size cap and none of
// its lines is dropped.
func TestBuildGraphKeepsWholeLongBody(t *testing.T) {
	line := "We decided to ship the retry backoff change next sprint.\n"
	const lines = 10_000
	nodes, _, _, _ := buildGraph(Message{Subject: "long", Body: strings.Repeat(line, lines)}, "email://message/long", time.Now())
	kept := 0
	for _, node := range nodes {
		if node.Kind != model.KindDocSection {
			continue
		}
		txt, _ := node.Attrs["text"].(string)
		if len(txt) > maxSectionBytes {
			t.Errorf("section %q is %d bytes, over the %d cap", node.Label, len(txt), maxSectionBytes)
		}
		kept += strings.Count(txt, "retry backoff")
	}
	if kept != lines {
		t.Errorf("sections hold %d of the body's %d lines", kept, lines)
	}
}

func TestParseMessageStripsBOM(t *testing.T) {
	const eml = "From: Alice <alice@example.com>\r\nTo: Bob <bob@example.com>\r\nSubject: Hi\r\nMessage-Id: <b@x>\r\n\r\nBody.\r\n"
	withBOM := append([]byte{0xEF, 0xBB, 0xBF}, []byte(eml)...)
	m, err := parseMessage(withBOM)
	if err != nil {
		t.Fatal(err)
	}
	if m.From.Email != "alice@example.com" {
		t.Errorf("From=%q want alice@example.com (BOM should not corrupt the first header)", m.From.Email)
	}
}

// fakeSource returns canned messages, exercising the Source path without files.
type fakeSource struct{ msgs []Message }

func (f *fakeSource) Name() string                                       { return "outlook" }
func (f *fakeSource) Messages(context.Context, Query) ([]Message, error) { return f.msgs, nil }

func TestIngestFromFakeSource(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	src := &fakeSource{msgs: []Message{{
		MessageID: "<x@y>",
		Subject:   "Quarterly planning",
		From:      Address{Name: "Carol", Email: "carol@corp.com"},
		To:        []Address{{Email: "team@corp.com"}},
		Date:      time.Date(2024, 3, 1, 0, 0, 0, 0, time.UTC),
		Body:      "We decided to ship the retry backoff change.",
	}}}
	rep, err := NewIngester(gs).Ingest(ctx, src, Query{})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Source != "outlook" {
		t.Errorf("source=%q want outlook", rep.Source)
	}
	if rep.Messages != 1 || rep.Persons != 2 {
		t.Errorf("messages=%d persons=%d want 1, 2", rep.Messages, rep.Persons)
	}
}
