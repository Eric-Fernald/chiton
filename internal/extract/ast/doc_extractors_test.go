package ast

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestForFileRoutesDocAndImage(t *testing.T) {
	if _, ok := ForFile("README.md").(*MarkdownExtractor); !ok {
		t.Fatal(".md should route to MarkdownExtractor")
	}
	if _, ok := ForFile("diagram.PNG").(*ImageExtractor); !ok {
		t.Fatal(".PNG should route to ImageExtractor (case-insensitive)")
	}
}

func TestMarkdownHeadingsToDocSections(t *testing.T) {
	md := `# Title

Intro prose.

## Auth

We authenticate with JWTs.

## Storage

We use BadgerDB.

### Encryption

AES-256-GCM at rest.
`
	path := writeTemp(t, "doc.md", md)
	nodes, edges, err := (&MarkdownExtractor{}).ExtractFile(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}

	var file int
	sections := map[string]model.Node{}
	for _, n := range nodes {
		switch n.Kind {
		case model.KindFile:
			file++
		case model.KindDocSection:
			sections[n.Label] = n
		}
	}
	if file != 1 {
		t.Fatalf("want 1 file node, got %d", file)
	}
	for _, want := range []string{"Title", "Auth", "Storage", "Encryption"} {
		if _, ok := sections[want]; !ok {
			t.Fatalf("missing section %q; got %v", want, keys(sections))
		}
	}
	if txt, _ := sections["Auth"].Attrs["text"].(string); txt == "" || !strings.Contains(txt, "JWT") {
		t.Fatalf("Auth section text missing body: %q", txt)
	}
	if lvl, _ := sections["Encryption"].Attrs["level"].(int); lvl != 3 {
		t.Fatalf("Encryption level = %v, want 3", sections["Encryption"].Attrs["level"])
	}

	// CONTAINS edges reconstruct heading nesting: file→Title (only H1), then
	// Title→Auth, Title→Storage, Storage→Encryption.
	containsEdges := map[string]bool{}
	for _, e := range edges {
		if e.Kind != model.EdgeContains {
			continue
		}
		containsEdges[e.Src+"->"+e.Dst] = true
	}
	if len(containsEdges) < 4 {
		t.Fatalf("expected at least 4 CONTAINS edges, got %d", len(containsEdges))
	}
	// Encryption (H3) should be contained by Storage (H2), not the file.
	enc := sections["Encryption"].ID
	storage := sections["Storage"].ID
	if !containsEdges[storage+"->"+enc] {
		t.Fatalf("Encryption should be nested under Storage")
	}
}

func TestMarkdownIgnoresHeadingsInCodeFence(t *testing.T) {
	md := "# Real\n\n```\n# not a heading\n```\n\n## AlsoReal\n"
	path := writeTemp(t, "fence.md", md)
	nodes, _, err := (&MarkdownExtractor{}).ExtractFile(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	var secCount int
	for _, n := range nodes {
		if n.Kind == model.KindDocSection {
			secCount++
			if n.Label == "not a heading" {
				t.Fatal("heading inside code fence was extracted")
			}
		}
	}
	if secCount != 2 {
		t.Fatalf("want 2 sections, got %d", secCount)
	}
}

func TestMarkdownNoHeadingsSingleSection(t *testing.T) {
	path := writeTemp(t, "plain.md", "Just some prose with no headings at all.\n")
	nodes, edges, err := (&MarkdownExtractor{}).ExtractFile(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	var sec int
	for _, n := range nodes {
		if n.Kind == model.KindDocSection {
			sec++
		}
	}
	if sec != 1 {
		t.Fatalf("want 1 fallback section, got %d", sec)
	}
	if len(edges) != 1 {
		t.Fatalf("want 1 CONTAINS edge, got %d", len(edges))
	}
}

func TestMarkdownStripsBOM(t *testing.T) {
	// Windows editors often prepend a UTF-8 BOM; the first heading must still
	// be recognised.
	content := "\xEF\xBB\xBF# Title\n\nbody\n"
	path := writeTemp(t, "bom.md", content)
	nodes, _, err := (&MarkdownExtractor{}).ExtractFile(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	var found bool
	for _, n := range nodes {
		if n.Kind == model.KindDocSection && n.Label == "Title" {
			found = true
		}
	}
	if !found {
		t.Fatal("BOM hid the first heading")
	}
}

func TestImageExtractor(t *testing.T) {
	path := writeTemp(t, "pic.png", "\x89PNG\r\n\x1a\nfakebytes")
	nodes, edges, err := (&ImageExtractor{}).ExtractFile(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes) != 1 || nodes[0].Kind != model.KindImage {
		t.Fatalf("want 1 image node, got %+v", nodes)
	}
	if len(edges) != 0 {
		t.Fatalf("image extractor should emit no edges, got %d", len(edges))
	}
	if nodes[0].Attrs["format"] != "png" {
		t.Fatalf("format = %v, want png", nodes[0].Attrs["format"])
	}
	if _, ok := nodes[0].Attrs["path"].(string); !ok {
		t.Fatal("image node missing path attr")
	}
	if nodes[0].Attrs["extraction_status"] != nil {
		t.Fatal("freshly extracted image should have no extraction_status")
	}
}

func keys(m map[string]model.Node) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
