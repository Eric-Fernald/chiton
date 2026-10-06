// Package url ingests web sources (PDF, HTML, text, Markdown, YouTube transcripts via yt-dlp) as a `url` node
// with `doc_section` children, which the existing pending_extractions flow extracts concepts from.
package url

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/ingest/chunk"
)

// defaultSectionBytes caps the text per doc_section so each downstream
// concept-extraction task stays within a sane context budget.
const defaultSectionBytes = 8000

// maxSections bounds how many doc_section children a single huge page produces.
const maxSections = 60

// Ingester turns a URL into graph nodes. The zero value is not usable; build one
// with New.
type Ingester struct {
	store   store.GraphStore
	fetcher fetcher
	now     func() time.Time
}

// New builds an Ingester over the given store using the default HTTP/file
// fetcher (which shells out to yt-dlp for YouTube URLs when present).
func New(s store.GraphStore) *Ingester {
	client := &http.Client{
		Timeout: 30 * time.Second,
		// Block redirects to private, loopback, and link-local hosts. add_url is reachable over MCP, so a
		// hostile page could otherwise redirect into the internal network (e.g. cloud metadata at 169.254.169.254).
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

// AddOptions carries optional metadata recorded on the url node.
type AddOptions struct {
	// Kind is a content hint stored on the node (e.g. "paper", "article",
	// "tweet", "video"); empty lets the ingester infer one from content type.
	Kind         string
	Author       string
	Contributor  string
	MaxTextBytes int // per-section cap; 0 = default
	// AllowLocal permits bare paths and file:// URLs. The CLI sets it; the MCP add_url tool never does.
	AllowLocal bool
}

// Report summarises one Add.
type Report struct {
	URL           string `json:"url"`
	Title         string `json:"title"`
	ContentType   string `json:"content_type"`
	Kind          string `json:"kind"`
	TextBytes     int    `json:"text_bytes"`
	Sections      int    `json:"sections"`
	NodesUpserted int    `json:"nodes_upserted"`
}

// Add fetches a URL (or local file) and upserts a url node and its doc_section children. It is idempotent
// because they all share the URL as source_uri, so a re-add deletes the prior version first.
func (ing *Ingester) Add(ctx context.Context, rawURL string, opts AddOptions) (Report, error) {
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return Report{}, fmt.Errorf("add-url: empty URL")
	}
	if store.IsReadOnly(ing.store) {
		return Report{}, fmt.Errorf("add-url: store is read-only — restart 'chiton serve --writable' or use the 'chiton add-url' CLI")
	}
	if !opts.AllowLocal {
		if err := requireRemote(rawURL); err != nil {
			return Report{}, err
		}
	}

	fr, err := ing.fetcher.fetch(ctx, rawURL)
	if err != nil {
		return Report{}, fmt.Errorf("fetch %s: %w", rawURL, err)
	}

	text, title := extractText(fr)
	text = strings.TrimSpace(text)
	if text == "" {
		return Report{}, fmt.Errorf("add-url: no readable text extracted from %s (content-type %q)", rawURL, fr.contentType)
	}
	if title == "" {
		title = fr.title
	}
	if title == "" {
		title = rawURL
	}

	uri := fr.canonicalURL
	if uri == "" {
		uri = rawURL
	}

	// Replace any prior version of this URL.
	if _, err := ing.store.DeleteBySourceURI(ctx, uri); err != nil {
		return Report{}, fmt.Errorf("replace prior %s: %w", uri, err)
	}

	now := ing.now()
	contentSHA := model.ContentSHA([]byte(text))
	kind := opts.Kind
	if kind == "" {
		kind = inferKind(fr, rawURL)
	}

	urlID := model.NewNodeID(uri, title)
	urlAttrs := map[string]any{
		"title":        title,
		"content_type": fr.contentType,
		"kind":         kind,
		"fetched_at":   now.UTC().Format(time.RFC3339),
		"content_sha":  contentSHA,
		"text_bytes":   len(text),
	}
	if opts.Author != "" {
		urlAttrs["author"] = opts.Author
	}
	if opts.Contributor != "" {
		urlAttrs["contributor"] = opts.Contributor
	}

	nodes := []model.Node{{
		ID:            urlID,
		Kind:          model.KindURL,
		Label:         title,
		SourceURI:     uri,
		Attrs:         urlAttrs,
		Provenance:    model.ProvExtracted,
		Confidence:    1.0,
		ContentSHA256: contentSHA,
		CreatedAt:     now,
		UpdatedAt:     now,
	}}
	var edges []model.Edge

	limit := opts.MaxTextBytes
	if limit <= 0 {
		limit = defaultSectionBytes
	}
	chunks := chunkText(text, limit, maxSections)
	for i, sec := range chunks {
		secLabel := title
		if len(chunks) > 1 {
			secLabel = fmt.Sprintf("%s §%d", title, i+1)
		}
		secID := model.NewNodeID(uri, fmt.Sprintf("section-%d", i))
		nodes = append(nodes, model.Node{
			ID:        secID,
			Kind:      model.KindDocSection,
			Label:     secLabel,
			SourceURI: uri,
			Attrs: map[string]any{
				"text":  sec,
				"index": i,
			},
			Provenance: model.ProvExtracted,
			Confidence: 1.0,
			CreatedAt:  now,
			UpdatedAt:  now,
		})
		edges = append(edges, model.Edge{
			ID:         model.NewEdgeID(urlID, secID, model.EdgeContains),
			Src:        urlID,
			Dst:        secID,
			Kind:       model.EdgeContains,
			Provenance: model.ProvExtracted,
			Confidence: 1.0,
			CreatedAt:  now,
		})
	}

	if err := ing.store.UpsertNodes(ctx, nodes); err != nil {
		return Report{}, fmt.Errorf("upsert url nodes: %w", err)
	}
	if len(edges) > 0 {
		if err := ing.store.UpsertEdges(ctx, edges); err != nil {
			return Report{}, fmt.Errorf("upsert url edges: %w", err)
		}
	}

	return Report{
		URL:           uri,
		Title:         title,
		ContentType:   fr.contentType,
		Kind:          kind,
		TextBytes:     len(text),
		Sections:      len(chunks),
		NodesUpserted: len(nodes),
	}, nil
}

// chunkText splits text into chunks of at most size bytes at paragraph boundaries. Once maxChunks is
// reached, the final chunk absorbs the remainder.
func chunkText(text string, size, maxChunks int) []string {
	if size <= 0 {
		size = defaultSectionBytes
	}
	return chunk.Paragraphs(text, size, maxChunks)
}
