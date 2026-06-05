// Package url ingests external web sources — papers (PDF), articles (HTML),
// plaintext/markdown, and (via yt-dlp) YouTube transcripts — into chiton's
// graph as a `url` node with `doc_section` children. The doc_section children
// share the doc-extraction shape, so the existing pending_extractions /
// submit_extraction subagent flow extracts concepts from an ingested URL for
// free; no URL-specific LLM path is needed.
//
// Fetching is isolated behind the fetcher interface so the node-building logic
// is testable without network access and so YouTube/social handling can be
// swapped without touching callers.
package url

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
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
		// Block redirects to private/loopback/link-local hosts: add_url is
		// reachable over MCP, so a hostile page could otherwise redirect into the
		// internal network (SSRF, e.g. cloud metadata at 169.254.169.254).
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

// Add fetches a URL (or local file), extracts its readable text, and upserts a
// url node plus doc_section children. It is idempotent: the url node and its
// sections share the URL as source_uri, so a re-add first removes the prior
// version (DeleteBySourceURI) and replaces it.
func (ing *Ingester) Add(ctx context.Context, rawURL string, opts AddOptions) (Report, error) {
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return Report{}, fmt.Errorf("add-url: empty URL")
	}
	if ro, ok := ing.store.(store.ReadOnlyAware); ok && ro.ReadOnly() {
		return Report{}, fmt.Errorf("add-url: store is read-only — restart 'chiton serve --writable' or use the 'chiton add-url' CLI")
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
	sum := sha256.Sum256([]byte(text))
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
		"content_sha":  hex.EncodeToString(sum[:]),
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
		ContentSHA256: hex.EncodeToString(sum[:]),
		CreatedAt:     now,
		UpdatedAt:     now,
	}}
	var edges []model.Edge

	limit := opts.MaxTextBytes
	if limit <= 0 {
		limit = defaultSectionBytes
	}
	chunks := chunkText(text, limit, maxSections)
	for i, chunk := range chunks {
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
				"text":  chunk,
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

// chunkText splits text into chunks of at most size bytes, preferring blank-line
// (paragraph) boundaries, capped at maxChunks (the final chunk absorbs the
// remainder once the cap is hit).
func chunkText(text string, size, maxChunks int) []string {
	if size <= 0 {
		size = defaultSectionBytes
	}
	if len(text) <= size {
		return []string{text}
	}
	paras := strings.Split(text, "\n\n")
	var chunks []string
	var cur strings.Builder
	// atCap reports that we have produced all but the final chunk; once true we
	// stop flushing and let the remaining paragraphs accumulate into the last
	// chunk, so the result never exceeds maxChunks.
	atCap := func() bool { return maxChunks > 0 && len(chunks) >= maxChunks-1 }
	for _, p := range paras {
		if cur.Len() > 0 && cur.Len()+len(p)+2 > size && !atCap() {
			chunks = append(chunks, strings.TrimSpace(cur.String()))
			cur.Reset()
		}
		if cur.Len() > 0 {
			cur.WriteString("\n\n")
		}
		cur.WriteString(p)
	}
	if rest := strings.TrimSpace(cur.String()); rest != "" {
		chunks = append(chunks, rest)
	}
	if len(chunks) == 0 {
		return []string{text}
	}
	return chunks
}
