package browser

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// defaultLimit caps how many (most-recent) history rows a single DB contributes
// when the caller does not specify a limit.
const defaultLimit = 1000

// historyURIScheme namespaces a history url node's source_uri so it never
// collides with an `add-url` fetched-content node for the same page (which uses
// the bare URL as its source_uri). The real URL is preserved in attrs["url"].
const historyURIScheme = "browser://"

// DetectedDB is a browser History database found on the local machine.
type DetectedDB struct {
	Browser string // "chrome" | "edge" | "brave" | ...
	Path    string // absolute path to the History SQLite file
}

// Ingester turns browser history into graph `url` nodes.
type Ingester struct {
	store store.GraphStore
	now   func() time.Time
}

// New builds an Ingester over the given graph store.
func New(s store.GraphStore) *Ingester {
	return &Ingester{store: s, now: time.Now}
}

// Options bounds and filters an ingest run.
type Options struct {
	Limit     int       // max rows per DB; 0 → defaultLimit
	Since     time.Time // only visits at/after this time
	MinVisits int       // skip URLs visited fewer than this many times (noise filter)
}

// Report summarises one History database's ingest.
type Report struct {
	Browser       string `json:"browser"`
	DB            string `json:"db"`
	Visits        int    `json:"visits"`
	NodesUpserted int    `json:"nodes_upserted"`
}

func (ing *Ingester) readOnly() bool {
	ro, ok := ing.store.(store.ReadOnlyAware)
	return ok && ro.ReadOnly()
}

// IngestDB reads one History database and upserts a url node per visited web
// page. It is idempotent: nodes are keyed on a stable ID derived from the URL +
// title, so re-running refreshes visit counts and last-visit times in place
// rather than duplicating. Non-web URLs (chrome://, about:, file://, extension
// pages) are skipped.
func (ing *Ingester) IngestDB(ctx context.Context, dbPath, browser string, opts Options) (Report, error) {
	if ing.readOnly() {
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

	now := ing.now()
	var nodes []model.Node
	for _, v := range visits {
		if !isWebURL(v.URL) {
			continue
		}
		label := strings.TrimSpace(v.Title)
		if label == "" {
			label = v.URL
		}
		// Key the node on the (stable) URL, not the mutable page title, and in a
		// namespaced source_uri — so a re-ingest after Chromium updates the title
		// refreshes the same node in place instead of orphaning it, and so the
		// node never collides with an add-url node for the same page.
		uri := historyURIScheme + v.URL
		attrs := map[string]any{
			"kind":        "history",
			"source":      "browser",
			"url":         v.URL,
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
			ID:         model.NewNodeID(uri, v.URL),
			Kind:       model.KindURL,
			Label:      label,
			SourceURI:  uri,
			Attrs:      attrs,
			Provenance: model.ProvExtracted,
			Confidence: 1.0,
			CreatedAt:  now,
			UpdatedAt:  now,
		})
	}

	if len(nodes) > 0 {
		if err := ing.store.UpsertNodes(ctx, nodes); err != nil {
			return Report{}, fmt.Errorf("upsert history nodes: %w", err)
		}
	}
	return Report{Browser: browser, DB: dbPath, Visits: len(nodes), NodesUpserted: len(nodes)}, nil
}

// IngestAuto detects every installed browser's History file and ingests each.
func (ing *Ingester) IngestAuto(ctx context.Context, opts Options) ([]Report, error) {
	dbs := DetectDBs()
	if len(dbs) == 0 {
		return nil, fmt.Errorf("browser: no Chrome/Edge/Brave history found on this machine; pass --db <path>")
	}
	var reports []Report
	for _, d := range dbs {
		rep, err := ing.IngestDB(ctx, d.Path, d.Browser, opts)
		if err != nil {
			return reports, fmt.Errorf("ingest %s (%s): %w", d.Browser, d.Path, err)
		}
		reports = append(reports, rep)
	}
	return reports, nil
}

// isWebURL reports whether raw is an http(s) URL with a host, excluding the
// browser-internal schemes (chrome://, about:, file://, chrome-extension://...).
func isWebURL(raw string) bool {
	u, err := url.Parse(raw)
	if err != nil {
		return false
	}
	switch strings.ToLower(u.Scheme) {
	case "http", "https":
		return u.Host != ""
	default:
		return false
	}
}
