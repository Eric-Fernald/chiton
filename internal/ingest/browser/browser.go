package browser

import (
	"cmp"
	"context"
	"fmt"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// defaultLimit caps how many (most-recent) history rows a single DB contributes
// when the caller does not specify a limit.
const defaultLimit = 1000

// historyURIScheme namespaces history source_uris so they never collide with an add-url node for the
// same page (which uses the bare URL). The real URL is stored in attrs["url"].
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

	mu     sync.Mutex
	purged bool // purgeUnredacted has run for this Ingester
}

// New builds an Ingester over the given graph store.
func New(s store.GraphStore) *Ingester {
	return &Ingester{store: s, now: time.Now}
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
			prev, _ := nodes[i].Attrs["visit_count"].(int)
			nodes[i].Attrs["visit_count"] = prev + v.VisitCount
			continue
		}
		index[page] = len(nodes)
		// Key on the stable URL, not the mutable title, so a re-ingest after a title change refreshes the same node.
		uri := historyURIScheme + page
		title := v.Title
		if !keepQuery {
			title = redactTitle(title, v.URL, u, page)
		}
		attrs := map[string]any{
			"kind":        "history",
			"source":      "browser",
			"url":         page,
			"title":       title,
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
			Label:      cmp.Or(strings.TrimSpace(title), page),
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
		if u, ok := parseWebURL(raw); ok && (u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.RawFragment != "") {
			stale = append(stale, n)
		}
	}
	deleted, err := store.DeleteNodes(ctx, ing.store, stale)
	if err == nil {
		ing.purged = true
	}
	return deleted, err
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

// redactTitle returns page when title embeds any part of the raw URL that redaction removed. Chromium often
// stores an untitled page's title as its URL, so the secrets would survive in title and label otherwise.
func redactTitle(title, raw string, u *url.URL, page string) string {
	secrets := []string{raw, strings.TrimPrefix(raw, u.Scheme+":"), u.RawQuery, u.Fragment, u.RawFragment}
	if u.User != nil {
		secrets = append(secrets, u.User.String())
	}
	for _, s := range secrets {
		if s != "" && strings.Contains(title, s) {
			return page
		}
	}
	return title
}
