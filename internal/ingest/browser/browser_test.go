package browser

import (
	"bytes"
	"context"
	"database/sql"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

// writeHistoryDB creates a minimal Chromium-shaped History database at path with
// the given rows, using the same pure-Go driver the reader uses.
func writeHistoryDB(t *testing.T, path string, rows []Visit) {
	t.Helper()
	db, err := sql.Open("sqlite", filepath.ToSlash(path))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TABLE urls (
		id INTEGER PRIMARY KEY,
		url TEXT, title TEXT, visit_count INTEGER, last_visit_time INTEGER)`); err != nil {
		t.Fatal(err)
	}
	for _, r := range rows {
		var lvt int64
		if !r.LastVisit.IsZero() {
			lvt = unixToChrome(r.LastVisit)
		}
		if _, err := db.Exec(
			`INSERT INTO urls (url, title, visit_count, last_visit_time) VALUES (?, ?, ?, ?)`,
			r.URL, r.Title, r.VisitCount, lvt); err != nil {
			t.Fatal(err)
		}
	}
}

func newStore(t *testing.T) *badgerstore.Store {
	t.Helper()
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	return gs
}

func TestIngestDB(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "History")
	lastVisit := time.Date(2024, 1, 2, 3, 4, 5, 0, time.UTC)
	writeHistoryDB(t, dbPath, []Visit{
		{URL: "https://example.com/docs", Title: "Docs", VisitCount: 7, LastVisit: lastVisit},
		{URL: "https://news.example.org/", Title: "", VisitCount: 1, LastVisit: lastVisit},
		{URL: "chrome://settings", Title: "Settings", VisitCount: 3, LastVisit: lastVisit}, // must be filtered
	})

	gs := newStore(t)
	ctx := context.Background()
	rep, err := New(gs).IngestDB(ctx, dbPath, "chrome", Options{})
	if err != nil {
		t.Fatal(err)
	}
	if rep.NodesUpserted != 2 {
		t.Fatalf("NodesUpserted=%d want 2 (chrome:// must be filtered)", rep.NodesUpserted)
	}

	nodes, _ := gs.AllNodes(ctx)
	var docs *model.Node
	urlNodes := 0
	for i := range nodes {
		if nodes[i].Kind == model.KindURL {
			urlNodes++
		}
		if nodes[i].SourceURI == "browser://https://example.com/docs" {
			docs = &nodes[i]
		}
	}
	if urlNodes != 2 {
		t.Fatalf("url nodes=%d want 2", urlNodes)
	}
	if docs == nil {
		t.Fatal("docs url node not created")
	}
	if docs.Label != "Docs" {
		t.Errorf("label=%q want Docs", docs.Label)
	}
	if docs.Attrs["browser"] != "chrome" {
		t.Errorf("browser=%v want chrome", docs.Attrs["browser"])
	}
	// Attrs round-trip through the store as JSON, so an int comes back float64.
	if vc, _ := docs.Attrs["visit_count"].(float64); vc != 7 {
		t.Errorf("visit_count=%v want 7", docs.Attrs["visit_count"])
	}
	if got := docs.Attrs["last_visit"]; got != lastVisit.Format(time.RFC3339) {
		t.Errorf("last_visit=%v want %s", got, lastVisit.Format(time.RFC3339))
	}
	if docs.Attrs["url"] != "https://example.com/docs" {
		t.Errorf("url attr=%v want the bare URL", docs.Attrs["url"])
	}
}

// TestIngestDBTitleChangeNoDuplicate guards the idempotency contract: a re-visit
// that changed the page <title> must update the same node, not create a second.
func TestIngestDBTitleChangeNoDuplicate(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "History")
	gs := newStore(t)
	ctx := context.Background()
	ing := New(gs)

	writeHistoryDB(t, dbPath, []Visit{
		{URL: "https://example.com/p", Title: "Old Title", VisitCount: 1, LastVisit: time.Now().UTC()},
	})
	if _, err := ing.IngestDB(ctx, dbPath, "chrome", Options{}); err != nil {
		t.Fatal(err)
	}

	// Same URL, new title (Chromium overwrites urls.title on revisit).
	dbPath2 := filepath.Join(t.TempDir(), "History")
	writeHistoryDB(t, dbPath2, []Visit{
		{URL: "https://example.com/p", Title: "New Title", VisitCount: 2, LastVisit: time.Now().UTC()},
	})
	if _, err := ing.IngestDB(ctx, dbPath2, "chrome", Options{}); err != nil {
		t.Fatal(err)
	}

	nodes, _ := gs.AllNodes(ctx)
	urlNodes := 0
	var label string
	for _, n := range nodes {
		if n.Kind == model.KindURL {
			urlNodes++
			label = n.Label
		}
	}
	if urlNodes != 1 {
		t.Fatalf("url nodes=%d want 1 (title change must not duplicate)", urlNodes)
	}
	if label != "New Title" {
		t.Errorf("label=%q want updated to 'New Title'", label)
	}
}

func TestIngestDBIdempotent(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "History")
	writeHistoryDB(t, dbPath, []Visit{
		{URL: "https://example.com/a", Title: "A", VisitCount: 2, LastVisit: time.Now().UTC()},
	})
	gs := newStore(t)
	ctx := context.Background()
	ing := New(gs)
	if _, err := ing.IngestDB(ctx, dbPath, "edge", Options{}); err != nil {
		t.Fatal(err)
	}
	if _, err := ing.IngestDB(ctx, dbPath, "edge", Options{}); err != nil {
		t.Fatal(err)
	}
	nodes, _ := gs.AllNodes(ctx)
	urlNodes := 0
	for _, n := range nodes {
		if n.Kind == model.KindURL {
			urlNodes++
		}
	}
	if urlNodes != 1 {
		t.Errorf("url nodes=%d want 1 (re-ingest must not duplicate)", urlNodes)
	}
}

func TestMinVisitsAndSinceFilter(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "History")
	old := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	recent := time.Date(2024, 6, 1, 0, 0, 0, 0, time.UTC)
	writeHistoryDB(t, dbPath, []Visit{
		{URL: "https://example.com/rare", Title: "Rare", VisitCount: 1, LastVisit: recent},
		{URL: "https://example.com/old", Title: "Old", VisitCount: 50, LastVisit: old},
		{URL: "https://example.com/hot", Title: "Hot", VisitCount: 50, LastVisit: recent},
	})
	gs := newStore(t)
	ctx := context.Background()
	rep, err := New(gs).IngestDB(ctx, dbPath, "chrome", Options{
		MinVisits: 5,
		Since:     time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatal(err)
	}
	// "rare" fails MinVisits; "old" fails Since; only "hot" survives both.
	if rep.NodesUpserted != 1 {
		t.Fatalf("NodesUpserted=%d want 1 (MinVisits + Since filters)", rep.NodesUpserted)
	}
}

func TestChromeToTime(t *testing.T) {
	// 2024-01-01T00:00:00Z round-trips through the Chrome epoch conversion.
	want := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	if got := chromeToTime(unixToChrome(want)); !got.Equal(want) {
		t.Errorf("chromeToTime round-trip = %v want %v", got, want)
	}
	if got := chromeToTime(0); !got.IsZero() {
		t.Errorf("chromeToTime(0) = %v want zero", got)
	}
}

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

func TestIngestDBRedactsSecretsInTitles(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "History")
	now := time.Now().UTC()
	writeHistoryDB(t, dbPath, []Visit{
		{URL: "https://a.example/cb?code=SECRET1", Title: "https://a.example/cb?code=SECRET1", VisitCount: 1, LastVisit: now},
		{URL: "https://b.example/cb?code=SECRET2", Title: "b.example/cb?code=SECRET2", VisitCount: 1, LastVisit: now},
		{URL: "https://c.example/p#tok=SECRET3", Title: "Page tok=SECRET3", VisitCount: 1, LastVisit: now},
	})
	gs := newStore(t)
	ctx := context.Background()
	if _, err := New(gs).IngestDB(ctx, dbPath, "chrome", Options{}); err != nil {
		t.Fatal(err)
	}
	var snap bytes.Buffer
	if err := gs.Snapshot(ctx, &snap); err != nil {
		t.Fatal(err)
	}
	for _, leak := range []string{"SECRET1", "SECRET2", "SECRET3"} {
		if strings.Contains(snap.String(), leak) {
			t.Errorf("stored graph still contains %q", leak)
		}
	}
}

func TestPurgeRunsOncePerIngesterAndSkipsKeepQuery(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	legacy := func(raw string) model.Node {
		uri := "browser://" + raw
		return model.Node{ID: model.NewNodeID(uri, raw), Kind: model.KindURL, Label: raw, SourceURI: uri,
			Attrs: map[string]any{"source": "browser", "url": raw}}
	}
	first, second, trailing := legacy("https://x.example/a?t=1"), legacy("https://x.example/b?t=2"), legacy("https://x.example/c#")
	if err := gs.UpsertNodes(ctx, []model.Node{first, trailing}); err != nil {
		t.Fatal(err)
	}
	dbPath := filepath.Join(t.TempDir(), "History")
	writeHistoryDB(t, dbPath, nil)

	kept := New(gs)
	if rep, _ := kept.IngestDB(ctx, dbPath, "chrome", Options{KeepQuery: true}); rep.Purged != 0 {
		t.Errorf("KeepQuery purged %d, want 0", rep.Purged)
	}
	ing := New(gs)
	if rep, _ := ing.IngestDB(ctx, dbPath, "chrome", Options{}); rep.Purged != 1 {
		t.Errorf("first run purged %d, want 1", rep.Purged)
	}
	if err := gs.UpsertNodes(ctx, []model.Node{second}); err != nil {
		t.Fatal(err)
	}
	if rep, _ := ing.IngestDB(ctx, dbPath, "chrome", Options{}); rep.Purged != 0 {
		t.Errorf("second run purged %d, want 0", rep.Purged)
	}
	for id, want := range map[string]bool{first.ID: false, second.ID: true, trailing.ID: true} {
		if n, _ := gs.GetNode(ctx, id); (n != nil) != want {
			t.Errorf("node %s present=%v, want %v", id, n != nil, want)
		}
	}
}
