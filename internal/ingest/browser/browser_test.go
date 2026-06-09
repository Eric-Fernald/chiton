package browser

import (
	"context"
	"database/sql"
	"path/filepath"
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

func TestIsWebURL(t *testing.T) {
	web := []string{"https://example.com", "http://x.org/path"}
	notWeb := []string{"chrome://settings", "about:blank", "file:///c:/x.txt", "chrome-extension://abc/page.html", "javascript:void(0)", ""}
	for _, u := range web {
		if !isWebURL(u) {
			t.Errorf("isWebURL(%q)=false want true", u)
		}
	}
	for _, u := range notWeb {
		if isWebURL(u) {
			t.Errorf("isWebURL(%q)=true want false", u)
		}
	}
}
