// Package browser ingests local web-browser history (Chrome, Edge, Brave and
// other Chromium-family browsers) into chiton's graph as `url` nodes, one per
// visited page. It is the Phase 6 "browser history ingestion" stretch item.
//
// The History store is a SQLite database. Chromium keeps the live file locked
// while the browser runs, so we copy it to a temp location and read the copy
// with the pure-Go modernc.org/sqlite driver (no CGo — same pure-Go principle
// that put BadgerDB ahead of Kuzu). Only page metadata (URL, title, visit count,
// last-visit time) is ingested; chiton never fetches the pages themselves, so
// browser ingest does no network I/O and raises no SSRF surface. To pull a
// page's content into the graph, use `chiton add-url`, which has the fetch path
// and its SSRF guards.
package browser

import (
	"context"
	"database/sql"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	// Pure-Go SQLite driver (registers the "sqlite" driver name). No CGo.
	_ "modernc.org/sqlite"
)

// chromeEpochOffsetMicros is the gap, in microseconds, between the Windows/
// Chrome epoch (1601-01-01 UTC) and the Unix epoch (1970-01-01 UTC). Chromium
// stores urls.last_visit_time as microseconds since 1601.
const chromeEpochOffsetMicros int64 = 11644473600 * 1_000_000

// Visit is one row from a Chromium History database's `urls` table.
type Visit struct {
	URL        string
	Title      string
	VisitCount int
	LastVisit  time.Time // zero when the stored value was 0 / unparseable
}

// readOptions bounds a history read.
type readOptions struct {
	limit     int
	since     time.Time
	minVisits int
}

// readHistory copies a (possibly live, locked) Chromium History SQLite file to a
// temp location, opens that copy with the pure-Go driver, and returns its visit
// rows. Copying first is the standard way to read the file without contending
// for the lock Chromium holds while running.
func readHistory(ctx context.Context, dbPath string, opts readOptions) ([]Visit, error) {
	tmp, cleanup, err := copyForRead(dbPath)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	// The copy is a disposable temp file, so opening it read-write is harmless;
	// we only ever issue SELECTs. A plain forward-slashed path is the most
	// portable DSN for the modernc driver across Windows/macOS/Linux.
	db, err := sql.Open("sqlite", filepath.ToSlash(tmp))
	if err != nil {
		return nil, fmt.Errorf("open history db: %w", err)
	}
	defer db.Close()

	query, args := buildQuery(opts)
	rows, err := db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("query urls: %w (is this a Chromium History file?)", err)
	}
	defer rows.Close()

	var visits []Visit
	for rows.Next() {
		var (
			u, title string
			vc       int
			lvt      int64
		)
		if err := rows.Scan(&u, &title, &vc, &lvt); err != nil {
			return nil, fmt.Errorf("scan url row: %w", err)
		}
		visits = append(visits, Visit{
			URL:        u,
			Title:      title,
			VisitCount: vc,
			LastVisit:  chromeToTime(lvt),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return visits, nil
}

// buildQuery assembles the SELECT over the `urls` table from the read options.
func buildQuery(opts readOptions) (string, []any) {
	var conds []string
	var args []any
	if opts.minVisits > 0 {
		conds = append(conds, "visit_count >= ?")
		args = append(args, opts.minVisits)
	}
	if !opts.since.IsZero() {
		conds = append(conds, "last_visit_time >= ?")
		args = append(args, unixToChrome(opts.since))
	}

	q := "SELECT url, COALESCE(title, ''), visit_count, last_visit_time FROM urls"
	if len(conds) > 0 {
		q += " WHERE " + strings.Join(conds, " AND ")
	}
	q += " ORDER BY last_visit_time DESC"
	if opts.limit > 0 {
		q += " LIMIT ?"
		args = append(args, opts.limit)
	}
	return q, args
}

// copyForRead copies path (and its -wal / -shm sidecars when present) into a new
// temp dir, returning the copied main-file path and a cleanup func that removes
// the dir.
func copyForRead(path string) (string, func(), error) {
	dir, err := os.MkdirTemp("", "chiton-hist-*")
	if err != nil {
		return "", func() {}, err
	}
	cleanup := func() { _ = os.RemoveAll(dir) }
	dst := filepath.Join(dir, filepath.Base(path))
	if err := copyFile(path, dst); err != nil {
		cleanup()
		return "", func() {}, fmt.Errorf("copy history db: %w", err)
	}
	// Best-effort: a checkpoint-pending DB needs its WAL/SHM (or rollback journal)
	// alongside the main file to read consistently.
	for _, suffix := range []string{"-wal", "-shm", "-journal"} {
		_ = copyFile(path+suffix, dst+suffix)
	}
	return dst, cleanup, nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	_, err = io.Copy(out, in)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	return err
}

// chromeToTime converts a Chrome last_visit_time (microseconds since 1601-01-01
// UTC) to a UTC time.Time. Zero/negative input yields the zero time.
func chromeToTime(chromeMicros int64) time.Time {
	if chromeMicros <= 0 {
		return time.Time{}
	}
	return time.UnixMicro(chromeMicros - chromeEpochOffsetMicros).UTC()
}

// unixToChrome is the inverse of chromeToTime, for a since-filter bound.
func unixToChrome(t time.Time) int64 {
	return t.UnixMicro() + chromeEpochOffsetMicros
}
