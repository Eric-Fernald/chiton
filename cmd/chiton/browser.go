package main

import (
	"fmt"
	"os"
	"os/signal"
	"time"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/config"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/ingest/browser"
)

// browserCmd is the Phase 6 parent for local web-browser history ingestion.
func browserCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "browser",
		Short: "Ingest local web-browser history (Chrome/Edge/Brave) into the graph",
		Long: `Ingest the pages you have visited from a Chromium-family browser's local
History database as url nodes (title, visit count, last-visit time).

  chiton browser ingest         detect installed browsers and ingest each
  chiton browser ingest --db <path>   ingest a specific History file
  chiton browser watch          re-ingest automatically as history changes

Only page metadata is read — chiton never fetches the pages, so this does no
network I/O. Use 'chiton add-url' to pull a page's content into the graph.`,
	}
	cmd.AddCommand(browserIngestCmd(), browserWatchCmd())
	return cmd
}

func browserIngestCmd() *cobra.Command {
	var dbPath, name, since string
	var limit, minVisits int

	cmd := &cobra.Command{
		Use:   "ingest",
		Short: "Ingest browser history (auto-detect, or --db <path>)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			sinceT, err := parseSince(since)
			if err != nil {
				return err
			}
			gs, closeStore, err := openWritableStore()
			if err != nil {
				return err
			}
			defer closeStore()

			ing := browser.New(gs)
			opts := browser.Options{Limit: limit, Since: sinceT, MinVisits: minVisits}
			out := cmd.OutOrStdout()

			if dbPath != "" {
				rep, err := ing.IngestDB(cmd.Context(), dbPath, name, opts)
				if err != nil {
					return err
				}
				fmt.Fprintf(out, "browser: ingested %d pages from %s\n", rep.NodesUpserted, rep.DB)
				return nil
			}

			reports, err := ing.IngestAuto(cmd.Context(), opts)
			if err != nil {
				return err
			}
			total := 0
			for _, r := range reports {
				fmt.Fprintf(out, "browser: %s — %d pages (%s)\n", r.Browser, r.NodesUpserted, r.DB)
				total += r.NodesUpserted
			}
			fmt.Fprintf(out, "browser: ingested %d pages from %d database(s)\n", total, len(reports))
			return nil
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", "", "Path to a specific History SQLite file (default: auto-detect)")
	cmd.Flags().StringVar(&name, "browser", "", "Browser label for --db (e.g. chrome, edge)")
	cmd.Flags().IntVar(&limit, "limit", 0, "Max most-recent history rows per database (0 = default 1000)")
	cmd.Flags().StringVar(&since, "since", "", "Only visits at/after this time (YYYY-MM-DD, RFC3339, or a duration like 168h)")
	cmd.Flags().IntVar(&minVisits, "min-visits", 0, "Skip URLs visited fewer than this many times")
	return cmd
}

func browserWatchCmd() *cobra.Command {
	var dbPath, name string
	var limit, minVisits, debounceMS int

	cmd := &cobra.Command{
		Use:   "watch",
		Short: "Watch browser history and re-ingest as it changes",
		RunE: func(cmd *cobra.Command, _ []string) error {
			var dbs []browser.DetectedDB
			if dbPath != "" {
				dbs = []browser.DetectedDB{{Browser: name, Path: dbPath}}
			} else {
				dbs = browser.DetectDBs()
				if len(dbs) == 0 {
					return fmt.Errorf("browser watch: no Chrome/Edge/Brave history found; pass --db <path>")
				}
			}

			gs, closeStore, err := openWritableStore()
			if err != nil {
				return err
			}
			defer closeStore()

			ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt)
			defer stop()

			ing := browser.New(gs)
			opts := browser.Options{Limit: limit, MinVisits: minVisits}
			logf := func(format string, args ...any) {
				fmt.Fprintf(cmd.ErrOrStderr(), format+"\n", args...)
			}
			fmt.Fprintf(cmd.OutOrStdout(), "browser: watching %d history database(s) (Ctrl-C to stop)\n", len(dbs))
			return ing.Watch(ctx, dbs, opts, time.Duration(debounceMS)*time.Millisecond, logf)
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", "", "Path to a specific History SQLite file (default: auto-detect)")
	cmd.Flags().StringVar(&name, "browser", "", "Browser label for --db (e.g. chrome, edge)")
	cmd.Flags().IntVar(&limit, "limit", 0, "Max most-recent history rows per database per pass (0 = default 1000)")
	cmd.Flags().IntVar(&minVisits, "min-visits", 0, "Skip URLs visited fewer than this many times")
	cmd.Flags().IntVar(&debounceMS, "debounce-ms", 5000, "Idle window before re-ingesting after a change")
	return cmd
}

// openWritableStore opens the configured graph store read-write and returns it
// with a close func, mirroring the pattern the teams/meeting ingest commands use.
func openWritableStore() (gs store.GraphStore, closeFn func(), err error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, nil, err
	}
	s, err := openGraphStore(cfg.GraphPath, false)
	if err != nil {
		return nil, nil, fmt.Errorf("open store: %w", err)
	}
	return s, func() { s.Close() }, nil
}

// parseSince parses a --since flag: an empty string yields the zero time; a
// YYYY-MM-DD or RFC3339 timestamp is taken literally; a Go duration (e.g. 168h)
// is interpreted as "that long ago".
func parseSince(s string) (time.Time, error) {
	if s == "" {
		return time.Time{}, nil
	}
	if t, err := time.Parse("2006-01-02", s); err == nil {
		return t.UTC(), nil
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.UTC(), nil
	}
	if d, err := time.ParseDuration(s); err == nil {
		return time.Now().Add(-d).UTC(), nil
	}
	return time.Time{}, fmt.Errorf("invalid --since %q: use YYYY-MM-DD, RFC3339, or a duration like 168h", s)
}
