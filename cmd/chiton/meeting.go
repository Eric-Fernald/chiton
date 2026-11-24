package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/ingest/audio"
	urlingest "github.com/Eric-Fernald/chiton/internal/ingest/url"
)

// meetingCmd is the parent for meeting ingest + meeting→code linkage.
func meetingCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "meeting",
		Short: "Ingest meeting transcripts and link their decisions to code",
		Long: `Ingest meeting transcripts (WebVTT, SRT, or "Speaker: text" notes — the formats
Teams, Zoom, and whisper.cpp export) into the graph, then link the decisions made
in them back to the code symbols they reference via the subagent flow:

  chiton meeting ingest standup.vtt          # parse a transcript into the graph
  chiton meeting transcribe call.mp3         # whisper-transcribe audio, then ingest
  chiton meeting pending --out tasks.json    # emit meeting→code linkage tasks
  # ... run each task, producing results.json ...
  chiton meeting submit results.json         # ingest decisions + DECIDED_IN edges`,
	}
	cmd.AddCommand(meetingIngestCmd(), meetingTranscribeCmd(), meetingPendingCmd(), meetingSubmitCmd())
	return cmd
}

// meetingFileURI returns the stable meeting source_uri for a local file, so a
// re-ingest of the same file replaces rather than duplicates.
func meetingFileURI(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		abs = path
	}
	return meeting.MeetingURI("file/" + abs)
}

// ingestTranscript replaces any prior version of the transcript with its meeting subgraph, keeping linkage
// edges on surviving nodes. Returns the number of nodes written.
func ingestTranscript(ctx context.Context, gs store.GraphStore, tr *meeting.Transcript) (int, error) {
	return meeting.Upsert(ctx, gs, tr, time.Now(), "meeting")
}

func meetingIngestCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "ingest <transcript-file>...",
		Short: "Parse transcript files (.vtt/.srt/.txt) into meeting nodes",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			_, gs, err := loadStore(false)
			if err != nil {
				return err
			}
			defer closeGraphStore(gs.Close, &err)

			out := cmd.OutOrStdout()
			ctx := cmd.Context()
			var totalMeetings, totalNodes int
			for _, path := range args {
				data, err := os.ReadFile(path)
				if err != nil {
					fmt.Fprintf(cmd.ErrOrStderr(), "meeting: skip %s: %v\n", path, err)
					continue
				}
				tr, err := meeting.Parse(filepath.Base(path), data)
				if err != nil {
					fmt.Fprintf(cmd.ErrOrStderr(), "meeting: skip %s: %v\n", path, err)
					continue
				}
				tr.SourceURI = meetingFileURI(path)
				n, err := ingestTranscript(ctx, gs, tr)
				if err != nil {
					return err
				}
				totalMeetings++
				totalNodes += n
				fmt.Fprintf(out, "meeting: ingested %s — %d turns, %d nodes\n", tr.Title, len(tr.Turns), n)
			}
			fmt.Fprintf(out, "meeting: done — %d meeting(s), %d nodes\n", totalMeetings, totalNodes)
			fmt.Fprintln(out, "meeting: next: 'chiton meeting pending' to link decisions to code")
			return nil
		},
	}
	return cmd
}

func meetingTranscribeCmd() *cobra.Command {
	var whisperBin, model, language, outFile string
	var ingest bool

	cmd := &cobra.Command{
		Use:   "transcribe <audio-or-video-file>",
		Short: "Transcribe audio via whisper, then ingest it as a meeting",
		Long: `Transcribe a local audio/video file by shelling out to a whisper CLI
(OpenAI 'whisper' or whisper.cpp 'whisper-cli'), then ingest the result as a
meeting. Build chiton with -tags whisper_cgo for the in-process whisper.cpp
backend instead of the shell-out.`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			audioPath := args[0]
			tr, err := audio.New(audio.Config{
				Binary:   whisperBin,
				Model:    model,
				Language: language,
			}).Transcribe(cmd.Context(), audioPath)
			if err != nil {
				return err
			}
			tr.SourceURI = meetingFileURI(audioPath)
			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "transcribe: %d turns from %s\n", len(tr.Turns), audioPath)

			if outFile != "" {
				if err := os.WriteFile(outFile, []byte(tr.Text()), 0o644); err != nil {
					return fmt.Errorf("write %s: %w", outFile, err)
				}
				fmt.Fprintf(out, "transcribe: wrote transcript to %s\n", outFile)
			}
			if !ingest {
				return nil
			}

			_, gs, err := loadStore(false)
			if err != nil {
				return err
			}
			defer closeGraphStore(gs.Close, &err)
			n, err := ingestTranscript(cmd.Context(), gs, tr)
			if err != nil {
				return err
			}
			fmt.Fprintf(out, "transcribe: ingested %s — %d nodes\n", tr.Title, n)
			return nil
		},
	}
	cmd.Flags().StringVar(&whisperBin, "whisper-bin", "", "whisper CLI to use (default: auto-detect on PATH)")
	cmd.Flags().StringVar(&model, "model", "", "whisper model (OpenAI name e.g. 'base', or whisper.cpp ggml path)")
	cmd.Flags().StringVar(&language, "language", "", "language code (e.g. 'en'; default: auto-detect)")
	cmd.Flags().StringVar(&outFile, "out", "", "also write the transcript text to this file")
	cmd.Flags().BoolVar(&ingest, "ingest", true, "ingest the transcript into the graph")
	return cmd
}

func meetingPendingCmd() *cobra.Command {
	var outFile string
	var limit int

	cmd := &cobra.Command{
		Use:   "pending",
		Short: "Emit meeting→code linkage task specs as JSON",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, gs, err := loadStore(true)
			if err != nil {
				return err
			}
			defer gs.Close()

			svc := meeting.NewLinkService(gs, cfg.Semantic.ConfidenceThreshold)
			specs, err := svc.Pending(cmd.Context(), limit)
			if err != nil {
				return err
			}
			return emitSpecs(cmd.OutOrStdout(), specs, outFile,
				fmt.Sprintf("meeting: %d pending linkage task(s)", len(specs)))
		},
	}
	cmd.Flags().StringVar(&outFile, "out", "", "write specs to this file instead of stdout")
	cmd.Flags().IntVar(&limit, "limit", 0, "max specs (0 = all)")
	return cmd
}

func meetingSubmitCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "submit <results.json>",
		Short: "Ingest meeting-linkage results (a LinkResult object or array)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			data, err := os.ReadFile(args[0])
			if err != nil {
				return fmt.Errorf("read %s: %w", args[0], err)
			}
			results, err := parseOneOrMany[meeting.LinkResult](data, "a LinkResult")
			if err != nil {
				return err
			}
			cfg, gs, err := loadStore(false)
			if err != nil {
				return err
			}
			defer closeGraphStore(gs.Close, &err)

			svc := meeting.NewLinkService(gs, cfg.Semantic.ConfidenceThreshold)
			return runMeetingSubmit(cmd.Context(), svc, results, cmd.OutOrStdout())
		},
	}
	return cmd
}

// runMeetingSubmit applies each result and prints what happened. One failed result does not stop the rest.
// It returns an error when any failed.
func runMeetingSubmit(ctx context.Context, svc *meeting.LinkService, results []meeting.LinkResult, out io.Writer) error {
	var decisions, edges, linked, ambiguous, failed int
	for _, r := range results {
		rep, err := svc.Submit(ctx, r)
		if err != nil {
			failed++
			fmt.Fprintf(out, "submit: %s: failed — %v\n", r.TaskID, err)
			continue
		}
		decisions += rep.DecisionsAdded
		edges += rep.EdgesAdded
		linked += rep.LinkedSymbols
		ambiguous += rep.Ambiguous
		fmt.Fprintf(out, "submit: %s — %d decision(s), %d edge(s), %d symbol(s) linked, %d ambiguous\n",
			rep.TaskID, rep.DecisionsAdded, rep.EdgesAdded, rep.LinkedSymbols, rep.Ambiguous)
		if len(rep.UnresolvedRefs) > 0 {
			fmt.Fprintf(out, "submit:   unresolved refs: %v\n", rep.UnresolvedRefs)
		}
	}
	fmt.Fprintf(out, "submit: done — %d task(s), %d decision(s), %d edge(s), %d symbol link(s), %d ambiguous, %d failed\n",
		len(results), decisions, edges, linked, ambiguous, failed)
	if failed > 0 {
		return fmt.Errorf("%d of %d meeting-linkage result(s) failed", failed, len(results))
	}
	return nil
}

// addURLCmd ingests an external URL or local file as a url node + doc sections.
func addURLCmd() *cobra.Command {
	var kind, author, contributor string

	cmd := &cobra.Command{
		Use:   "add-url <url-or-path>",
		Short: "Ingest a web page, PDF, or YouTube transcript as a url node",
		Long: `Fetch a URL (article HTML, paper PDF, or YouTube transcript via yt-dlp) or a
local file and ingest its readable text as a url node with doc_section children,
so the existing 'chiton extract' / pending_extractions flow extracts its concepts.`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			_, gs, err := loadStore(false)
			if err != nil {
				return err
			}
			defer closeGraphStore(gs.Close, &err)

			rep, err := urlingest.New(gs).Add(cmd.Context(), args[0], urlingest.AddOptions{
				Kind:        kind,
				Author:      author,
				Contributor: contributor,
				AllowLocal:  true, // the CLI user names the file; only MCP callers are held to http(s)
			})
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "add-url: %q (%s, %s) — %d bytes, %d section(s), %d node(s)\n",
				rep.Title, rep.Kind, rep.ContentType, rep.TextBytes, rep.Sections, rep.NodesUpserted)
			if rep.Truncated {
				fmt.Fprintf(cmd.ErrOrStderr(), "add-url: warning: the text ran past the section cap; only the first %d section(s) were kept\n", rep.Sections)
			}
			fmt.Fprintln(out, "add-url: next: 'chiton embed' + 'chiton extract pending' to index its concepts")
			return nil
		},
	}
	cmd.Flags().StringVar(&kind, "kind", "", "content hint: paper | article | tweet | video | document")
	cmd.Flags().StringVar(&author, "author", "", "author to record on the node")
	cmd.Flags().StringVar(&contributor, "contributor", "", "who added it, recorded on the node")
	return cmd
}
