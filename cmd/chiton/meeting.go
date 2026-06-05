package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/config"
	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/ingest/audio"
	urlingest "github.com/Eric-Fernald/chiton/internal/ingest/url"
)

// meetingCmd is the Phase 4 parent for meeting ingest + meeting→code linkage.
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

// ingestTranscript replaces any prior version of the transcript and upserts its
// meeting subgraph. Returns the number of nodes written.
func ingestTranscript(ctx context.Context, gs store.GraphStore, tr *meeting.Transcript) (int, error) {
	if _, err := gs.DeleteBySourceURI(ctx, tr.SourceURI); err != nil {
		return 0, fmt.Errorf("replace prior meeting: %w", err)
	}
	nodes, edges := meeting.ToGraph(tr, time.Now())
	if err := gs.UpsertNodes(ctx, nodes); err != nil {
		return 0, fmt.Errorf("upsert meeting nodes: %w", err)
	}
	if err := gs.UpsertEdges(ctx, edges); err != nil {
		return 0, fmt.Errorf("upsert meeting edges: %w", err)
	}
	return len(nodes), nil
}

func meetingIngestCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "ingest <transcript-file>...",
		Short: "Parse transcript files (.vtt/.srt/.txt) into meeting nodes",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			gs, err := openGraphStore(cfg.GraphPath, false)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

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
		RunE: func(cmd *cobra.Command, args []string) error {
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
				if err := os.WriteFile(outFile, []byte(transcriptToText(tr)), 0o644); err != nil {
					return fmt.Errorf("write %s: %w", outFile, err)
				}
				fmt.Fprintf(out, "transcribe: wrote transcript to %s\n", outFile)
			}
			if !ingest {
				return nil
			}

			cfg, err := config.Load()
			if err != nil {
				return err
			}
			gs, err := openGraphStore(cfg.GraphPath, false)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()
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
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			gs, err := openGraphStore(cfg.GraphPath, true)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			svc := meeting.NewLinkService(gs, cfg.Semantic.ConfidenceThreshold)
			specs, err := svc.Pending(cmd.Context(), limit)
			if err != nil {
				return err
			}
			b, err := json.MarshalIndent(specs, "", "  ")
			if err != nil {
				return err
			}
			if outFile != "" {
				if err := os.WriteFile(outFile, b, 0o644); err != nil {
					return fmt.Errorf("write %s: %w", outFile, err)
				}
				fmt.Fprintf(cmd.OutOrStdout(), "meeting: %d pending linkage task(s) written to %s\n", len(specs), outFile)
				return nil
			}
			fmt.Fprintln(cmd.OutOrStdout(), string(b))
			return nil
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
		RunE: func(cmd *cobra.Command, args []string) error {
			data, err := os.ReadFile(args[0])
			if err != nil {
				return fmt.Errorf("read %s: %w", args[0], err)
			}
			results, err := parseLinkResults(data)
			if err != nil {
				return err
			}
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			gs, err := openGraphStore(cfg.GraphPath, false)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			svc := meeting.NewLinkService(gs, cfg.Semantic.ConfidenceThreshold)
			out := cmd.OutOrStdout()
			var decisions, edges, linked, ambiguous int
			for _, r := range results {
				rep, err := svc.Submit(cmd.Context(), r)
				if err != nil {
					return err
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
			fmt.Fprintf(out, "submit: done — %d task(s), %d decision(s), %d edge(s), %d symbol link(s), %d ambiguous\n",
				len(results), decisions, edges, linked, ambiguous)
			return nil
		},
	}
	return cmd
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
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			gs, err := openGraphStore(cfg.GraphPath, false)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}
			defer gs.Close()

			rep, err := urlingest.New(gs).Add(cmd.Context(), args[0], urlingest.AddOptions{
				Kind:        kind,
				Author:      author,
				Contributor: contributor,
			})
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "add-url: %q (%s, %s) — %d bytes, %d section(s), %d node(s)\n",
				rep.Title, rep.Kind, rep.ContentType, rep.TextBytes, rep.Sections, rep.NodesUpserted)
			fmt.Fprintln(out, "add-url: next: 'chiton embed' + 'chiton extract pending' to index its concepts")
			return nil
		},
	}
	cmd.Flags().StringVar(&kind, "kind", "", "content hint: paper | article | tweet | video | document")
	cmd.Flags().StringVar(&author, "author", "", "author to record on the node")
	cmd.Flags().StringVar(&contributor, "contributor", "", "who added it, recorded on the node")
	return cmd
}

// transcriptToText renders a transcript to "Speaker: text" lines.
func transcriptToText(tr *meeting.Transcript) string {
	var b bytes.Buffer
	for _, t := range tr.Turns {
		if t.Speaker != "" {
			b.WriteString(t.Speaker)
			b.WriteString(": ")
		}
		b.WriteString(t.Text)
		b.WriteByte('\n')
	}
	return b.String()
}

// parseLinkResults accepts either a single LinkResult object or an array, after
// stripping a leading UTF-8 BOM (common on Windows-authored files).
func parseLinkResults(data []byte) ([]meeting.LinkResult, error) {
	data = bytes.TrimPrefix(data, []byte{0xEF, 0xBB, 0xBF})
	var arr []meeting.LinkResult
	if err := json.Unmarshal(data, &arr); err == nil {
		return arr, nil
	}
	var single meeting.LinkResult
	if err := json.Unmarshal(data, &single); err != nil {
		return nil, fmt.Errorf("parse results: expected a LinkResult object or array: %w", err)
	}
	return []meeting.LinkResult{single}, nil
}
