package main

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/ingest/teams"
)

// teamsCmd is the Phase 4 parent for Microsoft Teams transcript ingest.
func teamsCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "teams",
		Short: "Ingest Microsoft Teams meeting transcripts",
		Long: `Fetch Teams meeting transcripts over Microsoft Graph and ingest them as
meetings, then link their decisions to code like any other transcript.

Authentication is app-only (client credentials). Set:
  CHITON_TEAMS_TENANT_ID, CHITON_TEAMS_CLIENT_ID, CHITON_TEAMS_CLIENT_SECRET,
  CHITON_TEAMS_USER_ID  (the meeting organizer whose meetings to read)
The app needs the OnlineMeetingTranscript.Read.All application permission
(admin-consented) and an application access policy for the organizer.

Build requirement: these commands need a binary built with -tags msgraph
(make build-msgraph). Released binaries omit the Microsoft Graph SDK because it
adds ~117 MB. Without the tag, the commands exist but fail with a clear error.`,
	}
	cmd.AddCommand(teamsFetchCmd(), teamsWebhookCmd())
	return cmd
}

func teamsFetchCmd() *cobra.Command {
	var meetingID string

	cmd := &cobra.Command{
		Use:   "fetch --meeting <online-meeting-id>",
		Short: "Fetch and ingest all transcripts for one Teams meeting",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if meetingID == "" {
				return fmt.Errorf("--meeting <online-meeting-id> is required")
			}
			src, err := teams.NewGraphSourceFromEnv()
			if err != nil {
				return err
			}
			_, gs, err := loadStore(false)
			if err != nil {
				return err
			}
			defer gs.Close()

			rep, err := teams.NewIngester(gs, src).IngestMeeting(cmd.Context(), meetingID)
			if err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(),
				"teams: ingested meeting %s — %d transcript(s), %d turns, %d nodes\n",
				rep.MeetingID, rep.Transcripts, rep.Turns, rep.NodesUpdated)
			return nil
		},
	}
	cmd.Flags().StringVar(&meetingID, "meeting", "", "Teams online meeting id")
	return cmd
}

func teamsWebhookCmd() *cobra.Command {
	var addr, stateFile string

	cmd := &cobra.Command{
		Use:   "serve-webhook",
		Short: "Run the Graph change-notification webhook receiver (auto-ingest new transcripts)",
		Long: `Run an HTTP receiver for Microsoft Graph change notifications. On a
transcript-created notification it fetches and ingests that meeting's
transcripts. Register a Graph subscription pointing at this endpoint with a
clientState matching CHITON_TEAMS_CLIENT_STATE (or the contents of the file
named by --client-state-file).

Bind to localhost behind a tunnel/reverse proxy; Graph requires HTTPS and a
public URL for the subscription itself.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if err := secretFlagRemoved(cmd, "client-state", "client-state-file", "CHITON_TEAMS_CLIENT_STATE"); err != nil {
				return err
			}
			clientState, err := readSecret(stateFile, "CHITON_TEAMS_CLIENT_STATE")
			if err != nil {
				return err
			}
			if clientState == "" {
				// Fail fast rather than starting a receiver that cannot
				// authenticate anything. Matches `slack serve-events`.
				return fmt.Errorf("a Graph subscription clientState is required: set CHITON_TEAMS_CLIENT_STATE or pass --client-state-file <path>")
			}
			warnIfNonLoopback(cmd.ErrOrStderr(), addr, "the Teams webhook receiver")
			src, err := teams.NewGraphSourceFromEnv()
			if err != nil {
				return err
			}
			_, gs, err := loadStore(false)
			if err != nil {
				return err
			}
			defer gs.Close()

			ing := teams.NewIngester(gs, src)
			out := cmd.OutOrStdout()
			logf := errLogf(cmd)
			handler := teams.NewWebhookHandler(clientState, func(ctx context.Context, c teams.ChangeItem) error {
				meetingID, _, ok := teams.ParseTranscriptResource(c.Resource)
				if !ok {
					logf("teams webhook: ignoring non-transcript resource %q", c.Resource)
					return nil
				}
				rep, err := ing.IngestMeeting(ctx, meetingID)
				if err != nil {
					return err
				}
				fmt.Fprintf(out, "teams webhook: ingested meeting %s — %d transcript(s), %d turns\n",
					meetingID, rep.Transcripts, rep.Turns)
				return nil
			}, logf)
			defer handler.Close() // runs before the deferred gs.Close, so queued ingests finish first

			mux := http.NewServeMux()
			mux.Handle("/notifications", handler)
			srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second, BaseContext: baseContext(cmd.Context())}

			fmt.Fprintf(out, "teams: webhook listening on %s/notifications (Ctrl-C to stop)\n", addr)
			return serveUntilDone(cmd.Context(), srv.ListenAndServe, srv.Shutdown)
		},
	}
	cmd.Flags().StringVar(&addr, "addr", "127.0.0.1:8444", "HTTP listen address (loopback by default; put a tunnel or reverse proxy in front)")
	cmd.Flags().StringVar(&stateFile, "client-state-file", "", "file holding the shared secret that validates notifications (default: CHITON_TEAMS_CLIENT_STATE)")
	cmd.Flags().String("client-state", "", "removed: use --client-state-file or CHITON_TEAMS_CLIENT_STATE")
	_ = cmd.Flags().MarkHidden("client-state")
	return cmd
}
