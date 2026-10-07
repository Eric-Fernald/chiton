package main

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/ingest/slack"
)

// slackCmd is the parent for Slack conversation ingest.
func slackCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "slack",
		Short: "Ingest Slack channel and thread conversations",
		Long: `Fetch Slack conversations over the Web API and ingest them as meetings,
then link their decisions to code like any other transcript. A channel or
thread becomes a meeting subgraph (meeting + turns + persons), so
search_meetings and decisions_about serve Slack discussions unchanged.

Authentication uses a bot/user OAuth token. Set:
  CHITON_SLACK_TOKEN            xoxb-… (needs channels:history / groups:history
                                and users:read to resolve author names)
For the Events API receiver also set:
  CHITON_SLACK_SIGNING_SECRET   the app signing secret (request verification)`,
	}
	cmd.AddCommand(slackFetchCmd(), slackEventsCmd())
	return cmd
}

func slackFetchCmd() *cobra.Command {
	var channelID, threadTS string

	cmd := &cobra.Command{
		Use:   "fetch --channel <channel-id> [--thread <thread-ts>]",
		Short: "Fetch and ingest a Slack channel (or a single thread)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if channelID == "" {
				return fmt.Errorf("--channel <channel-id> is required")
			}
			src, err := slack.NewWebClientFromEnv()
			if err != nil {
				return err
			}
			_, gs, err := loadStore(false)
			if err != nil {
				return err
			}
			defer gs.Close()

			rep, err := slack.NewIngester(gs, src).IngestConversation(cmd.Context(), channelID, threadTS)
			if err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(),
				"slack: ingested %s — %d message(s), %d turns, %d nodes\n",
				slackScope(rep), rep.Messages, rep.Turns, rep.NodesUpserted)
			return nil
		},
	}
	cmd.Flags().StringVar(&channelID, "channel", "", "Slack channel id (e.g. C0123ABCD)")
	cmd.Flags().StringVar(&threadTS, "thread", "", "thread ts to ingest a single thread (optional)")
	return cmd
}

func slackEventsCmd() *cobra.Command {
	var addr, secretFile string

	cmd := &cobra.Command{
		Use:   "serve-events",
		Short: "Run the Slack Events API receiver (auto-ingest new messages)",
		Long: `Run an HTTP receiver for the Slack Events API. On a new message it
re-ingests that message's thread (or channel) so the graph stays current.
Configure the app's Request URL to point at this endpoint; chiton performs the
url_verification handshake and verifies every request's X-Slack-Signature.

Set CHITON_SLACK_SIGNING_SECRET, or pass --signing-secret-file with the path of
a file that holds it. Bind to localhost behind a tunnel/reverse proxy; Slack
requires a public HTTPS Request URL.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if err := secretFlagRemoved(cmd, "signing-secret", "signing-secret-file", "CHITON_SLACK_SIGNING_SECRET"); err != nil {
				return err
			}
			signingSecret, err := readSecret(secretFile, "CHITON_SLACK_SIGNING_SECRET")
			if err != nil {
				return err
			}
			if signingSecret == "" {
				// Fail fast with a clear message rather than starting a server
				// that rejects every (unauthenticatable) request.
				return fmt.Errorf("a Slack signing secret is required: set CHITON_SLACK_SIGNING_SECRET or pass --signing-secret-file <path>")
			}
			warnIfNonLoopback(cmd.ErrOrStderr(), addr, "the Slack events receiver")
			src, err := slack.NewWebClientFromEnv()
			if err != nil {
				return err
			}
			_, gs, err := loadStore(false)
			if err != nil {
				return err
			}
			defer gs.Close()

			ing := slack.NewIngester(gs, src)
			out := cmd.OutOrStdout()
			handler := slack.NewEventsHandler(signingSecret, func(ctx context.Context, ev slack.MessageEvent) error {
				// A threaded reply re-ingests only its thread, and a top-level message the whole channel. Re-ingest
				// replaces the conversation by its slack:// URI, so this is idempotent.
				rep, err := ing.IngestConversation(ctx, ev.ChannelID, ev.Thread())
				if err != nil {
					return err
				}
				fmt.Fprintf(out, "slack events: ingested %s — %d message(s), %d turns\n",
					slackScope(rep), rep.Messages, rep.Turns)
				return nil
			}, errLogf(cmd))
			defer handler.Close() // runs before the deferred gs.Close, so queued re-ingests finish first

			mux := http.NewServeMux()
			mux.Handle("/slack/events", handler)
			srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second, BaseContext: baseContext(cmd.Context())}

			fmt.Fprintf(out, "slack: events receiver listening on %s/slack/events (Ctrl-C to stop)\n", addr)
			return serveUntilDone(cmd.Context(), srv.ListenAndServe, srv.Shutdown)
		},
	}
	cmd.Flags().StringVar(&addr, "addr", "127.0.0.1:8445", "HTTP listen address (loopback by default; put a tunnel or reverse proxy in front)")
	cmd.Flags().StringVar(&secretFile, "signing-secret-file", "", "file holding the Slack app signing secret (default: CHITON_SLACK_SIGNING_SECRET)")
	cmd.Flags().String("signing-secret", "", "removed: use --signing-secret-file or CHITON_SLACK_SIGNING_SECRET")
	_ = cmd.Flags().MarkHidden("signing-secret")
	return cmd
}

// slackScope describes an ingested conversation: "#<channel>" or "thread <ts> in <channel>".
func slackScope(rep slack.Report) string {
	if rep.ThreadTS != "" {
		return fmt.Sprintf("thread %s in %s", rep.ThreadTS, rep.ChannelID)
	}
	return "#" + rep.ChannelID
}
