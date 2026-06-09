package main

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/ingest/email"
)

// emailCmd is the Phase 6 parent for email ingestion.
func emailCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "email",
		Short: "Ingest email (local files, Outlook, or Gmail) into the graph",
		Long: `Ingest email as graph nodes: an email node per message with doc_section body
children (so concept extraction picks them up for free) and PII-tagged person
nodes for the participants. Replies link to the message they answer.

  chiton email ingest <file|dir>   parse local .eml / .mbox files
  chiton email gmail               fetch via the Gmail API (CHITON_GMAIL_TOKEN)
  chiton email outlook             fetch via Microsoft Graph (CHITON_OUTLOOK_*)`,
	}
	cmd.AddCommand(emailIngestCmd(), emailGmailCmd(), emailOutlookCmd())
	return cmd
}

// runEmailSource opens the store read-write, ingests from src, and prints a
// summary — the shared tail of every email subcommand.
func runEmailSource(cmd *cobra.Command, src email.Source, opts email.Query) error {
	gs, closeStore, err := openWritableStore()
	if err != nil {
		return err
	}
	defer closeStore()

	rep, err := email.NewIngester(gs).Ingest(cmd.Context(), src, opts)
	if err != nil {
		return err
	}
	fmt.Fprintf(cmd.OutOrStdout(),
		"email: ingested %d message(s) from %s — %d nodes, %d persons, %d thread links\n",
		rep.Messages, rep.Source, rep.NodesUpserted, rep.Persons, rep.References)
	return nil
}

func emailIngestCmd() *cobra.Command {
	var since string
	var max int
	cmd := &cobra.Command{
		Use:   "ingest <file-or-dir>",
		Short: "Ingest local .eml / .mbox files",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			sinceT, err := parseSince(since)
			if err != nil {
				return err
			}
			return runEmailSource(cmd, email.NewLocalSource(args[0]), email.Query{Since: sinceT, Max: max})
		},
	}
	cmd.Flags().StringVar(&since, "since", "", "Only messages at/after this time (YYYY-MM-DD, RFC3339, or a duration like 168h)")
	cmd.Flags().IntVar(&max, "max", 0, "Max messages to ingest (0 = all)")
	return cmd
}

func emailGmailCmd() *cobra.Command {
	var since, query string
	var max int
	cmd := &cobra.Command{
		Use:   "gmail",
		Short: "Ingest mail via the Gmail API (requires CHITON_GMAIL_TOKEN)",
		Long: `Fetch messages over the Gmail REST API and ingest them.

Set CHITON_GMAIL_TOKEN to an OAuth2 access token with the gmail.readonly scope.
--query takes a Gmail search expression (e.g. "from:boss is:important").`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			sinceT, err := parseSince(since)
			if err != nil {
				return err
			}
			src, err := email.NewGmailSourceFromEnv()
			if err != nil {
				return err
			}
			return runEmailSource(cmd, src, email.Query{Since: sinceT, Max: max, Mailbox: query})
		},
	}
	cmd.Flags().StringVar(&since, "since", "", "Only messages at/after this time (YYYY-MM-DD, RFC3339, or a duration like 168h)")
	cmd.Flags().StringVar(&query, "query", "", "Gmail search query")
	cmd.Flags().IntVar(&max, "max", 50, "Max messages to fetch")
	return cmd
}

func emailOutlookCmd() *cobra.Command {
	var since string
	var max int
	cmd := &cobra.Command{
		Use:   "outlook",
		Short: "Ingest mail via Microsoft Graph (requires CHITON_OUTLOOK_*)",
		Long: `Fetch messages over Microsoft Graph and ingest them.

Authentication is app-only (client credentials). Set:
  CHITON_OUTLOOK_TENANT_ID, CHITON_OUTLOOK_CLIENT_ID,
  CHITON_OUTLOOK_CLIENT_SECRET, CHITON_OUTLOOK_USER_ID (the mailbox to read)
The app needs the Mail.Read application permission (admin-consented).`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			sinceT, err := parseSince(since)
			if err != nil {
				return err
			}
			src, err := email.NewOutlookSourceFromEnv()
			if err != nil {
				return err
			}
			return runEmailSource(cmd, src, email.Query{Since: sinceT, Max: max})
		},
	}
	cmd.Flags().StringVar(&since, "since", "", "Only messages at/after this time (YYYY-MM-DD, RFC3339, or a duration like 168h)")
	cmd.Flags().IntVar(&max, "max", 50, "Max messages to fetch")
	return cmd
}
