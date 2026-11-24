package email

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/ingest/bearer"
)

// gmailBase is the Gmail REST endpoint for the authenticated user ("me").
const gmailBase = "https://gmail.googleapis.com/gmail/v1/users/me"

// GmailSource reads mail via the Gmail REST API with an OAuth2 access token, fetching raw RFC 5322 messages
// so they go through the same parser as LocalSource.
type GmailSource struct {
	token  string
	client *http.Client
	base   string
}

// NewGmailSource builds a GmailSource from an OAuth2 access token.
func NewGmailSource(token string) (*GmailSource, error) {
	token = strings.TrimSpace(token)
	if token == "" {
		return nil, fmt.Errorf("gmail: an OAuth2 access token is required (set CHITON_GMAIL_TOKEN)")
	}
	return &GmailSource{
		token:  token,
		client: &http.Client{Timeout: 30 * time.Second},
		base:   gmailBase,
	}, nil
}

// NewGmailSourceFromEnv builds a GmailSource from CHITON_GMAIL_TOKEN.
func NewGmailSourceFromEnv() (*GmailSource, error) {
	return NewGmailSource(os.Getenv("CHITON_GMAIL_TOKEN"))
}

// Name implements Source.
func (s *GmailSource) Name() string { return "gmail" }

// Messages implements Source: it lists message ids matching the query window,
// then fetches and parses each raw message.
func (s *GmailSource) Messages(ctx context.Context, opts Query) ([]Message, error) {
	limit := opts.Max
	if limit <= 0 {
		limit = 50
	}
	q := strings.TrimSpace(opts.Mailbox)
	if !opts.Since.IsZero() {
		q = strings.TrimSpace(q + " after:" + opts.Since.Format("2006/01/02"))
	}

	listURL := fmt.Sprintf("%s/messages?maxResults=%d", s.base, limit)
	if q != "" {
		listURL += "&q=" + url.QueryEscape(q)
	}
	var list struct {
		Messages []struct {
			ID string `json:"id"`
		} `json:"messages"`
	}
	if err := s.getJSON(ctx, listURL, &list); err != nil {
		return nil, err
	}

	var msgs []Message
	for _, ref := range list.Messages {
		if len(msgs) >= limit {
			break
		}
		var raw struct {
			Raw      string `json:"raw"`
			ThreadID string `json:"threadId"`
		}
		if err := s.getJSON(ctx, fmt.Sprintf("%s/messages/%s?format=raw", s.base, url.PathEscape(ref.ID)), &raw); err != nil {
			return msgs, err
		}
		decoded, err := decodeBase64URL(raw.Raw)
		if err != nil {
			continue // skip an undecodable message rather than abort the batch
		}
		m, err := parseMessage(decoded)
		if err != nil {
			continue
		}
		m.ID = ref.ID
		m.ThreadID = raw.ThreadID
		m.Source = "gmail"
		msgs = append(msgs, m)
	}
	return msgs, nil
}

// getJSON performs an authenticated GET and decodes a JSON response.
func (s *GmailSource) getJSON(ctx context.Context, reqURL string, out any) error {
	status, body, err := bearer.Get(ctx, s.client, reqURL, s.token)
	if err != nil {
		return err
	}
	if status < 200 || status >= 300 {
		return fmt.Errorf("gmail API %d: %s", status, strings.TrimSpace(string(body)))
	}
	return json.Unmarshal(body, out)
}

// decodeBase64URL decodes Gmail's URL-safe base64 (padded or unpadded).
func decodeBase64URL(s string) ([]byte, error) {
	if dec, err := base64.URLEncoding.DecodeString(s); err == nil {
		return dec, nil
	}
	return base64.RawURLEncoding.DecodeString(s)
}

var _ Source = (*GmailSource)(nil)
