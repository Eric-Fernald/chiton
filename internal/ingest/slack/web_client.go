package slack

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// slackAPIBase is the Slack Web API root.
const slackAPIBase = "https://slack.com/api"

// maxMessages caps how many messages a single channel fetch pulls, so a busy
// channel can't run unbounded. Full-history backfill is out of scope (Phase
// 4.5 deferred); a thread fetch is naturally bounded by the thread size.
const maxMessages = 1000

// pageLimit is the per-request message count requested from Slack (the
// documented sweet spot for conversations.history; the hard max is 1000).
const pageLimit = 200

// userNegativeTTL bounds how long a failed users.info resolution is cached
// before it is retried, so a transient failure (a 429, a network blip) does not
// permanently degrade a user's name in the long-lived Events receiver while
// still avoiding a per-message refetch storm within one ingest.
const userNegativeTTL = time.Minute

// WebClient is a thin net/http Slack Web API client implementing
// ConversationSource. Like the email GmailSource it deliberately avoids the
// vendor SDK (github.com/slack-go/slack) — keeping the dependency surface
// pure-Go and minimal — and talks to the documented JSON endpoints directly
// with a bot/user OAuth token. Messages are fetched via conversations.history
// (or conversations.replies for a thread) and authors resolved via users.info.
//
// A WebClient is safe for concurrent use: the user-name cache is mutex-guarded
// so the Events receiver can share one client across concurrent HTTP handlers.
type WebClient struct {
	token  string
	client *http.Client
	base   string
	now    func() time.Time

	mu        sync.Mutex
	userCache map[string]cachedUser // resolved author name by user id
}

// cachedUser is a memoised users.info lookup. Successful resolutions are cached
// for the client lifetime; failed ones expire after userNegativeTTL.
type cachedUser struct {
	name string
	ok   bool
	at   time.Time
}

// NewWebClient builds a WebClient from an OAuth token (xoxb-… or xoxp-…).
func NewWebClient(token string) (*WebClient, error) {
	token = strings.TrimSpace(token)
	if token == "" {
		return nil, fmt.Errorf("slack: an OAuth token is required (set CHITON_SLACK_TOKEN)")
	}
	return &WebClient{
		token:     token,
		client:    &http.Client{Timeout: 30 * time.Second},
		base:      slackAPIBase,
		now:       time.Now,
		userCache: map[string]cachedUser{},
	}, nil
}

// NewWebClientFromEnv builds a WebClient from CHITON_SLACK_TOKEN.
func NewWebClientFromEnv() (*WebClient, error) {
	return NewWebClient(os.Getenv("CHITON_SLACK_TOKEN"))
}

// Name implements ConversationSource.
func (c *WebClient) Name() string { return "slack" }

// FetchConversation implements ConversationSource.
func (c *WebClient) FetchConversation(ctx context.Context, channelID, threadTS string) (Conversation, error) {
	channelID = strings.TrimSpace(channelID)
	if channelID == "" {
		return Conversation{}, fmt.Errorf("slack: a channel id is required")
	}

	conv := Conversation{ChannelID: channelID, ThreadTS: threadTS}
	// The channel name only decorates the title; a failure to resolve it (e.g.
	// a missing scope) must not abort ingestion.
	if name, err := c.resolveChannelName(ctx, channelID); err == nil {
		conv.ChannelName = name
	}

	raws, err := c.fetchRaw(ctx, channelID, threadTS)
	if err != nil {
		return Conversation{}, err
	}
	// Slack returns history newest-first; sort chronological (oldest-first).
	sort.SliceStable(raws, func(a, b int) bool { return tsLess(raws[a].TS, raws[b].TS) })

	for _, rm := range raws {
		if !rm.ingestable() {
			continue
		}
		msg := Message{
			User:     rm.User,
			Text:     cleanText(rm.Text),
			TS:       rm.TS,
			ThreadTS: rm.ThreadTS,
		}
		switch {
		case rm.User != "":
			msg.Name = c.resolveUser(ctx, rm.User)
		case rm.Username != "": // bot messages carry an inline username
			msg.Name = rm.Username
		}
		conv.Messages = append(conv.Messages, msg)
	}
	return conv, nil
}

// fetchRaw pages through conversations.history (or conversations.replies for a
// thread), honouring the cursor until exhausted or the message cap is hit.
// Messages are de-duplicated by ts because conversations.replies repeats the
// thread's parent message at the top of every page.
func (c *WebClient) fetchRaw(ctx context.Context, channelID, threadTS string) ([]rawMessage, error) {
	method := "conversations.history"
	if threadTS != "" {
		method = "conversations.replies"
	}

	var all []rawMessage
	seen := map[string]bool{}
	cursor := ""
	for {
		if err := ctx.Err(); err != nil {
			return all, err
		}
		params := url.Values{}
		params.Set("channel", channelID)
		params.Set("limit", strconv.Itoa(pageLimit))
		if threadTS != "" {
			params.Set("ts", threadTS)
		}
		if cursor != "" {
			params.Set("cursor", cursor)
		}

		var resp historyResponse
		if err := c.apiGet(ctx, method, params, &resp); err != nil {
			return all, err
		}
		for _, rm := range resp.Messages {
			if rm.TS != "" {
				if seen[rm.TS] {
					continue
				}
				seen[rm.TS] = true
			}
			all = append(all, rm)
		}
		if len(all) >= maxMessages {
			all = all[:maxMessages]
			break
		}
		cursor = resp.ResponseMetadata.NextCursor
		if cursor == "" {
			break
		}
	}
	return all, nil
}

// resolveUser returns an author's display name, caching the result. On any
// failure it falls back to the raw user id and caches that negatively (with a
// TTL) so a transient error self-heals on the next ingest.
func (c *WebClient) resolveUser(ctx context.Context, userID string) string {
	c.mu.Lock()
	ce, hit := c.userCache[userID]
	c.mu.Unlock()
	if hit && (ce.ok || c.now().Sub(ce.at) < userNegativeTTL) {
		return ce.name
	}

	name, ok := userID, false
	var resp userInfoResponse
	params := url.Values{}
	params.Set("user", userID)
	if err := c.apiGet(ctx, "users.info", params, &resp); err == nil && resp.User.ID != "" {
		name, ok = pickName(resp, userID), true
	}

	c.mu.Lock()
	c.userCache[userID] = cachedUser{name: name, ok: ok, at: c.now()}
	c.mu.Unlock()
	return name
}

// pickName prefers the display name, then real name, then handle, then the id.
func pickName(resp userInfoResponse, fallback string) string {
	switch p := resp.User.Profile; {
	case p.DisplayName != "":
		return p.DisplayName
	case p.RealName != "":
		return p.RealName
	case resp.User.RealName != "":
		return resp.User.RealName
	case resp.User.Name != "":
		return resp.User.Name
	default:
		return fallback
	}
}

// resolveChannelName best-effort resolves a channel's name via conversations.info.
func (c *WebClient) resolveChannelName(ctx context.Context, channelID string) (string, error) {
	var resp channelInfoResponse
	params := url.Values{}
	params.Set("channel", channelID)
	if err := c.apiGet(ctx, "conversations.info", params, &resp); err != nil {
		return "", err
	}
	return resp.Channel.Name, nil
}

// apiGet performs an authenticated GET and decodes a Slack JSON response. Slack
// returns HTTP 200 with {"ok":false,"error":"…"} on logical failures, so the
// envelope's ok flag is checked in addition to the HTTP status.
func (c *WebClient) apiGet(ctx context.Context, method string, params url.Values, out slackResult) error {
	reqURL := c.base + "/" + method
	if enc := params.Encode(); enc != "" {
		reqURL += "?" + enc
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("slack API %s: http %d: %s", method, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	if err := json.Unmarshal(body, out); err != nil {
		return fmt.Errorf("slack API %s: decode: %w", method, err)
	}
	if !out.isOK() {
		return fmt.Errorf("slack API %s: %s", method, out.apiError())
	}
	return nil
}

// cleanText unescapes the &amp;/&lt;/&gt; entities Slack puts in message text.
// Mention/link unwrapping (<@U…>, <http://…|text>) is left for a later pass.
func cleanText(s string) string {
	r := strings.NewReplacer("&amp;", "&", "&lt;", "<", "&gt;", ">")
	return strings.TrimSpace(r.Replace(s))
}

// tsLess orders two Slack ts strings chronologically.
func tsLess(a, b string) bool {
	ta, ea := parseSlackTS(a)
	tb, eb := parseSlackTS(b)
	if ea == nil && eb == nil {
		return ta.Before(tb)
	}
	return a < b
}

// --- Slack Web API response envelopes ---

// slackEnvelope is the common header on every Slack Web API response.
type slackEnvelope struct {
	OK               bool   `json:"ok"`
	Error            string `json:"error"`
	ResponseMetadata struct {
		NextCursor string `json:"next_cursor"`
	} `json:"response_metadata"`
}

func (e slackEnvelope) isOK() bool { return e.OK }
func (e slackEnvelope) apiError() string {
	if e.Error == "" {
		return "unknown error"
	}
	return e.Error
}

// slackResult is satisfied by any response type embedding slackEnvelope.
type slackResult interface {
	isOK() bool
	apiError() string
}

// rawMessage is one message from conversations.history / .replies.
type rawMessage struct {
	Type     string `json:"type"`
	Subtype  string `json:"subtype"`
	User     string `json:"user"`
	BotID    string `json:"bot_id"`
	Username string `json:"username"`
	Text     string `json:"text"`
	TS       string `json:"ts"`
	ThreadTS string `json:"thread_ts"`
}

// ingestable reports whether a raw message carries prose worth ingesting,
// filtering out join/leave/topic notices and empty text.
func (m rawMessage) ingestable() bool {
	if m.Type != "" && m.Type != "message" {
		return false
	}
	if strings.TrimSpace(m.Text) == "" {
		return false
	}
	switch m.Subtype {
	case "", "thread_broadcast", "bot_message", "me_message":
		return true
	default:
		return false
	}
}

type historyResponse struct {
	slackEnvelope
	Messages []rawMessage `json:"messages"`
	HasMore  bool         `json:"has_more"`
}

type userInfoResponse struct {
	slackEnvelope
	User struct {
		ID       string `json:"id"`
		Name     string `json:"name"`
		RealName string `json:"real_name"`
		Profile  struct {
			DisplayName string `json:"display_name"`
			RealName    string `json:"real_name"`
		} `json:"profile"`
	} `json:"user"`
}

type channelInfoResponse struct {
	slackEnvelope
	Channel struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	} `json:"channel"`
}

var _ ConversationSource = (*WebClient)(nil)
