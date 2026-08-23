//go:build msgraph

// The Outlook/Microsoft Graph mail source. Excluded from the default build —
// see outlook_stub.go for why, and for the msgraph-free counterpart.
package email

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	msgraphsdk "github.com/microsoftgraph/msgraph-sdk-go"
	"github.com/microsoftgraph/msgraph-sdk-go/models"
	"github.com/microsoftgraph/msgraph-sdk-go/users"
)

// outlookScopes requests the application (.default) scope: app-only auth with
// the Mail.Read application permission (admin-consented). This mirrors the Teams
// transcript client and is the unattended-daemon path.
var outlookScopes = []string{"https://graph.microsoft.com/.default"}

// OutlookConfig configures the Microsoft Graph mail source (app-only auth).
type OutlookConfig struct {
	TenantID     string
	ClientID     string
	ClientSecret string
	// UserID is the mailbox owner's id or UPN: messages are read under
	// /users/{UserID}/messages (app-only has no /me).
	UserID string
}

// OutlookSource is the Microsoft Graph implementation of Source.
type OutlookSource struct {
	client *msgraphsdk.GraphServiceClient
	userID string
}

// NewOutlookSource builds an OutlookSource using client-credentials auth.
func NewOutlookSource(cfg OutlookConfig) (*OutlookSource, error) {
	if cfg.TenantID == "" || cfg.ClientID == "" || cfg.ClientSecret == "" {
		return nil, fmt.Errorf("outlook: tenant_id, client_id, and client_secret are required")
	}
	if cfg.UserID == "" {
		return nil, fmt.Errorf("outlook: user_id (the mailbox to read) is required for app-only mail reads")
	}
	cred, err := azidentity.NewClientSecretCredential(cfg.TenantID, cfg.ClientID, cfg.ClientSecret, nil)
	if err != nil {
		return nil, fmt.Errorf("outlook: client credential: %w", err)
	}
	client, err := msgraphsdk.NewGraphServiceClientWithCredentials(cred, outlookScopes)
	if err != nil {
		return nil, fmt.Errorf("outlook: graph client: %w", err)
	}
	return &OutlookSource{client: client, userID: cfg.UserID}, nil
}

// NewOutlookSourceFromEnv builds an OutlookSource from CHITON_OUTLOOK_*
// environment variables (TENANT_ID, CLIENT_ID, CLIENT_SECRET, USER_ID).
func NewOutlookSourceFromEnv() (*OutlookSource, error) {
	return NewOutlookSource(OutlookConfig{
		TenantID:     os.Getenv("CHITON_OUTLOOK_TENANT_ID"),
		ClientID:     os.Getenv("CHITON_OUTLOOK_CLIENT_ID"),
		ClientSecret: os.Getenv("CHITON_OUTLOOK_CLIENT_SECRET"),
		UserID:       os.Getenv("CHITON_OUTLOOK_USER_ID"),
	})
}

// Name implements Source.
func (s *OutlookSource) Name() string { return "outlook" }

// Messages implements Source: it lists the mailbox's most recent messages
// (optionally filtered by received date) and maps them to provider-agnostic
// Messages.
func (s *OutlookSource) Messages(ctx context.Context, opts Query) ([]Message, error) {
	top := int32(50)
	if opts.Max > 0 {
		top = int32(opts.Max)
	}
	params := &users.ItemMessagesRequestBuilderGetQueryParameters{
		Top:     &top,
		Orderby: []string{"receivedDateTime DESC"},
		Select:  []string{"subject", "from", "toRecipients", "ccRecipients", "receivedDateTime", "internetMessageId", "internetMessageHeaders", "conversationId", "body"},
	}
	if !opts.Since.IsZero() {
		filter := "receivedDateTime ge " + opts.Since.UTC().Format("2006-01-02T15:04:05Z")
		params.Filter = &filter
	}
	cfg := &users.ItemMessagesRequestBuilderGetRequestConfiguration{QueryParameters: params}

	coll, err := s.client.Users().ByUserId(s.userID).Messages().Get(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("graph list messages: %w", err)
	}
	if coll == nil {
		return nil, nil
	}
	var msgs []Message
	for _, gm := range coll.GetValue() {
		msgs = append(msgs, outlookToMessage(gm))
	}
	return msgs, nil
}

// outlookToMessage maps a Graph message to a provider-agnostic Message.
func outlookToMessage(gm models.Messageable) Message {
	m := Message{Source: "outlook"}
	if v := gm.GetId(); v != nil {
		m.ID = *v
	}
	if v := gm.GetInternetMessageId(); v != nil {
		m.MessageID = *v
	}
	if v := gm.GetConversationId(); v != nil {
		m.ThreadID = *v
	}
	// In-Reply-To lives in the internetMessageHeaders collection (requested via
	// $select), not as a first-class property — pull it out so reply threading
	// works for Outlook the same way it does for local/Gmail.
	for _, h := range gm.GetInternetMessageHeaders() {
		name, val := "", ""
		if v := h.GetName(); v != nil {
			name = *v
		}
		if v := h.GetValue(); v != nil {
			val = *v
		}
		if strings.EqualFold(name, "In-Reply-To") {
			m.InReplyTo = strings.TrimSpace(val)
			break
		}
	}
	if v := gm.GetSubject(); v != nil {
		m.Subject = *v
	}
	if v := gm.GetReceivedDateTime(); v != nil {
		m.Date = v.UTC()
	}
	m.From = recipientToAddress(gm.GetFrom())
	m.To = recipientsToAddresses(gm.GetToRecipients())
	m.Cc = recipientsToAddresses(gm.GetCcRecipients())
	if body := gm.GetBody(); body != nil {
		content := ""
		if v := body.GetContent(); v != nil {
			content = *v
		}
		if bt := body.GetContentType(); bt != nil && *bt == models.HTML_BODYTYPE {
			content = htmlToText(content)
		}
		m.Body = strings.TrimSpace(content)
	}
	return m
}

func recipientToAddress(r models.Recipientable) Address {
	if r == nil {
		return Address{}
	}
	ea := r.GetEmailAddress()
	if ea == nil {
		return Address{}
	}
	var a Address
	if v := ea.GetName(); v != nil {
		a.Name = *v
	}
	if v := ea.GetAddress(); v != nil {
		a.Email = *v
	}
	return a
}

func recipientsToAddresses(rs []models.Recipientable) []Address {
	var out []Address
	for _, r := range rs {
		if a := recipientToAddress(r); a.Email != "" || a.Name != "" {
			out = append(out, a)
		}
	}
	return out
}

var _ Source = (*OutlookSource)(nil)
