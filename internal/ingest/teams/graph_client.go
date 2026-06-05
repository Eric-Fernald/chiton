package teams

import (
	"context"
	"fmt"
	"os"

	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	abstractions "github.com/microsoft/kiota-abstractions-go"
	msgraphsdk "github.com/microsoftgraph/msgraph-sdk-go"
	"github.com/microsoftgraph/msgraph-sdk-go/users"
)

// graphScopes requests the application (.default) scope: app-only auth with the
// OnlineMeetingTranscript.Read.All application permission (admin-consented) plus
// an application access policy granting the app rights to the organizer's
// meetings. See docs — this is the unattended-daemon path the plan calls for.
var graphScopes = []string{"https://graph.microsoft.com/.default"}

// GraphConfig configures the Microsoft Graph transcript source (app-only auth).
type GraphConfig struct {
	TenantID     string
	ClientID     string
	ClientSecret string
	// UserID is the meeting organizer's id or UPN: transcripts are read under
	// /users/{UserID}/onlineMeetings/... (app-only has no /me).
	UserID string
}

// GraphSource is the Microsoft Graph implementation of TranscriptSource.
type GraphSource struct {
	client *msgraphsdk.GraphServiceClient
	userID string
}

// NewGraphSource builds a GraphSource using client-credentials (app-only) auth.
func NewGraphSource(cfg GraphConfig) (*GraphSource, error) {
	if cfg.TenantID == "" || cfg.ClientID == "" || cfg.ClientSecret == "" {
		return nil, fmt.Errorf("teams: tenant_id, client_id, and client_secret are required")
	}
	if cfg.UserID == "" {
		return nil, fmt.Errorf("teams: user_id (the meeting organizer) is required for app-only transcript reads")
	}
	cred, err := azidentity.NewClientSecretCredential(cfg.TenantID, cfg.ClientID, cfg.ClientSecret, nil)
	if err != nil {
		return nil, fmt.Errorf("teams: client credential: %w", err)
	}
	client, err := msgraphsdk.NewGraphServiceClientWithCredentials(cred, graphScopes)
	if err != nil {
		return nil, fmt.Errorf("teams: graph client: %w", err)
	}
	return &GraphSource{client: client, userID: cfg.UserID}, nil
}

// NewGraphSourceFromEnv builds a GraphSource from CHITON_TEAMS_* environment
// variables (TENANT_ID, CLIENT_ID, CLIENT_SECRET, USER_ID).
func NewGraphSourceFromEnv() (*GraphSource, error) {
	return NewGraphSource(GraphConfig{
		TenantID:     os.Getenv("CHITON_TEAMS_TENANT_ID"),
		ClientID:     os.Getenv("CHITON_TEAMS_CLIENT_ID"),
		ClientSecret: os.Getenv("CHITON_TEAMS_CLIENT_SECRET"),
		UserID:       os.Getenv("CHITON_TEAMS_USER_ID"),
	})
}

// ListTranscripts lists the transcripts for an online meeting.
func (g *GraphSource) ListTranscripts(ctx context.Context, meetingID string) ([]TranscriptMeta, error) {
	coll, err := g.client.
		Users().ByUserId(g.userID).
		OnlineMeetings().ByOnlineMeetingId(meetingID).
		Transcripts().
		Get(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("graph list transcripts: %w", err)
	}
	if coll == nil {
		return nil, nil
	}
	var metas []TranscriptMeta
	for _, t := range coll.GetValue() {
		m := TranscriptMeta{MeetingID: meetingID}
		if id := t.GetId(); id != nil {
			m.ID = *id
		}
		if created := t.GetCreatedDateTime(); created != nil {
			m.CreatedAt = created.UTC()
		}
		if m.ID != "" {
			metas = append(metas, m)
		}
	}
	return metas, nil
}

// FetchContent fetches a transcript's content as WebVTT. The Accept header
// requests text/vtt; ToGetRequestInformation only TryAdds its own Accept, so
// ours wins.
func (g *GraphSource) FetchContent(ctx context.Context, meetingID, transcriptID string) ([]byte, error) {
	headers := abstractions.NewRequestHeaders()
	headers.Add("Accept", "text/vtt")
	cfg := &users.ItemOnlineMeetingsItemTranscriptsItemContentRequestBuilderGetRequestConfiguration{
		Headers: headers,
	}
	data, err := g.client.
		Users().ByUserId(g.userID).
		OnlineMeetings().ByOnlineMeetingId(meetingID).
		Transcripts().ByCallTranscriptId(transcriptID).
		Content().
		Get(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("graph fetch transcript content: %w", err)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("graph returned empty transcript content for %s", transcriptID)
	}
	return data, nil
}

// compile-time assertion that GraphSource satisfies the TranscriptSource interface.
var _ TranscriptSource = (*GraphSource)(nil)
