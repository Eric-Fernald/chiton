//go:build msgraph

// The Microsoft Graph client. Excluded from the default build — see
// graph_stub.go for why, and for the msgraph-free counterpart.

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

// graphScopes is the app-only .default scope; it needs OnlineMeetingTranscript.Read.All plus an application
// access policy covering the organizer's meetings.
var graphScopes = []string{"https://graph.microsoft.com/.default"}

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
	vals := coll.GetValue()
	metas := make([]TranscriptMeta, 0, len(vals))
	for _, t := range vals {
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

// FetchContent fetches a transcript as WebVTT; our Accept: text/vtt wins because ToGetRequestInformation only
// TryAdds its own Accept header.
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
