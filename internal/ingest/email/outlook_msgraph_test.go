//go:build msgraph

package email

import (
	"testing"

	"github.com/microsoftgraph/msgraph-sdk-go/models"
)

// This test lives behind the msgraph tag because its subject, outlookToMessage,
// takes a Microsoft Graph type. Keeping it in email_test.go would drag the whole
// Graph SDK into the default test binary even though the default product binary
// does not link it.
func TestOutlookInReplyToFromHeaders(t *testing.T) {
	gm := models.NewMessage()
	mid := "<reply@x>"
	gm.SetInternetMessageId(&mid)
	h := models.NewInternetMessageHeader()
	hn, hv := "In-Reply-To", "<orig@x>"
	h.SetName(&hn)
	h.SetValue(&hv)
	gm.SetInternetMessageHeaders([]models.InternetMessageHeaderable{h})
	subj := "Re: orig"
	gm.SetSubject(&subj)

	m := outlookToMessage(gm)
	if m.Source != "outlook" {
		t.Errorf("source=%q want outlook", m.Source)
	}
	if m.MessageID != "<reply@x>" {
		t.Errorf("MessageID=%q want <reply@x>", m.MessageID)
	}
	if m.InReplyTo != "<orig@x>" {
		t.Errorf("InReplyTo=%q want <orig@x> (must be pulled from internetMessageHeaders)", m.InReplyTo)
	}
}
