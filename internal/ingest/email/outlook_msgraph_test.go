//go:build msgraph

package email

import (
	"testing"

	"github.com/microsoftgraph/msgraph-sdk-go/models"
)

// Behind the msgraph tag because outlookToMessage takes a Graph type; in email_test.go it would drag the
// Graph SDK into the default test binary.
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
