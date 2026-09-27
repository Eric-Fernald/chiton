// Package email ingests .eml/.mbox files, Outlook and Gmail as email nodes with doc_section body children.
// Participants become PII person:// nodes keyed on address; they do not merge with meeting speaker nodes.
package email

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/ingest/chunk"
)

// Section sizing for the doc_section children that carry an email's body, so
// each downstream concept-extraction task stays within a sane context budget.
const (
	maxSectionBytes = 8000
	maxSections     = 40
)

// Address is one mailbox: a display name and/or an email address.
type Address struct {
	Name  string `json:"name,omitempty"`
	Email string `json:"email,omitempty"`
}

// Message is a provider-agnostic email, the unit every Source yields and the
// Ingester turns into graph nodes.
type Message struct {
	ID        string    // provider message id (informational; may be empty)
	MessageID string    // RFC 5322 Message-ID header — stable cross-provider identity
	InReplyTo string    // In-Reply-To header (a Message-ID), for thread linkage
	ThreadID  string    // provider thread/conversation id (optional)
	From      Address   //
	To        []Address //
	Cc        []Address //
	Subject   string    //
	Date      time.Time //
	Body      string    // plain-text body (HTML already stripped)
	Source    string    // "local" | "outlook" | "gmail"
}

// Query bounds and filters a Source's results.
type Query struct {
	Since   time.Time // only messages at/after this time
	Max     int       // cap on messages returned; 0 = source default
	Mailbox string    // folder / label / search query, interpreted per-source
}

// Source lists messages from a backend. Implementations: LocalSource (files),
// OutlookSource (Microsoft Graph), GmailSource (Gmail REST). Tests use a fake.
type Source interface {
	// Messages returns the messages matching opts.
	Messages(ctx context.Context, opts Query) ([]Message, error)
	// Name identifies the source kind ("local", "outlook", "gmail").
	Name() string
}

// OutlookConfig configures the Microsoft Graph mail source (app-only auth); shared by both msgraph builds.
type OutlookConfig struct {
	TenantID     string
	ClientID     string
	ClientSecret string
	// UserID is the mailbox owner's id or UPN: messages are read under
	// /users/{UserID}/messages (app-only has no /me).
	UserID string
}

// Ingester turns messages from a Source into graph nodes.
type Ingester struct {
	store store.GraphStore
	now   func() time.Time
}

// NewIngester builds an Ingester over the given graph store.
func NewIngester(s store.GraphStore) *Ingester {
	return &Ingester{store: s, now: time.Now}
}

// Report summarises an ingest run.
type Report struct {
	Source        string `json:"source"`
	Messages      int    `json:"messages"`
	NodesUpserted int    `json:"nodes_upserted"`
	Persons       int    `json:"persons"`
	References    int    `json:"references"`
}

// MessageURI returns a stable source_uri: the Message-ID when present (so the same mail from a file and from
// Gmail dedupes), else a content hash.
func MessageURI(m Message) string {
	if id := sanitizeMessageID(m.MessageID); id != "" {
		return "email://message/" + id
	}
	src := m.Source
	if src == "" {
		src = "local"
	}
	sum := sha256.Sum256([]byte(src + "|" + m.From.Email + "|" + m.Subject + "|" + m.Date.UTC().Format(time.RFC3339)))
	return "email://" + src + "/" + hex.EncodeToString(sum[:8])
}

// Ingest upserts each message from src as an email subgraph keyed on its email:// URI, so re-ingest replaces
// rather than duplicates; replies get a REFERENCES edge only when the parent is in the same batch.
func (i *Ingester) Ingest(ctx context.Context, src Source, opts Query) (Report, error) {
	if store.IsReadOnly(i.store) {
		return Report{}, fmt.Errorf("email: store is read-only — use the 'chiton email' CLI or restart 'chiton serve --writable'")
	}
	msgs, err := src.Messages(ctx, opts)
	if err != nil {
		return Report{}, fmt.Errorf("list messages: %w", err)
	}

	now := i.now()
	srcName := src.Name()
	rep := Report{Source: srcName}
	idByMessageID := map[string]string{} // Message-ID → email node ID (this batch)
	type reply struct{ childID, inReplyTo string }
	var replies []reply

	for _, m := range msgs {
		m.Source = srcName
		uri := MessageURI(m)
		// Replace any prior version of this message (email + its doc_sections
		// share uri; person nodes have their own person:// uri and survive).
		if _, err := i.store.DeleteBySourceURI(ctx, uri); err != nil {
			return rep, fmt.Errorf("replace prior %s: %w", uri, err)
		}
		nodes, edges, emailID, persons := buildGraph(m, uri, now)
		if err := i.store.UpsertNodes(ctx, nodes); err != nil {
			return rep, fmt.Errorf("upsert email nodes: %w", err)
		}
		if len(edges) > 0 {
			if err := i.store.UpsertEdges(ctx, edges); err != nil {
				return rep, fmt.Errorf("upsert email edges: %w", err)
			}
		}
		rep.Messages++
		rep.NodesUpserted += len(nodes)
		rep.Persons += persons
		// Key on the sanitized (bracket-stripped) Message-ID so a reply's
		// In-Reply-To (also sanitized) resolves to the parent.
		if mid := sanitizeMessageID(m.MessageID); mid != "" {
			idByMessageID[mid] = emailID
		}
		if irt := firstMessageID(m.InReplyTo); irt != "" {
			replies = append(replies, reply{childID: emailID, inReplyTo: irt})
		}
	}

	// Second pass: link replies to parents present in this batch.
	var refEdges []model.Edge
	for _, r := range replies {
		if parentID, ok := idByMessageID[r.inReplyTo]; ok && parentID != r.childID {
			refEdges = append(refEdges, model.Edge{
				ID:         model.NewEdgeID(r.childID, parentID, model.EdgeReferences),
				Src:        r.childID,
				Dst:        parentID,
				Kind:       model.EdgeReferences,
				Provenance: model.ProvExtracted,
				Confidence: 1.0,
				CreatedAt:  now,
			})
		}
	}
	if len(refEdges) > 0 {
		if err := i.store.UpsertEdges(ctx, refEdges); err != nil {
			return rep, fmt.Errorf("upsert reply edges: %w", err)
		}
		rep.References = len(refEdges)
	}
	return rep, nil
}

// buildGraph converts one message into nodes + edges, returning the email node
// ID and the number of person nodes created.
func buildGraph(m Message, uri string, now time.Time) ([]model.Node, []model.Edge, string, int) {
	emailID := model.NewNodeID(uri, idLabel(m, uri))
	subject := strings.TrimSpace(m.Subject)
	if subject == "" {
		subject = "(no subject)"
	}

	attrs := map[string]any{
		"source":  m.Source,
		"subject": subject,
		"from":    m.From.Email,
		"to":      addrEmails(m.To),
	}
	if m.From.Name != "" {
		attrs["from_name"] = m.From.Name
	}
	if len(m.Cc) > 0 {
		attrs["cc"] = addrEmails(m.Cc)
	}
	if m.MessageID != "" {
		attrs["message_id"] = m.MessageID
	}
	if m.InReplyTo != "" {
		attrs["in_reply_to"] = m.InReplyTo
	}
	if m.ThreadID != "" {
		attrs["thread_id"] = m.ThreadID
	}
	if !m.Date.IsZero() {
		attrs["date"] = m.Date.UTC().Format(time.RFC3339)
	}
	body := strings.TrimSpace(m.Body)
	attrs["text_bytes"] = len(body)

	nodes := []model.Node{{
		ID:         emailID,
		Kind:       model.KindEmail,
		Label:      subject,
		SourceURI:  uri,
		Attrs:      attrs,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		CreatedAt:  now,
		UpdatedAt:  now,
	}}
	var edges []model.Edge

	// Body → doc_section children (so concept extraction picks them up for free).
	for idx, sec := range chunkText(body, maxSectionBytes, maxSections) {
		secLabel := subject
		if idx > 0 {
			secLabel = fmt.Sprintf("%s §%d", subject, idx+1)
		}
		secID := model.NewNodeID(uri, fmt.Sprintf("section-%d", idx))
		nodes = append(nodes, model.Node{
			ID:         secID,
			Kind:       model.KindDocSection,
			Label:      secLabel,
			SourceURI:  uri,
			Attrs:      map[string]any{"text": sec, "index": idx},
			Provenance: model.ProvExtracted,
			Confidence: 1.0,
			CreatedAt:  now,
			UpdatedAt:  now,
		})
		edges = append(edges, structEdge(emailID, secID, model.EdgeContains, now))
	}

	// Participants → PII person nodes (sender + recipients), deduped, sharing the
	// meeting subsystem's person:// identity.
	seen := map[string]bool{}
	persons := 0
	addPerson := func(a Address) {
		identity := strings.ToLower(strings.TrimSpace(a.Email))
		if identity == "" {
			identity = strings.TrimSpace(a.Name)
		}
		if identity == "" {
			return
		}
		pid := meeting.PersonID(identity)
		if seen[pid] {
			return
		}
		seen[pid] = true
		persons++
		label := a.Name
		if label == "" {
			label = a.Email
		}
		pAttrs := map[string]any{"pii": true}
		if a.Email != "" {
			pAttrs["email"] = a.Email
		}
		nodes = append(nodes, model.Node{
			ID:         pid,
			Kind:       model.KindPerson,
			Label:      label,
			SourceURI:  meeting.PersonURI(identity),
			Attrs:      pAttrs,
			Provenance: model.ProvExtracted,
			Confidence: 1.0,
			CreatedAt:  now,
			UpdatedAt:  now,
		})
		edges = append(edges, structEdge(emailID, pid, model.EdgeMentioned, now))
	}
	addPerson(m.From)
	for _, a := range m.To {
		addPerson(a)
	}
	for _, a := range m.Cc {
		addPerson(a)
	}

	return nodes, edges, emailID, persons
}

// idLabel derives the email node ID from the Message-ID when present (resolvable from a reply's In-Reply-To),
// else from the uri.
func idLabel(m Message, uri string) string {
	if mid := strings.TrimSpace(m.MessageID); mid != "" {
		return mid
	}
	return uri
}

func structEdge(src, dst string, kind model.EdgeKind, now time.Time) model.Edge {
	return model.Edge{
		ID:         model.NewEdgeID(src, dst, kind),
		Src:        src,
		Dst:        dst,
		Kind:       kind,
		Provenance: model.ProvExtracted,
		Confidence: 1.0,
		CreatedAt:  now,
	}
}

func addrEmails(as []Address) []string {
	out := make([]string, 0, len(as))
	for _, a := range as {
		if a.Email != "" {
			out = append(out, a.Email)
		}
	}
	return out
}

// sanitizeMessageID trims the angle brackets and whitespace from a Message-ID,
// yielding the bare token used in URIs (e.g. "<abc@host>" → "abc@host").
func sanitizeMessageID(id string) string {
	id = strings.TrimSpace(id)
	id = strings.TrimPrefix(id, "<")
	id = strings.TrimSuffix(id, ">")
	return strings.TrimSpace(id)
}

// firstMessageID returns the first bare Message-ID token in an In-Reply-To
// header (which may, rarely, list several).
func firstMessageID(header string) string {
	header = strings.TrimSpace(header)
	if header == "" {
		return ""
	}
	if i := strings.IndexByte(header, '>'); i >= 0 {
		header = header[:i+1]
	} else if f := strings.Fields(header); len(f) > 0 {
		header = f[0]
	}
	return sanitizeMessageID(header)
}

// chunkText splits text into chunks of at most size bytes on paragraph boundaries. Past maxChunks the tail is
// still hard-split, so size wins over maxChunks. Empty text yields no chunks.
func chunkText(text string, size, maxChunks int) []string {
	if strings.TrimSpace(text) == "" {
		return nil
	}
	if size <= 0 {
		size = maxSectionBytes
	}
	// Hard-split oversized paragraphs (HTML bodies lose their breaks) and the post-cap tail so no doc_section
	// overflows a downstream extraction context.
	return splitOversized(chunk.Paragraphs(text, size, maxChunks), size)
}

// splitOversized splits any chunk longer than size into size-bounded pieces on
// UTF-8 rune boundaries, leaving within-budget chunks untouched.
func splitOversized(chunks []string, size int) []string {
	out := make([]string, 0, len(chunks))
	for _, c := range chunks {
		for len(c) > size {
			cut := runeBoundary(c, size)
			out = append(out, c[:cut])
			c = c[cut:]
		}
		if c != "" {
			out = append(out, c)
		}
	}
	return out
}

// runeBoundary returns the largest index <= limit that does not split a UTF-8
// rune. It always returns > 0 so callers make progress.
func runeBoundary(s string, limit int) int {
	if limit >= len(s) {
		return len(s)
	}
	i := limit
	for i > 0 && !utf8.RuneStart(s[i]) {
		i--
	}
	if i == 0 {
		return limit // pathological (no boundary found); force progress
	}
	return i
}
