// Package email ingests email — from local .eml/.mbox files, Microsoft Outlook
// (via Graph), or Gmail (via its REST API) — into chiton's graph. It is the
// Phase 6 "email ingestion" stretch item.
//
// Each message becomes an `email` node with `doc_section` children carrying the
// body text, so the existing pending_extractions / submit_extraction subagent
// flow extracts concepts from an email for free — no email-specific LLM path is
// needed (the same trick the URL ingester uses). Sender and recipients become
// PII-tagged `person` nodes that share the meeting subsystem's person:// scheme
// and PII convention; they are keyed on the email address, so they dedupe
// participants across email but do NOT automatically merge with a meeting's
// display-name person nodes (cross-source identity resolution would need an
// alias layer — out of scope). Replies link to the message they answer with a
// REFERENCES edge when the parent is in the same ingest batch.
//
// The provider wiring (Outlook over Microsoft Graph, Gmail over REST) lives
// behind the Source interface; the orchestration and graph-building here are
// pure-Go and unit-tested with local files and a fake source.
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

// MessageURI returns a stable source_uri for a message. A Message-ID gives a
// provider-independent identity (so the same mail ingested from a file and from
// Gmail dedupes); without one we fall back to a content hash.
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

// Ingest reads messages from src and upserts each as an email subgraph (email
// node + doc_section body children + PII person nodes), keyed on a stable
// email:// URI so a re-ingest replaces rather than duplicates. Replies link to
// their parent (REFERENCES) when the parent appears in the same batch. Read-only
// stores are rejected up front.
func (i *Ingester) Ingest(ctx context.Context, src Source, opts Query) (Report, error) {
	if ro, ok := i.store.(store.ReadOnlyAware); ok && ro.ReadOnly() {
		return Report{}, fmt.Errorf("email: store is read-only — use the 'chiton email' CLI or restart 'chiton serve --writable'")
	}
	msgs, err := src.Messages(ctx, opts)
	if err != nil {
		return Report{}, fmt.Errorf("list messages: %w", err)
	}

	now := i.now()
	rep := Report{Source: src.Name()}
	idByMessageID := map[string]string{} // Message-ID → email node ID (this batch)
	type reply struct{ childID, inReplyTo string }
	var replies []reply

	for _, m := range msgs {
		m.Source = src.Name()
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
	for idx, chunk := range chunkText(body, maxSectionBytes, maxSections) {
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
			Attrs:      map[string]any{"text": chunk, "index": idx},
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

// idLabel is the canonical label used to derive the email node's stable ID:
// the Message-ID when present (resolvable from a reply's In-Reply-To), else the
// uri itself.
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

// chunkText splits text into chunks of at most size bytes, preferring blank-line
// (paragraph) boundaries, capped at maxChunks (the final chunk absorbs the
// remainder once the cap is hit). An empty body yields no chunks.
func chunkText(text string, size, maxChunks int) []string {
	if strings.TrimSpace(text) == "" {
		return nil
	}
	if size <= 0 {
		size = maxSectionBytes
	}
	if len(text) <= size {
		return []string{text}
	}
	paras := strings.Split(text, "\n\n")
	var chunks []string
	var cur strings.Builder
	atCap := func() bool { return maxChunks > 0 && len(chunks) >= maxChunks-1 }
	for _, p := range paras {
		if cur.Len() > 0 && cur.Len()+len(p)+2 > size && !atCap() {
			chunks = append(chunks, strings.TrimSpace(cur.String()))
			cur.Reset()
		}
		if cur.Len() > 0 {
			cur.WriteString("\n\n")
		}
		cur.WriteString(p)
	}
	if rest := strings.TrimSpace(cur.String()); rest != "" {
		chunks = append(chunks, rest)
	}
	if len(chunks) == 0 {
		chunks = []string{text}
	}
	// Paragraph grouping leaves two cases over the size budget: a single
	// paragraph longer than size (common for HTML bodies, whose paragraph breaks
	// htmlToText strips), and the final chunk once the maxChunks cap stops
	// flushing. Hard-split both so no doc_section overflows a downstream
	// extraction context. size wins over maxChunks here.
	return splitOversized(chunks, size)
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

// runeBoundary returns the largest index <= max that does not split a UTF-8
// rune. It always returns > 0 so callers make progress.
func runeBoundary(s string, max int) int {
	if max >= len(s) {
		return len(s)
	}
	i := max
	for i > 0 && !utf8.RuneStart(s[i]) {
		i--
	}
	if i == 0 {
		return max // pathological (no boundary found); force progress
	}
	return i
}
