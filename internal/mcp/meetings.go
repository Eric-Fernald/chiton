package mcpsrv

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	mcplib "github.com/mark3labs/mcp-go/mcp"
	mcpsdk "github.com/mark3labs/mcp-go/server"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	urlingest "github.com/Eric-Fernald/chiton/internal/ingest/url"
)

// --- search_meetings: find meetings + their linked decisions ---

func searchMeetingsTool() mcplib.Tool {
	return mcplib.NewTool("search_meetings",
		mcplib.WithDescription("Find ingested meetings by content and return each meeting with its linked decisions and the code symbols they touch. Use this for 'what meetings discussed X?' / 'when did we last talk about the retry policy?'. Participant names are PII and hidden unless include_pii is set."),
		mcplib.WithString("query",
			mcplib.Description("Text to match against meeting title, transcript, and participants (case-insensitive). Empty = all meetings."),
		),
		mcplib.WithString("since",
			mcplib.Description("Only meetings on/after this date (RFC3339 or YYYY-MM-DD)."),
		),
		mcplib.WithInteger("limit",
			mcplib.DefaultNumber(10),
			mcplib.Description("Max meetings to return (default 10)"),
			mcplib.Min(1),
			mcplib.Max(100),
		),
		mcplib.WithBoolean("include_pii",
			mcplib.Description("Include participant names (PII). Default false: only a participant count is returned."),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func searchMeetingsHandler(gs store.GraphStore) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		query := strings.ToLower(strings.TrimSpace(req.GetString("query", "")))
		since, sinceOK := parseSince(req.GetString("since", ""))
		limit := clampInt(req.GetInt("limit", 10), 1, 100)
		includePII := req.GetBool("include_pii", false)

		views, err := buildMeetingViews(ctx, gs)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("load graph failed", err), nil
		}

		var out []meetingView
		for _, v := range views {
			if sinceOK && (v.parsedDate.IsZero() || v.parsedDate.Before(since)) {
				continue
			}
			if query != "" && !v.matches(query, includePII) {
				continue
			}
			out = append(out, v.public(includePII))
			if len(out) >= limit {
				break
			}
		}
		if len(out) == 0 {
			return mcplib.NewToolResultText("no meetings match (have you run 'chiton meeting ingest'?)"), nil
		}
		return jsonResult(out)
	}
}

// --- decisions_about: decisions touching a topic or symbol ---

func decisionsAboutTool() mcplib.Tool {
	return mcplib.NewTool("decisions_about",
		mcplib.WithDescription("Return the decisions made about a topic or code symbol, each with its rationale, the meeting it was decided in, and the code it governs. This is the meeting→code link: answer 'why is this function like this?' with the meeting that decided it. Requires meetings ingested and linked (chiton meeting ingest + the meeting-linkage subagent flow)."),
		mcplib.WithString("topic",
			mcplib.Required(),
			mcplib.Description("A code symbol label (e.g. \"fetchAll\") or a free-text topic (e.g. \"retry policy\")."),
		),
		mcplib.WithString("since",
			mcplib.Description("Only decisions from meetings on/after this date (RFC3339 or YYYY-MM-DD)."),
		),
		mcplib.WithInteger("limit",
			mcplib.DefaultNumber(20),
			mcplib.Description("Max decisions to return (default 20)"),
			mcplib.Min(1),
			mcplib.Max(100),
		),
		mcplib.WithBoolean("include_pii",
			mcplib.Description("Include action-item assignee names (PII). Default false."),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func decisionsAboutHandler(gs store.GraphStore) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		topic, err := req.RequireString("topic")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		topicLower := strings.ToLower(strings.TrimSpace(topic))
		since, sinceOK := parseSince(req.GetString("since", ""))
		limit := clampInt(req.GetInt("limit", 20), 1, 100)
		includePII := req.GetBool("include_pii", false)

		views, err := buildMeetingViews(ctx, gs)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("load graph failed", err), nil
		}

		type decisionHit struct {
			Summary    string           `json:"summary"`
			Rationale  string           `json:"rationale,omitempty"`
			Confidence float32          `json:"confidence"`
			Provenance model.Provenance `json:"provenance"`
			Governs    []string         `json:"governs,omitempty"`
			Meeting    string           `json:"meeting"`
			Date       string           `json:"date,omitempty"`
			Actions    []actionView     `json:"action_items,omitempty"`
		}
		var hits []decisionHit
		for _, v := range views {
			if sinceOK && (v.parsedDate.IsZero() || v.parsedDate.Before(since)) {
				continue
			}
			for _, d := range v.Decisions {
				if !decisionMatchesTopic(d, topicLower) {
					continue
				}
				actions := d.Actions
				if !includePII {
					actions = actionTextsOnly(actions)
				}
				hits = append(hits, decisionHit{
					Summary:    d.Summary,
					Rationale:  d.Rationale,
					Confidence: d.Confidence,
					Provenance: d.Provenance,
					Governs:    d.Governs,
					Meeting:    v.Title,
					Date:       v.Date,
					Actions:    actions,
				})
				if len(hits) >= limit {
					break
				}
			}
			if len(hits) >= limit {
				break
			}
		}
		if len(hits) == 0 {
			return mcplib.NewToolResultText(fmt.Sprintf("no decisions found about %q", topic)), nil
		}
		return jsonResult(hits)
	}
}

// --- add_url: ingest an external source (write) ---

func addURLTool() mcplib.Tool {
	return mcplib.NewTool("add_url",
		mcplib.WithDescription("Fetch a URL (article, paper PDF, or YouTube transcript) or local file and ingest its readable text as a url node with doc_section children, so its concepts are extracted by the normal pending_extractions flow. Idempotent: re-adding a URL replaces it. Requires a writable store (chiton serve --writable)."),
		mcplib.WithString("url",
			mcplib.Required(),
			mcplib.Description("The URL (http/https/file) or local path to ingest."),
		),
		mcplib.WithString("kind",
			mcplib.Description("Optional content hint: paper | article | tweet | video | document."),
		),
		mcplib.WithString("author",
			mcplib.Description("Optional author to record on the node."),
		),
		mcplib.WithString("contributor",
			mcplib.Description("Optional contributor (who added it) to record on the node."),
		),
	)
}

func addURLHandler(ing *urlingest.Ingester) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		if ing == nil {
			return mcplib.NewToolResultText("add_url unavailable — start 'chiton serve --writable' to enable URL ingest."), nil
		}
		rawURL, err := req.RequireString("url")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		rep, err := ing.Add(ctx, rawURL, urlingest.AddOptions{
			Kind:        req.GetString("kind", ""),
			Author:      req.GetString("author", ""),
			Contributor: req.GetString("contributor", ""),
		})
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("add_url failed", err), nil
		}
		return jsonResult(rep)
	}
}

// --- pending_meeting_links / submit_meeting_links: subagent linkage over MCP ---

func pendingMeetingLinksTool() mcplib.Tool {
	return mcplib.NewTool("pending_meeting_links",
		mcplib.WithDescription("Return meeting→code linkage task specs for ingested meetings not yet linked. Each carries the transcript and a menu of candidate code-symbol labels; find the decisions made and the code they govern, then call submit_meeting_links. chiton calls no LLM itself."),
		mcplib.WithInteger("limit",
			mcplib.DefaultNumber(10),
			mcplib.Description("Max task specs to return (default 10)"),
			mcplib.Min(1),
			mcplib.Max(100),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func pendingMeetingLinksHandler(svc *meeting.LinkService) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		if svc == nil {
			return mcplib.NewToolResultText("meeting linkage unavailable — the server was started without a graph store."), nil
		}
		specs, err := svc.Pending(ctx, clampInt(req.GetInt("limit", 10), 1, 100))
		return pendingResult(specs, err, "pending_meeting_links",
			"no pending meeting links — every ingested meeting is already linked.")
	}
}

func submitMeetingLinksTool() mcplib.Tool {
	return mcplib.NewTool("submit_meeting_links",
		mcplib.WithDescription("Ingest the result of one meeting-linkage task (from pending_meeting_links): creates decision/rationale nodes and DECIDED_IN / DISCUSSED_IN / ACTION_FOR / RATIONALE_FOR / MENTIONED edges tagged INFERRED (or AMBIGUOUS below the confidence threshold). Re-submitting supersedes the meeting's prior decisions. Requires a writable store."),
		mcplib.WithString("task_id",
			mcplib.Required(),
			mcplib.Description("The meeting node ID from pending_meeting_links."),
		),
		mcplib.WithArray("decisions",
			mcplib.Description("Decisions made: {summary, rationale, confidence, references:[symbol labels], action_items:[{text, assignee}]}"),
			mcplib.Items(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"summary":    map[string]any{"type": "string"},
					"rationale":  map[string]any{"type": "string"},
					"confidence": map[string]any{"type": "number"},
					"references": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
					"action_items": map[string]any{"type": "array", "items": map[string]any{
						"type":       "object",
						"properties": map[string]any{"text": map[string]any{"type": "string"}, "assignee": map[string]any{"type": "string"}},
					}},
				},
				"required": []string{"summary"},
			}),
		),
		mcplib.WithArray("discussed",
			mcplib.Description("Code symbol labels discussed but not the subject of a decision."),
			mcplib.WithStringItems(),
		),
	)
}

func submitMeetingLinksHandler(svc *meeting.LinkService) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		if svc == nil {
			return mcplib.NewToolResultText("meeting linkage unavailable — the server was started without a graph store."), nil
		}
		return bindAndSubmit(ctx, req, "submit_meeting_links", svc.Submit)
	}
}

// --- meeting view assembly (shared by search_meetings + decisions_about) ---

// actionView is one action item. Assignee is a person name (PII), carried
// separately so it can be dropped when include_pii is false.
type actionView struct {
	Text     string `json:"text"`
	Assignee string `json:"assignee,omitempty"`
}

type decisionView struct {
	Summary    string           `json:"summary"`
	Rationale  string           `json:"rationale,omitempty"`
	Confidence float32          `json:"confidence"`
	Provenance model.Provenance `json:"provenance"`
	Governs    []string         `json:"governs,omitempty"`
	Actions    []actionView     `json:"action_items,omitempty"`
}

type meetingView struct {
	Title            string         `json:"title"`
	SourceURI        string         `json:"source_uri"`
	Date             string         `json:"date,omitempty"`
	ParticipantCount int            `json:"participant_count"`
	Participants     []string       `json:"participants,omitempty"`
	Decisions        []decisionView `json:"decisions,omitempty"`

	// unexported scratch for filtering.
	text       string
	parsedDate time.Time
}

// matches reports whether the meeting matches the query. Without includePII participants aren't
// matched: returning a meeting because a queried name attended would itself disclose that PII.
func (v meetingView) matches(lowerQuery string, includePII bool) bool {
	if strings.Contains(strings.ToLower(v.Title), lowerQuery) {
		return true
	}
	if strings.Contains(v.text, lowerQuery) {
		return true
	}
	if includePII {
		for _, p := range v.Participants {
			if strings.Contains(strings.ToLower(p), lowerQuery) {
				return true
			}
		}
	}
	for _, d := range v.Decisions {
		if strings.Contains(strings.ToLower(d.Summary), lowerQuery) {
			return true
		}
	}
	return false
}

// public returns a copy safe to serialize: no transcript scratch text and, unless includePII, no
// participant names or action-item assignees (both PII) — just a participant count and action text.
func (v meetingView) public(includePII bool) meetingView {
	out := v
	out.text = ""
	out.parsedDate = time.Time{}
	if !includePII {
		out.Participants = nil
		out.Decisions = stripActionAssignees(out.Decisions)
	}
	return out
}

// stripActionAssignees returns decisions with action-item assignee names removed, deep-copying so
// the source views (whose slices public's shallow copy shares) are not mutated.
func stripActionAssignees(decisions []decisionView) []decisionView {
	if len(decisions) == 0 {
		return decisions
	}
	out := make([]decisionView, len(decisions))
	for i, d := range decisions {
		d.Actions = actionTextsOnly(d.Actions)
		out[i] = d
	}
	return out
}

// actionTextsOnly returns a copy of actions without assignees (PII).
func actionTextsOnly(actions []actionView) []actionView {
	if len(actions) == 0 {
		return actions
	}
	out := make([]actionView, len(actions))
	for i, a := range actions {
		out[i] = actionView{Text: a.Text}
	}
	return out
}

func decisionMatchesTopic(d decisionView, topicLower string) bool {
	if strings.Contains(strings.ToLower(d.Summary), topicLower) {
		return true
	}
	if strings.Contains(strings.ToLower(d.Rationale), topicLower) {
		return true
	}
	for _, g := range d.Governs {
		if strings.Contains(strings.ToLower(g), topicLower) {
			return true
		}
	}
	return false
}

// buildMeetingViews assembles one view per meeting node, newest first, with its decisions and the symbols
// they govern (DECIDED_IN edges to non-meeting nodes). Scans all edges but loads only the nodes it needs.
func buildMeetingViews(ctx context.Context, gs store.GraphStore) ([]meetingView, error) {
	nodes, err := store.NodesMatching(ctx, gs, store.NodeFilter{Kinds: []model.NodeKind{model.KindMeeting, model.KindDecision}})
	if err != nil {
		return nil, err
	}
	edges, err := gs.AllEdges(ctx)
	if err != nil {
		return nil, err
	}

	byID := make(map[string]*model.Node, len(nodes)) // nil caches a missing node
	for i := range nodes {
		byID[nodes[i].ID] = &nodes[i]
	}

	// Index decision → (governed symbol labels, meeting id) and meeting → label.
	decGoverns := map[string][]string{}
	decMeeting := map[string]string{}
	for _, e := range edges {
		if e.Kind != model.EdgeDecidedIn {
			continue
		}
		dst, ok := byID[e.Dst]
		if !ok {
			if dst, err = gs.GetNode(ctx, e.Dst); err != nil {
				return nil, err
			}
			byID[e.Dst] = dst
		}
		if dst == nil {
			continue
		}
		if dst.Kind == model.KindMeeting {
			decMeeting[e.Src] = e.Dst
		} else {
			decGoverns[e.Src] = append(decGoverns[e.Src], dst.Label)
		}
	}

	// Group decisions by meeting.
	meetingDecisions := map[string][]decisionView{}
	for _, n := range nodes {
		if n.Kind != model.KindDecision {
			continue
		}
		mID := decMeeting[n.ID]
		if mID == "" {
			continue
		}
		meetingDecisions[mID] = append(meetingDecisions[mID], decisionView{
			Summary:    model.AttrString(n.Attrs, "summary"),
			Rationale:  model.AttrString(n.Attrs, "rationale"),
			Confidence: n.Confidence,
			Provenance: n.Provenance,
			Governs:    dedupeStrings(decGoverns[n.ID]),
			Actions:    actionItems(n),
		})
	}

	var views []meetingView
	for _, n := range nodes {
		if n.Kind != model.KindMeeting {
			continue
		}
		v := meetingView{
			Title:        n.Label,
			SourceURI:    n.SourceURI,
			text:         strings.ToLower(model.AttrString(n.Attrs, "text")),
			Participants: model.AttrStrings(n.Attrs, "participants"),
			Decisions:    meetingDecisions[n.ID],
		}
		v.ParticipantCount = len(v.Participants)
		if d := model.AttrString(n.Attrs, "date"); d != "" {
			v.Date = d
			if t, err := time.Parse(time.RFC3339, d); err == nil {
				v.parsedDate = t
			}
		}
		views = append(views, v)
	}

	sort.SliceStable(views, func(i, j int) bool {
		return views[i].parsedDate.After(views[j].parsedDate)
	})
	return views, nil
}

// parseSince parses an RFC3339 timestamp or a YYYY-MM-DD date.
func parseSince(s string) (time.Time, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, false
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t, true
	}
	if t, err := time.Parse("2006-01-02", s); err == nil {
		return t, true
	}
	return time.Time{}, false
}

// actionItems extracts action items (text + assignee) stored on a decision
// node. Assignee is kept separate so PII gating can drop it at output time.
func actionItems(n model.Node) []actionView {
	raw, ok := n.Attrs["action_items"].([]any)
	if !ok {
		return nil
	}
	var out []actionView
	for _, item := range raw {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		text, _ := m["text"].(string)
		if text == "" {
			continue
		}
		assignee, _ := m["assignee"].(string)
		out = append(out, actionView{Text: text, Assignee: assignee})
	}
	return out
}

func dedupeStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, s := range in {
		if seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	return out
}
