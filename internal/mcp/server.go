// Package mcpsrv builds and returns an MCP server wired to a GraphStore.
package mcpsrv

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"

	mcplib "github.com/mark3labs/mcp-go/mcp"
	mcpsdk "github.com/mark3labs/mcp-go/server"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/config"
	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/extract/subagent"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/retrieve"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	urlingest "github.com/Eric-Fernald/chiton/internal/ingest/url"
	"github.com/Eric-Fernald/chiton/internal/verify"
)

// Deps carries the optional services behind the semantic, analysis, and meeting tools. Every field can be
// nil (Retriever is nil without embeddings), and the matching tools then reply that they are unavailable.
type Deps struct {
	Retriever *retrieve.Engine
	Extractor *subagent.Service
	// LLM backs pending_analyses / submit_analysis (LLM rules).
	LLM *analyze.LLMService
	// MeetingLinker backs pending_meeting_links / submit_meeting_links
	// (meeting→code linkage). nil disables those tools.
	MeetingLinker *meeting.LinkService
	// Verifier backs pending_verifications / submit_verification. nil disables those tools.
	Verifier *verify.Service
	// URLIngester backs the add_url write tool. nil (e.g. a read-only
	// server) makes add_url report that it is unavailable.
	URLIngester *urlingest.Ingester
	// BaselineDir is where diff_findings reads saved baselines from
	// (default: .chiton/baselines).
	BaselineDir string
	// ProjectRoot confines pending_analyses' path argument and is where diff_findings resolves git refs
	// (default "."; serve passes config.ProjectDir()).
	ProjectRoot string
	// IncludePII turns off redaction of people's names and addresses in the graph tools (serve --include-pii).
	IncludePII bool
	// Version is the server version advertised to clients; empty advertises "dev".
	Version string
}

// New creates an MCPServer exposing chiton's tools over gs. overlay (nil before any cluster pass)
// and each Deps service are optional. A tool missing its service replies that it is unavailable.
func New(gs store.GraphStore, overlay *cluster.Results, deps Deps) *mcpsdk.MCPServer {
	return newServer(gs, overlay, deps)
}

// NewUnbuilt creates the server for a workspace that has no graph yet. It lists the same tools as New, and
// every call replies with how to build the graph expected at graphPath, or to reconnect once one is there.
func NewUnbuilt(graphPath, version string) *mcpsdk.MCPServer {
	explain := func(mcpsdk.ToolHandlerFunc) mcpsdk.ToolHandlerFunc {
		return func(context.Context, mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
			if _, err := os.Stat(graphPath); err == nil {
				return mcplib.NewToolResultError(fmt.Sprintf(
					"a graph was built at %s after this chiton server started; reconnect the server (/mcp or /reload-plugins) to load it",
					graphPath)), nil
			}
			return mcplib.NewToolResultError(fmt.Sprintf(
				"chiton has no graph for this workspace yet (expected at %s). Run `chiton build .` in the project root, or the chiton setup skill, then reconnect the server (/mcp or /reload-plugins).",
				graphPath)), nil
		}
	}
	// The handlers never run, so they need no store or services.
	return newServer(nil, nil, Deps{Version: version}, mcpsdk.WithToolHandlerMiddleware(explain))
}

func newServer(gs store.GraphStore, overlay *cluster.Results, deps Deps, opts ...mcpsdk.ServerOption) *mcpsdk.MCPServer {
	version := deps.Version
	if version == "" {
		version = "dev"
	}
	s := mcpsdk.NewMCPServer("chiton", version, opts...)
	r := redactor{includePII: deps.IncludePII}
	s.AddTool(getNodeTool(), getNodeHandler(gs, r))
	s.AddTool(neighborhoodTool(), neighborhoodHandler(gs, r))
	s.AddTool(queryTool(), queryHandler(gs, r))
	s.AddTool(shortestPathTool(), shortestPathHandler(gs, r))
	s.AddTool(packageDepsTool(), packageDepsHandler(gs))
	s.AddTool(godNodesTool(), godNodesHandler(gs, overlay, r))
	s.AddTool(graphSummaryTool(), graphSummaryHandler(gs, overlay, r))
	s.AddTool(listFindingsTool(), listFindingsHandler(gs))
	s.AddTool(explainFindingTool(), explainFindingHandler(gs))
	s.AddTool(semanticSearchTool(), semanticSearchHandler(deps.Retriever, r))
	s.AddTool(pendingExtractionsTool(), pendingExtractionsHandler(deps.Extractor))
	s.AddTool(submitExtractionTool(), submitExtractionHandler(deps.Extractor))
	baselineDir := deps.BaselineDir
	if baselineDir == "" {
		baselineDir = analyze.BaselineDir(config.StateDir())
	}
	s.AddTool(diffFindingsTool(), diffFindingsHandler(gs, baselineDir, cmp.Or(deps.ProjectRoot, ".")))
	s.AddTool(pendingAnalysesTool(), pendingAnalysesHandler(deps.LLM, cmp.Or(deps.ProjectRoot, ".")))
	s.AddTool(submitAnalysisTool(), submitAnalysisHandler(deps.LLM))
	// Meetings & external sources.
	s.AddTool(searchMeetingsTool(), searchMeetingsHandler(gs))
	s.AddTool(decisionsAboutTool(), decisionsAboutHandler(gs))
	s.AddTool(pendingMeetingLinksTool(), pendingMeetingLinksHandler(deps.MeetingLinker))
	s.AddTool(submitMeetingLinksTool(), submitMeetingLinksHandler(deps.MeetingLinker))
	s.AddTool(addURLTool(), addURLHandler(deps.URLIngester))
	s.AddTool(pendingVerificationsTool(), pendingVerificationsHandler(deps.Verifier))
	s.AddTool(submitVerificationTool(), submitVerificationHandler(deps.Verifier))
	return s
}

// --- tool schemas ---

func getNodeTool() mcplib.Tool {
	return mcplib.NewTool("get_node",
		mcplib.WithDescription("Return a single node by label: the symbol or file with that name, its source location and provenance. Exact names win over substrings, and definitions over references. When several symbols share the name, the result says so."),
		mcplib.WithString("label",
			mcplib.Required(),
			mcplib.Description("Name to look up (case-insensitive), e.g. \"New\", \"Server.Run\", or \"<dir>:<Name>\" such as \"internal/analyze:New\" to name the symbol in one package directory"),
		),
		mcplib.WithString("file",
			mcplib.Description("Optional source file hint to disambiguate when multiple nodes match (case-insensitive substring of source_uri, e.g. \"fetchall.go\")"),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func neighborhoodTool() mcplib.Tool {
	return mcplib.NewTool("get_node_neighborhood",
		mcplib.WithDescription("Walk the code graph from a symbol or file. One call answers questions that take many greps: "+
			"who calls a function (direction \"in\", edge_kinds [\"CALLS\"]), everything that reaches it through a chain of calls (the same, with depth 2-5), "+
			"what it calls (direction \"out\"), and what a file imports or contains. Calls are resolved across files and packages, "+
			"so callers in other packages are included, and each edge comes from the parser rather than a text match. "+
			"Use it before grep for any \"who calls / what calls / what depends on\" question. Method calls on values are not tracked; package-level function calls are."),
		mcplib.WithString("label",
			mcplib.Required(),
			mcplib.Description("Name to start from (case-insensitive), e.g. \"New\", \"Server.Run\", or \"<dir>:<Name>\" such as \"internal/analyze:New\" to pick the symbol in one package directory"),
		),
		mcplib.WithString("direction",
			mcplib.DefaultString(dirBoth),
			mcplib.Description("\"in\" follows edges into each node (callers, importers, the containing file), \"out\" follows edges out of it (callees, imports, contents), \"both\" (default) follows either"),
			mcplib.Enum(dirBoth, dirIn, dirOut),
		),
		mcplib.WithString("file",
			mcplib.Description("Optional source file hint to disambiguate when label matches multiple nodes (case-insensitive substring of source_uri, e.g. \"fetchall.go\")"),
		),
		mcplib.WithInteger("depth",
			mcplib.DefaultNumber(2),
			mcplib.Description("Hop depth (default 2, max 5). Depth 1 with direction \"in\" and edge_kinds [\"CALLS\"] lists direct callers; depth N lists every caller within N calls."),
			mcplib.Min(1),
			mcplib.Max(5),
		),
		mcplib.WithArray("edge_kinds",
			mcplib.Description("Follow only these edge types, e.g. [\"CALLS\"] for the call graph or [\"IMPORTS\"]. Empty = all kinds, which mixes in CONTAINS and IMPORTS edges."),
			mcplib.WithStringItems(),
		),
		mcplib.WithString("source_prefix",
			mcplib.Description("Optional filter: only return neighbor nodes whose source_uri contains this substring (e.g. \"Go_Projects\" to scope to one corpus)"),
		),
		mcplib.WithBoolean("exclude_tests",
			mcplib.DefaultBool(false),
			mcplib.Description("Leave out nodes from _test.go files, and don't walk through them; set it when the question is about non-test code"),
		),
		mcplib.WithInteger("max_nodes",
			mcplib.DefaultNumber(100),
			mcplib.Description("Max nodes to return, nearest hops first (default 100, max 500). The result sets truncated when more were reachable."),
			mcplib.Min(1),
			mcplib.Max(500),
		),
		mcplib.WithInteger("max_tokens",
			mcplib.DefaultNumber(4000),
			mcplib.Description("Approximate token budget for the returned nodes and edges (default 4000, max 16000)."),
			mcplib.Min(500),
			mcplib.Max(16000),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func queryTool() mcplib.Tool {
	return mcplib.NewTool("query_graph",
		mcplib.WithDescription("Search for nodes by label substring. Returns matching nodes without edges."),
		mcplib.WithString("query",
			mcplib.Required(),
			mcplib.Description("Substring to match against node labels (case-insensitive)"),
		),
		mcplib.WithInteger("limit",
			mcplib.DefaultNumber(10),
			mcplib.Description("Max results to return (default 10)"),
			mcplib.Min(1),
			mcplib.Max(100),
		),
		mcplib.WithString("source_prefix",
			mcplib.Description("Optional filter: only return nodes whose source_uri contains this substring"),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func shortestPathTool() mcplib.Tool {
	return mcplib.NewTool("shortest_path",
		mcplib.WithDescription("Find the shortest undirected path between two nodes."),
		mcplib.WithString("from",
			mcplib.Required(),
			mcplib.Description("Label of the source node (case-insensitive substring)"),
		),
		mcplib.WithString("to",
			mcplib.Required(),
			mcplib.Description("Label of the destination node (case-insensitive substring)"),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

// --- handlers ---

// clampInt bounds v to [lo, hi]. mcp-go does NOT enforce a schema's Min/Max at dispatch, so an
// out-of-range input (e.g. depth=100000) must be clamped before it drives an unbounded traversal.
func clampInt(v, lo, hi int) int {
	return min(max(v, lo), hi)
}

// lookupNode resolves label (plus an optional file hint) to one node, with a note when other symbols share its
// name. A non-nil result is the tool response to return instead: a search failure, or a no-match message, also
// given for a node whose label names a person.
func lookupNode(ctx context.Context, res *resolver, r redactor, label, file string) (model.Node, string, *mcplib.CallToolResult) {
	node, others, found, err := res.lookup(ctx, label, file)
	if err != nil {
		return model.Node{}, "", mcplib.NewToolResultErrorFromErr("search failed", err)
	}
	if !found || r.hidden(node) {
		return model.Node{}, "", mcplib.NewToolResultText(fmt.Sprintf("no node found matching %q", label))
	}
	note := ""
	if others > 0 {
		note = fmt.Sprintf("%d other symbol(s) are also named %q; this is the one in %s. Pass file, or a label like \"<dir>:%s\", to pick another.",
			others, node.Label, r.sourceURI(node), node.Label)
	}
	return node, note, nil
}

// parseEdgeKinds upper-cases the edge_kinds argument. It returns nil (all kinds) when none are given.
func parseEdgeKinds(req mcplib.CallToolRequest) []model.EdgeKind {
	return model.ParseEdgeKinds(req.GetStringSlice("edge_kinds", nil))
}

func getNodeHandler(gs store.GraphStore, r redactor) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		label, err := req.RequireString("label")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		node, note, errRes := lookupNode(ctx, newResolver(gs), r, label, req.GetString("file", ""))
		if errRes != nil {
			return errRes, nil
		}
		if note == "" {
			return jsonResult(r.node(node))
		}
		return jsonResult(struct {
			leanNode
			Note string `json:"note"`
		}{r.node(node), note})
	}
}

func neighborhoodHandler(gs store.GraphStore, r redactor) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		label, err := req.RequireString("label")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		depth := clampInt(req.GetInt("depth", 2), 1, 5)
		maxNodes := clampInt(req.GetInt("max_nodes", 100), 1, 500)
		maxTokens := clampInt(req.GetInt("max_tokens", 4000), 500, 16000)

		direction := strings.ToLower(req.GetString("direction", dirBoth))
		if direction != dirIn && direction != dirOut {
			direction = dirBoth
		}

		resv := newResolver(gs)
		seed, note, errRes := lookupNode(ctx, resv, r, label, req.GetString("file", ""))
		if errRes != nil {
			return errRes, nil
		}
		walked, walkedEdges, err := resv.walk(ctx, seed, depth, parseEdgeKinds(req), direction, req.GetBool("exclude_tests", false))
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("neighbors failed", err), nil
		}
		seed = walked[0]
		reachable := keepPrefix(seed.ID, walked, req.GetString("source_prefix", ""), r)
		nodes, edges := capNeighborhood(reachable, walkedEdges, maxNodes, maxTokens, r)

		type result struct {
			Seed      leanNode   `json:"seed"`
			Nodes     []leanNode `json:"nodes"`
			Edges     []leanEdge `json:"edges"`
			Truncated bool       `json:"truncated,omitempty"`
			Summary   string     `json:"summary"`
			Note      string     `json:"note,omitempty"`
		}
		where := "within"
		switch direction {
		case dirIn:
			where = "pointing into, within"
		case dirOut:
			where = "pointing out of, within"
		}
		res := result{
			Seed:      r.node(seed),
			Nodes:     nodes,
			Edges:     edges,
			Truncated: len(nodes) < len(reachable),
			Summary:   fmt.Sprintf("%d nodes, %d edges %s depth %d of %q", len(nodes), len(edges), where, depth, r.label(seed)),
			Note:      note,
		}
		if res.Truncated {
			res.Summary += fmt.Sprintf(" (truncated from %d reachable nodes; raise max_nodes/max_tokens or narrow edge_kinds/source_prefix)", len(reachable))
		}
		return jsonResult(res)
	}
}

// keepPrefix keeps the seed and the nodes whose redacted source_uri contains prefix (case-insensitive).
func keepPrefix(seedID string, nodes []model.Node, prefix string, r redactor) []model.Node {
	if prefix == "" {
		return nodes
	}
	lower := strings.ToLower(prefix)
	var out []model.Node
	for _, n := range nodes {
		if n.ID == seedID || strings.Contains(strings.ToLower(r.sourceURI(n)), lower) {
			out = append(out, n)
		}
	}
	return out
}

// capNeighborhood keeps the longest BFS-order prefix of nodes within maxNodes and maxTokens, pricing each node
// as its lean JSON plus its edges back to earlier nodes, and returns that prefix with the edges among it.
func capNeighborhood(nodes []model.Node, edges []model.Edge, maxNodes, maxTokens int, r redactor) ([]leanNode, []leanEdge) {
	pos := make(map[string]int, len(nodes))
	for i, n := range nodes {
		pos[n.ID] = i
	}
	labels := r.labels(nodes)
	lean := r.nodes(nodes)
	cost := make([]int, len(nodes))
	for i := range lean {
		cost[i] = jsonLen(lean[i])
	}
	var within []model.Edge
	for _, e := range edges {
		i, okSrc := pos[e.Src]
		j, okDst := pos[e.Dst]
		if okSrc && okDst {
			within = append(within, e)
			cost[max(i, j)] += jsonLen(toLeanEdge(e, labels))
		}
	}
	idx := make([]int, min(len(nodes), maxNodes))
	for i := range idx {
		idx[i] = i
	}
	kept := len(retrieve.TrimToBudget(idx, maxTokens, func(i int) int { return cost[i] }))
	keptEdges := []leanEdge{}
	for _, e := range within {
		if pos[e.Src] < kept && pos[e.Dst] < kept {
			keptEdges = append(keptEdges, toLeanEdge(e, labels))
		}
	}
	return lean[:kept], keptEdges
}

// jsonLen is v's compact JSON size, the unit token budgets are priced in (~4 bytes per token).
func jsonLen(v any) int {
	b, _ := json.Marshal(v)
	return len(b)
}

func queryHandler(gs store.GraphStore, r redactor) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		query, err := req.RequireString("query")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		limit := clampInt(req.GetInt("limit", 10), 1, 100)
		prefix := req.GetString("source_prefix", "")
		lower := strings.ToLower(prefix)
		nodes, err := searchKept(ctx, gs, query, limit, func(n model.Node) bool {
			return !r.hidden(n) && (prefix == "" || strings.Contains(strings.ToLower(n.SourceURI), lower))
		})
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search failed", err), nil
		}
		lean := r.nodes(nodes)
		switch {
		case len(lean) > 0:
			return jsonResult(lean)
		case prefix != "":
			return mcplib.NewToolResultText(fmt.Sprintf("no nodes found matching %q with source_prefix %q", query, prefix)), nil
		default:
			return mcplib.NewToolResultText(fmt.Sprintf("no nodes found matching %q", query)), nil
		}
	}
}

// searchKept returns up to limit nodes whose label contains query and that keep accepts. It widens the search
// until enough survive, so nodes keep rejects can't use up the limit and hide later matches.
func searchKept(ctx context.Context, gs store.GraphStore, query string, limit int, keep func(model.Node) bool) ([]model.Node, error) {
	for fetch := limit; ; fetch *= 4 {
		nodes, err := gs.SearchNodes(ctx, query, fetch)
		if err != nil {
			return nil, err
		}
		exhausted := len(nodes) < fetch
		kept := slices.DeleteFunc(nodes, func(n model.Node) bool { return !keep(n) })
		if len(kept) >= limit {
			return kept[:limit], nil
		}
		if exhausted {
			return kept, nil
		}
	}
}

func shortestPathHandler(gs store.GraphStore, r redactor) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		fromLabel, err := req.RequireString("from")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		toLabel, err := req.RequireString("to")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}

		// Endpoints resolve as get_node does, so a short name lands on its definition, not the first substring match.
		res := newResolver(gs)
		from, fromNote, errRes := lookupNode(ctx, res, r, fromLabel, "")
		if errRes != nil {
			return errRes, nil
		}
		to, toNote, errRes := lookupNode(ctx, res, r, toLabel, "")
		if errRes != nil {
			return errRes, nil
		}

		path, err := gs.ShortestPath(ctx, from.ID, to.ID)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("shortest path failed", err), nil
		}
		if len(path.Nodes) == 0 {
			return mcplib.NewToolResultText(
				fmt.Sprintf("no path found between %q and %q", r.label(from), r.label(to)),
			), nil
		}
		type leanPath struct {
			Nodes []leanNode `json:"nodes"`
			Edges []leanEdge `json:"edges"`
			Note  string     `json:"note,omitempty"`
		}
		return jsonResult(leanPath{
			Nodes: r.nodes(path.Nodes),
			Edges: toLeanEdges(path.Edges, r.labels(path.Nodes)),
			Note:  strings.TrimSpace(fromNote + " " + toNote),
		})
	}
}

// --- lean output types (strip internal metadata the LLM doesn't need) ---

type leanNode struct {
	Kind       model.NodeKind    `json:"kind"`
	Label      string            `json:"label"`
	SourceURI  string            `json:"source_uri"`
	Span       *model.SourceSpan `json:"span,omitempty"`
	Attrs      map[string]any    `json:"attrs,omitempty"`
	Provenance model.Provenance  `json:"provenance"`
	Confidence float32           `json:"confidence"`
}

type leanEdge struct {
	Kind  model.EdgeKind `json:"kind"`
	From  string         `json:"from"`
	To    string         `json:"to"`
	Attrs map[string]any `json:"attrs,omitempty"`
}

// toLeanEdges projects edges to their lean shape, naming endpoints by label. The result is never nil.
func toLeanEdges(edges []model.Edge, labels map[string]string) []leanEdge {
	out := make([]leanEdge, len(edges))
	for i, e := range edges {
		out[i] = toLeanEdge(e, labels)
	}
	return out
}

func toLeanEdge(e model.Edge, labels map[string]string) leanEdge {
	return leanEdge{
		Kind:  e.Kind,
		From:  labels[e.Src],
		To:    labels[e.Dst],
		Attrs: e.Attrs,
	}
}

func godNodesTool() mcplib.Tool {
	return mcplib.NewTool("god_nodes",
		mcplib.WithDescription("Return the highest-betweenness nodes in the graph (requires a prior 'chiton cluster' run). These are the architectural hubs that most shortest paths pass through."),
		mcplib.WithInteger("limit",
			mcplib.DefaultNumber(10),
			mcplib.Description("Max nodes to return (default 10)"),
			mcplib.Min(1),
			mcplib.Max(100),
		),
		mcplib.WithInteger("community_id",
			mcplib.Description("Optional: restrict results to nodes in this community (from chiton cluster output)"),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func godNodesHandler(gs store.GraphStore, overlay *cluster.Results, r redactor) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		if overlay == nil {
			return mcplib.NewToolResultText("no cluster data available — run 'chiton cluster' then restart chiton serve"), nil
		}

		limit := clampInt(req.GetInt("limit", 10), 1, 100)
		filterComm := req.GetInt("community_id", -1)

		type godNode struct {
			Label       string  `json:"label"`
			Community   int     `json:"community_id"`
			Betweenness float64 `json:"betweenness"`
			Degree      int     `json:"degree"`
		}

		ids := rankedIDs(overlay, func(_ string, nr cluster.NodeResult) bool {
			return filterComm < 0 || nr.CommunityID == filterComm
		})
		if len(ids) > limit {
			ids = ids[:limit]
		}
		stored, err := r.loadLabels(ctx, gs, ids)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("failed to load nodes", err), nil
		}

		out := make([]godNode, len(ids))
		for i, id := range ids {
			nr := overlay.Nodes[id]
			out[i] = godNode{
				Label:       r.overlayLabel(id, nr, stored),
				Community:   nr.CommunityID,
				Betweenness: nr.Betweenness,
				Degree:      nr.Degree,
			}
		}
		return jsonResult(out)
	}
}

func graphSummaryTool() mcplib.Tool {
	return mcplib.NewTool("graph_summary",
		mcplib.WithDescription("Return a token-budgeted overview of the graph: node/edge counts by kind, community summaries, and top god nodes. Use this for a high-level architectural snapshot before drilling into specific neighborhoods."),
		mcplib.WithString("scope",
			mcplib.Description("Optional source URI prefix to restrict the summary to a subdirectory or package (e.g. \"internal/auth\"). Empty = entire graph."),
		),
		mcplib.WithInteger("max_tokens",
			mcplib.DefaultNumber(4000),
			mcplib.Description("Approximate token budget for the response (default 4000). Controls how many communities are included."),
			mcplib.Min(500),
			mcplib.Max(16000),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

// scopedKindCounts counts nodes and edges by kind among the nodes whose redacted source URI contains scope
// (case-insensitive), so a person:// scope can't probe for a name. It also returns the in-scope node IDs.
func scopedKindCounts(ctx context.Context, gs store.GraphStore, r redactor, scope string) (map[model.NodeKind]int, map[model.EdgeKind]int, map[string]bool, error) {
	allNodes, err := gs.AllNodes(ctx)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load nodes: %w", err)
	}
	allEdges, err := gs.AllEdges(ctx)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load edges: %w", err)
	}
	scopeLower := strings.ToLower(scope)
	inScope := make(map[string]bool)
	byNode := make(map[model.NodeKind]int)
	for _, n := range allNodes {
		if strings.Contains(strings.ToLower(r.sourceURI(n)), scopeLower) {
			inScope[n.ID] = true
			byNode[n.Kind]++
		}
	}
	byEdge := make(map[model.EdgeKind]int)
	for _, e := range allEdges {
		if inScope[e.Src] && inScope[e.Dst] {
			byEdge[e.Kind]++
		}
	}
	return byNode, byEdge, inScope, nil
}

// sumCounts totals a by-kind count map.
func sumCounts[K comparable](m map[K]int) int {
	total := 0
	for _, c := range m {
		total += c
	}
	return total
}

func graphSummaryHandler(gs store.GraphStore, overlay *cluster.Results, r redactor) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		scope := req.GetString("scope", "")
		maxTokens := req.GetInt("max_tokens", 4000)

		var (
			byNodeKind map[model.NodeKind]int
			byEdgeKind map[model.EdgeKind]int
			inScope    map[string]bool // built only when scoped, since every node counts when unscoped
			err        error
		)
		if scope == "" {
			byNodeKind, byEdgeKind, err = store.CountByKind(ctx, gs)
		} else {
			byNodeKind, byEdgeKind, inScope, err = scopedKindCounts(ctx, gs, r, scope)
		}
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("graph summary failed", err), nil
		}

		type commSummary struct {
			ID        int      `json:"id"`
			NodeCount int      `json:"node_count"`
			TopNodes  []string `json:"top_nodes"`
			topIDs    []string // TopNodes' node IDs, labelled once the trimmed set is known
		}
		type godNodeSummary struct {
			Label       string  `json:"label"`
			CommunityID int     `json:"community_id"`
			Betweenness float64 `json:"betweenness"`
			Degree      int     `json:"degree"`
		}

		var communities []commSummary
		var godNodes []godNodeSummary

		if overlay != nil {
			// Rank first so each community's top_nodes are its highest-betweenness members.
			ids := rankedIDs(overlay, func(id string, _ cluster.NodeResult) bool { return scope == "" || inScope[id] })
			godIDs := ids[:min(10, len(ids))]

			commIdx := make(map[int]int) // maps a community ID to its index in communities
			for _, id := range ids {
				nr := overlay.Nodes[id]
				i, ok := commIdx[nr.CommunityID]
				if !ok {
					i = len(communities)
					commIdx[nr.CommunityID] = i
					communities = append(communities, commSummary{ID: nr.CommunityID})
				}
				c := &communities[i]
				c.NodeCount++
				if len(c.topIDs) < 5 {
					c.topIDs = append(c.topIDs, id)
				}
			}
			slices.SortFunc(communities, func(a, b commSummary) int {
				return cmp.Or(cmp.Compare(b.NodeCount, a.NodeCount), cmp.Compare(a.ID, b.ID))
			})

			// Trim communities to stay within token budget (~4 chars/token,
			// ~100 chars per community entry, 300 chars fixed overhead).
			charBudget := maxTokens*4 - 300 - len(godIDs)*80
			maxComms := max(charBudget/100, 0)
			if len(communities) > maxComms {
				communities = communities[:maxComms]
			}

			// Label only the nodes shown, from their stored (redacted) nodes.
			shown := slices.Clone(godIDs)
			for _, c := range communities {
				shown = append(shown, c.topIDs...)
			}
			stored, err := r.loadLabels(ctx, gs, shown)
			if err != nil {
				return mcplib.NewToolResultErrorFromErr("failed to load nodes", err), nil
			}
			for i := range communities {
				c := &communities[i]
				for _, id := range c.topIDs {
					c.TopNodes = append(c.TopNodes, r.overlayLabel(id, overlay.Nodes[id], stored))
				}
			}
			for _, id := range godIDs {
				nr := overlay.Nodes[id]
				godNodes = append(godNodes, godNodeSummary{
					Label:       r.overlayLabel(id, nr, stored),
					CommunityID: nr.CommunityID,
					Betweenness: nr.Betweenness,
					Degree:      nr.Degree,
				})
			}
		}

		type summary struct {
			Scope       string                 `json:"scope,omitempty"`
			NodeCount   int                    `json:"node_count"`
			EdgeCount   int                    `json:"edge_count"`
			ByNodeKind  map[model.NodeKind]int `json:"by_node_kind"`
			ByEdgeKind  map[model.EdgeKind]int `json:"by_edge_kind"`
			Communities []commSummary          `json:"communities,omitempty"`
			GodNodes    []godNodeSummary       `json:"god_nodes,omitempty"`
		}
		return jsonResult(summary{
			Scope:       scope,
			NodeCount:   sumCounts(byNodeKind),
			EdgeCount:   sumCounts(byEdgeKind),
			ByNodeKind:  byNodeKind,
			ByEdgeKind:  byEdgeKind,
			Communities: communities,
			GodNodes:    godNodes,
		})
	}
}

// rankedIDs returns the IDs of the overlay results keep accepts, ranked by cluster.CompareByBetweenness (the
// shared god-node order), then by ID.
func rankedIDs(overlay *cluster.Results, keep func(id string, nr cluster.NodeResult) bool) []string {
	var ids []string
	for id, nr := range overlay.Nodes {
		if keep(id, nr) {
			ids = append(ids, id)
		}
	}
	slices.SortFunc(ids, func(a, b string) int {
		return cmp.Or(cluster.CompareByBetweenness(overlay.Nodes[a], overlay.Nodes[b]), cmp.Compare(a, b))
	})
	return ids
}

// pendingResult renders a pending-task listing. It reports "<tool, spaced> failed" on error, the empty
// message when there are no specs, and the specs as JSON otherwise.
func pendingResult[T any](specs []T, err error, tool, empty string) (*mcplib.CallToolResult, error) {
	if err != nil {
		return mcplib.NewToolResultErrorFromErr(strings.ReplaceAll(tool, "_", " ")+" failed", err), nil
	}
	if len(specs) == 0 {
		return mcplib.NewToolResultText(empty), nil
	}
	return jsonResult(specs)
}

// bindAndSubmit binds the tool arguments into R, passes them to submit, and renders the report.
func bindAndSubmit[R, Rep any](ctx context.Context, req mcplib.CallToolRequest, tool string,
	submit func(context.Context, R) (Rep, error)) (*mcplib.CallToolResult, error) {
	var result R
	if err := req.BindArguments(&result); err != nil {
		return mcplib.NewToolResultErrorFromErr("invalid "+tool+" arguments", err), nil
	}
	rep, err := submit(ctx, result)
	if err != nil {
		return mcplib.NewToolResultErrorFromErr(strings.ReplaceAll(tool, "_", " ")+" failed", err), nil
	}
	return jsonResult(rep)
}

// jsonResult renders v as compact JSON. HTML escaping is off, so code like "<-ch" or "a && b" stays readable
// and doesn't grow into `\u003c` / `\u0026` sequences that cost tokens.
func jsonResult(v any) (*mcplib.CallToolResult, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return mcplib.NewToolResultErrorFromErr("marshal failed", err), nil
	}
	return mcplib.NewToolResultText(strings.TrimSuffix(buf.String(), "\n")), nil
}
