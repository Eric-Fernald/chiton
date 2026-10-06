// Package mcpsrv builds and returns an MCP server wired to a GraphStore.
package mcpsrv

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
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
)

// Deps carries the optional services behind the semantic, analysis, and meeting tools. Every field can be
// nil (Retriever is nil without embeddings), and the matching tools then reply that they are unavailable.
type Deps struct {
	Retriever *retrieve.Engine
	Extractor *subagent.Service
	// LLM backs pending_analyses / submit_analysis (Phase 2.5 LLM rules).
	LLM *analyze.LLMService
	// MeetingLinker backs pending_meeting_links / submit_meeting_links (Phase 4
	// meeting→code linkage). nil disables those tools.
	MeetingLinker *meeting.LinkService
	// URLIngester backs the add_url write tool (Phase 4). nil (e.g. a read-only
	// server) makes add_url report that it is unavailable.
	URLIngester *urlingest.Ingester
	// BaselineDir is where diff_findings reads saved baselines from
	// (default: .chiton/baselines).
	BaselineDir string
	// ProjectRoot confines pending_analyses' path argument (default "."; serve passes config.ProjectDir()).
	ProjectRoot string
	// IncludePII turns off redaction of people's names and addresses in the graph tools (serve --include-pii).
	IncludePII bool
}

// New creates an MCPServer exposing chiton's tools over gs. overlay (nil before any cluster pass)
// and each Deps service are optional. A tool missing its service replies that it is unavailable.
func New(gs store.GraphStore, overlay *cluster.Results, deps Deps) *mcpsdk.MCPServer {
	s := mcpsdk.NewMCPServer("chiton", "0.1.0")
	r := redactor{includePII: deps.IncludePII}
	s.AddTool(getNodeTool(), getNodeHandler(gs, r))
	s.AddTool(neighborhoodTool(), neighborhoodHandler(gs, r))
	s.AddTool(queryTool(), queryHandler(gs, r))
	s.AddTool(shortestPathTool(), shortestPathHandler(gs, r))
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
	s.AddTool(diffFindingsTool(), diffFindingsHandler(gs, baselineDir))
	s.AddTool(pendingAnalysesTool(), pendingAnalysesHandler(deps.LLM, cmp.Or(deps.ProjectRoot, ".")))
	s.AddTool(submitAnalysisTool(), submitAnalysisHandler(deps.LLM))
	// Phase 4: meetings & external sources.
	s.AddTool(searchMeetingsTool(), searchMeetingsHandler(gs))
	s.AddTool(decisionsAboutTool(), decisionsAboutHandler(gs))
	s.AddTool(pendingMeetingLinksTool(), pendingMeetingLinksHandler(deps.MeetingLinker))
	s.AddTool(submitMeetingLinksTool(), submitMeetingLinksHandler(deps.MeetingLinker))
	s.AddTool(addURLTool(), addURLHandler(deps.URLIngester))
	return s
}

// --- tool schemas ---

func getNodeTool() mcplib.Tool {
	return mcplib.NewTool("get_node",
		mcplib.WithDescription("Return a single node by label exact or substring match."),
		mcplib.WithString("label",
			mcplib.Required(),
			mcplib.Description("Node label or substring to look up (case-insensitive)"),
		),
		mcplib.WithString("file",
			mcplib.Description("Optional source file hint to disambiguate when multiple nodes match (case-insensitive substring of source_uri, e.g. \"fetchall.go\")"),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func neighborhoodTool() mcplib.Tool {
	return mcplib.NewTool("get_node_neighborhood",
		mcplib.WithDescription("Return a node and all nodes/edges reachable within depth hops. Prefer this over grep or file-read for architecture questions."),
		mcplib.WithString("label",
			mcplib.Required(),
			mcplib.Description("Node label or substring (case-insensitive)"),
		),
		mcplib.WithString("file",
			mcplib.Description("Optional source file hint to disambiguate when label matches multiple nodes (case-insensitive substring of source_uri, e.g. \"fetchall.go\")"),
		),
		mcplib.WithInteger("depth",
			mcplib.DefaultNumber(2),
			mcplib.Description("Hop depth (default 2, max 5)"),
			mcplib.Min(1),
			mcplib.Max(5),
		),
		mcplib.WithArray("edge_kinds",
			mcplib.Description("Filter to these edge types e.g. [\"CALLS\",\"IMPORTS\"]. Empty = all kinds."),
			mcplib.WithStringItems(),
		),
		mcplib.WithString("source_prefix",
			mcplib.Description("Optional filter: only return neighbor nodes whose source_uri contains this substring (e.g. \"Go_Projects\" to scope to one corpus)"),
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

// lookupNode resolves label (plus an optional file hint) to one node. A non-nil result is the tool response
// to return instead: a search failure, or a no-match message, also given for a node whose label names a person.
func lookupNode(ctx context.Context, gs store.GraphStore, r redactor, label, file string) (model.Node, *mcplib.CallToolResult) {
	node, found, err := store.FindNode(ctx, gs, label, file)
	if err != nil {
		return model.Node{}, mcplib.NewToolResultErrorFromErr("search failed", err)
	}
	if !found || r.hidden(node) {
		return model.Node{}, mcplib.NewToolResultText(fmt.Sprintf("no node found matching %q", label))
	}
	return node, nil
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
		node, errRes := lookupNode(ctx, gs, r, label, req.GetString("file", ""))
		if errRes != nil {
			return errRes, nil
		}
		return jsonResult(r.node(node))
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

		seed, errRes := lookupNode(ctx, gs, r, label, req.GetString("file", ""))
		if errRes != nil {
			return errRes, nil
		}
		sub, err := gs.Neighbors(ctx, seed.ID, depth, parseEdgeKinds(req))
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("neighbors failed", err), nil
		}
		reachable := keepPrefix(seed.ID, sub.Nodes, req.GetString("source_prefix", ""), r)
		nodes, edges := capNeighborhood(reachable, sub.Edges, maxNodes, maxTokens, r)

		type result struct {
			Seed      leanNode   `json:"seed"`
			Nodes     []leanNode `json:"nodes"`
			Edges     []leanEdge `json:"edges"`
			Truncated bool       `json:"truncated,omitempty"`
			Summary   string     `json:"summary"`
		}
		res := result{
			Seed:      r.node(seed),
			Nodes:     nodes,
			Edges:     edges,
			Truncated: len(nodes) < len(reachable),
			Summary:   fmt.Sprintf("%d nodes, %d edges within depth %d of %q", len(nodes), len(edges), depth, r.label(seed)),
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
		nodes, err := gs.SearchNodes(ctx, query, limit)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search failed", err), nil
		}
		lower := strings.ToLower(prefix)
		lean := make([]leanNode, 0, len(nodes))
		for _, n := range nodes {
			if !r.hidden(n) && (prefix == "" || strings.Contains(strings.ToLower(n.SourceURI), lower)) {
				lean = append(lean, r.node(n))
			}
		}
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

		fromNodes, err := gs.SearchNodes(ctx, fromLabel, 1)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search 'from' failed", err), nil
		}
		if len(fromNodes) == 0 || r.hidden(fromNodes[0]) {
			return mcplib.NewToolResultText(fmt.Sprintf("no node found matching 'from' %q", fromLabel)), nil
		}

		toNodes, err := gs.SearchNodes(ctx, toLabel, 1)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search 'to' failed", err), nil
		}
		if len(toNodes) == 0 || r.hidden(toNodes[0]) {
			return mcplib.NewToolResultText(fmt.Sprintf("no node found matching 'to' %q", toLabel)), nil
		}

		path, err := gs.ShortestPath(ctx, fromNodes[0].ID, toNodes[0].ID)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("shortest path failed", err), nil
		}
		if len(path.Nodes) == 0 {
			return mcplib.NewToolResultText(
				fmt.Sprintf("no path found between %q and %q", fromNodes[0].Label, toNodes[0].Label),
			), nil
		}
		type leanPath struct {
			Nodes []leanNode `json:"nodes"`
			Edges []leanEdge `json:"edges"`
		}
		return jsonResult(leanPath{
			Nodes: r.nodes(path.Nodes),
			Edges: toLeanEdges(path.Edges, r.labels(path.Nodes)),
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

func graphSummaryHandler(gs store.GraphStore, overlay *cluster.Results, r redactor) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		scope := req.GetString("scope", "")
		maxTokens := req.GetInt("max_tokens", 4000)

		allNodes, err := gs.AllNodes(ctx)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("failed to load nodes", err), nil
		}
		allEdges, err := gs.AllEdges(ctx)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("failed to load edges", err), nil
		}

		scopeLower := strings.ToLower(scope)
		var inScope map[string]bool // built only when scoped, since every node counts when unscoped
		if scope != "" {
			inScope = make(map[string]bool)
		}
		nodeCount := 0
		byNodeKind := make(map[model.NodeKind]int)
		for _, n := range allNodes {
			if scope != "" {
				// Match the redacted URI, so a person:// scope can't probe for a name.
				if !strings.Contains(strings.ToLower(r.sourceURI(n)), scopeLower) {
					continue
				}
				inScope[n.ID] = true
			}
			nodeCount++
			byNodeKind[n.Kind]++
		}

		edgeCount := 0
		byEdgeKind := make(map[model.EdgeKind]int)
		for _, e := range allEdges {
			if scope == "" || (inScope[e.Src] && inScope[e.Dst]) {
				byEdgeKind[e.Kind]++
				edgeCount++
			}
		}

		type commSummary struct {
			ID        int      `json:"id"`
			NodeCount int      `json:"node_count"`
			TopNodes  []string `json:"top_nodes"`
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
			var stored map[string]string
			if !r.includePII {
				stored = r.labels(allNodes)
			}

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
				if len(c.TopNodes) < 5 {
					c.TopNodes = append(c.TopNodes, r.overlayLabel(id, nr, stored))
				}
			}
			slices.SortFunc(communities, func(a, b commSummary) int {
				return cmp.Or(cmp.Compare(b.NodeCount, a.NodeCount), cmp.Compare(a.ID, b.ID))
			})

			for _, id := range ids[:min(10, len(ids))] {
				nr := overlay.Nodes[id]
				godNodes = append(godNodes, godNodeSummary{
					Label:       r.overlayLabel(id, nr, stored),
					CommunityID: nr.CommunityID,
					Betweenness: nr.Betweenness,
					Degree:      nr.Degree,
				})
			}

			// Trim communities to stay within token budget (~4 chars/token,
			// ~100 chars per community entry, 300 chars fixed overhead).
			charBudget := maxTokens*4 - 300 - len(godNodes)*80
			maxComms := max(charBudget/100, 0)
			if len(communities) > maxComms {
				communities = communities[:maxComms]
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
			NodeCount:   nodeCount,
			EdgeCount:   edgeCount,
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
// and doesn't grow into < / & sequences that cost tokens.
func jsonResult(v any) (*mcplib.CallToolResult, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return mcplib.NewToolResultErrorFromErr("marshal failed", err), nil
	}
	return mcplib.NewToolResultText(strings.TrimSuffix(buf.String(), "\n")), nil
}
