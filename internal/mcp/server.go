// Package mcpsrv builds and returns an MCP server wired to a GraphStore.
package mcpsrv

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	mcplib "github.com/mark3labs/mcp-go/mcp"
	mcpsdk "github.com/mark3labs/mcp-go/server"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/extract/subagent"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/retrieve"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// Deps carries the optional semantic- and analysis-layer services the server
// wires into its Phase 2 / 2.5 tools. All are nilable: a graph without
// embeddings has no Retriever (semantic_search degrades gracefully), and any
// may be absent in tests.
type Deps struct {
	Retriever *retrieve.Engine
	Extractor *subagent.Service
	// LLM backs pending_analyses / submit_analysis (Phase 2.5 LLM rules).
	LLM *analyze.LLMService
	// BaselineDir is where diff_findings reads saved baselines from
	// (default: .chiton/baselines).
	BaselineDir string
}

// New creates an MCPServer exposing tools against the given graph store.
// overlay may be nil if no cluster pass has been run yet; god_nodes will
// return a helpful message in that case. deps carries the semantic-layer
// services (see Deps); the corresponding tools report a helpful message when
// their dependency is nil.
func New(gs store.GraphStore, overlay *cluster.Results, deps Deps) *mcpsdk.MCPServer {
	s := mcpsdk.NewMCPServer("chiton", "0.1.0")
	s.AddTool(getNodeTool(), getNodeHandler(gs))
	s.AddTool(neighborhoodTool(), neighborhoodHandler(gs))
	s.AddTool(queryTool(), queryHandler(gs))
	s.AddTool(shortestPathTool(), shortestPathHandler(gs))
	s.AddTool(godNodesTool(), godNodesHandler(overlay))
	s.AddTool(graphSummaryTool(), graphSummaryHandler(gs, overlay))
	s.AddTool(listFindingsTool(), listFindingsHandler(gs))
	s.AddTool(explainFindingTool(), explainFindingHandler(gs))
	s.AddTool(semanticSearchTool(), semanticSearchHandler(deps.Retriever))
	s.AddTool(pendingExtractionsTool(), pendingExtractionsHandler(deps.Extractor))
	s.AddTool(submitExtractionTool(), submitExtractionHandler(deps.Extractor))
	baselineDir := deps.BaselineDir
	if baselineDir == "" {
		baselineDir = analyze.BaselineDir(".chiton")
	}
	s.AddTool(diffFindingsTool(), diffFindingsHandler(gs, baselineDir))
	s.AddTool(pendingAnalysesTool(), pendingAnalysesHandler(deps.LLM))
	s.AddTool(submitAnalysisTool(), submitAnalysisHandler(deps.LLM))
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

func getNodeHandler(gs store.GraphStore) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		label, err := req.RequireString("label")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		file := req.GetString("file", "")
		limit := 1
		if file != "" {
			limit = 20
		}
		nodes, err := gs.SearchNodes(ctx, label, limit)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search failed", err), nil
		}
		if len(nodes) == 0 {
			return mcplib.NewToolResultText(fmt.Sprintf("no node found matching %q", label)), nil
		}
		return jsonResult(toLeanNode(pickByFile(nodes, file)))
	}
}

func neighborhoodHandler(gs store.GraphStore) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		label, err := req.RequireString("label")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		file := req.GetString("file", "")
		prefix := req.GetString("source_prefix", "")
		depth := req.GetInt("depth", 2)
		kindStrs := req.GetStringSlice("edge_kinds", nil)

		limit := 1
		if file != "" {
			limit = 20
		}
		nodes, err := gs.SearchNodes(ctx, label, limit)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search failed", err), nil
		}
		if len(nodes) == 0 {
			return mcplib.NewToolResultText(fmt.Sprintf("no node found matching %q", label)), nil
		}
		seed := pickByFile(nodes, file)

		var kinds []model.EdgeKind
		for _, k := range kindStrs {
			kinds = append(kinds, model.EdgeKind(strings.ToUpper(k)))
		}

		sub, err := gs.Neighbors(ctx, seed.ID, depth, kinds)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("neighbors failed", err), nil
		}

		allNodes := append([]model.Node{seed}, sub.Nodes...)
		labels := nodeLabels(allNodes)
		leanNodes := make([]leanNode, len(sub.Nodes))
		for i, n := range sub.Nodes {
			leanNodes[i] = toLeanNode(n)
		}
		leanEdges := make([]leanEdge, len(sub.Edges))
		for i, e := range sub.Edges {
			leanEdges[i] = toLeanEdge(e, labels)
		}

		leanNodes, leanEdges = filterByPrefix(seed.Label, leanNodes, leanEdges, prefix)

		type result struct {
			Seed    leanNode   `json:"seed"`
			Nodes   []leanNode `json:"nodes"`
			Edges   []leanEdge `json:"edges"`
			Summary string     `json:"summary"`
		}
		return jsonResult(result{
			Seed:  toLeanNode(seed),
			Nodes: leanNodes,
			Edges: leanEdges,
			Summary: fmt.Sprintf("%d nodes, %d edges within depth %d of %q",
				len(leanNodes), len(leanEdges), depth, seed.Label),
		})
	}
}

func queryHandler(gs store.GraphStore) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		query, err := req.RequireString("query")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		limit := req.GetInt("limit", 10)
		prefix := req.GetString("source_prefix", "")
		nodes, err := gs.SearchNodes(ctx, query, limit)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search failed", err), nil
		}
		if len(nodes) == 0 {
			return mcplib.NewToolResultText(fmt.Sprintf("no nodes found matching %q", query)), nil
		}
		lower := strings.ToLower(prefix)
		lean := make([]leanNode, 0, len(nodes))
		for _, n := range nodes {
			if prefix == "" || strings.Contains(strings.ToLower(n.SourceURI), lower) {
				lean = append(lean, toLeanNode(n))
			}
		}
		if len(lean) == 0 {
			return mcplib.NewToolResultText(fmt.Sprintf("no nodes found matching %q with source_prefix %q", query, prefix)), nil
		}
		return jsonResult(lean)
	}
}

func shortestPathHandler(gs store.GraphStore) mcpsdk.ToolHandlerFunc {
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
		if len(fromNodes) == 0 {
			return mcplib.NewToolResultText(fmt.Sprintf("no node found matching 'from' %q", fromLabel)), nil
		}

		toNodes, err := gs.SearchNodes(ctx, toLabel, 1)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search 'to' failed", err), nil
		}
		if len(toNodes) == 0 {
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
		labels := nodeLabels(path.Nodes)
		leanNodes := make([]leanNode, len(path.Nodes))
		for i, n := range path.Nodes {
			leanNodes[i] = toLeanNode(n)
		}
		leanEdges := make([]leanEdge, len(path.Edges))
		for i, e := range path.Edges {
			leanEdges[i] = toLeanEdge(e, labels)
		}
		type leanPath struct {
			Nodes []leanNode `json:"nodes"`
			Edges []leanEdge `json:"edges"`
		}
		return jsonResult(leanPath{Nodes: leanNodes, Edges: leanEdges})
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

func toLeanNode(n model.Node) leanNode {
	return leanNode{
		Kind:       n.Kind,
		Label:      n.Label,
		SourceURI:  n.SourceURI,
		Span:       n.Span,
		Attrs:      n.Attrs,
		Provenance: n.Provenance,
		Confidence: n.Confidence,
	}
}

func nodeLabels(nodes []model.Node) map[string]string {
	m := make(map[string]string, len(nodes))
	for _, n := range nodes {
		m[n.ID] = n.Label
	}
	return m
}

// pickByFile selects the best match for label among nodes, using file as a
// source_uri hint. Among candidates matching the hint, symbol nodes are preferred
// over file nodes — callers specifying a file hint almost always want a symbol
// from that file, not the file node itself. Falls back to nodes[0] if no hint
// is given or no candidate matches.
func pickByFile(nodes []model.Node, file string) model.Node {
	if file == "" {
		return nodes[0]
	}
	lower := strings.ToLower(file)
	var firstMatch *model.Node
	for i := range nodes {
		n := &nodes[i]
		if strings.Contains(strings.ToLower(n.SourceURI), lower) {
			if firstMatch == nil {
				firstMatch = n
			}
			if n.Kind == model.KindSymbol {
				return *n
			}
		}
	}
	if firstMatch != nil {
		return *firstMatch
	}
	return nodes[0]
}

// filterByPrefix removes neighbor nodes whose source_uri does not contain prefix,
// and removes edges where either endpoint is no longer present. The seed node is
// always retained so its edges survive the filter.
func filterByPrefix(seedLabel string, nodes []leanNode, edges []leanEdge, prefix string) ([]leanNode, []leanEdge) {
	if prefix == "" {
		return nodes, edges
	}
	lower := strings.ToLower(prefix)
	kept := map[string]bool{seedLabel: true}
	var filteredNodes []leanNode
	for _, n := range nodes {
		if strings.Contains(strings.ToLower(n.SourceURI), lower) {
			filteredNodes = append(filteredNodes, n)
			kept[n.Label] = true
		}
	}
	var filteredEdges []leanEdge
	for _, e := range edges {
		if kept[e.From] && kept[e.To] {
			filteredEdges = append(filteredEdges, e)
		}
	}
	return filteredNodes, filteredEdges
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

func godNodesHandler(overlay *cluster.Results) mcpsdk.ToolHandlerFunc {
	return func(_ context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		if overlay == nil {
			return mcplib.NewToolResultText("no cluster data available — run 'chiton cluster' then restart chiton serve"), nil
		}

		limit := req.GetInt("limit", 10)
		filterComm := req.GetInt("community_id", -1)

		type godNode struct {
			Label       string  `json:"label"`
			Community   int     `json:"community_id"`
			Betweenness float64 `json:"betweenness"`
			Degree      int     `json:"degree"`
		}

		// overlay.Nodes is keyed by node ID; we need labels too. We only have
		// IDs in the sidecar, so surface what we have and sort by betweenness.
		type ranked struct {
			id string
			nr cluster.NodeResult
		}
		var candidates []ranked
		for id, nr := range overlay.Nodes {
			if filterComm >= 0 && nr.CommunityID != filterComm {
				continue
			}
			candidates = append(candidates, ranked{id, nr})
		}
		// Sort descending by betweenness.
		for i := 1; i < len(candidates); i++ {
			key := candidates[i]
			j := i - 1
			for j >= 0 && candidates[j].nr.Betweenness < key.nr.Betweenness {
				candidates[j+1] = candidates[j]
				j--
			}
			candidates[j+1] = key
		}
		if limit > 0 && len(candidates) > limit {
			candidates = candidates[:limit]
		}

		out := make([]godNode, len(candidates))
		for i, c := range candidates {
			out[i] = godNode{
				Label:       c.nr.Label,
				Community:   c.nr.CommunityID,
				Betweenness: c.nr.Betweenness,
				Degree:      c.nr.Degree,
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

func graphSummaryHandler(gs store.GraphStore, overlay *cluster.Results) mcpsdk.ToolHandlerFunc {
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
		inScope := make(map[string]bool, len(allNodes))
		byNodeKind := make(map[model.NodeKind]int)
		for _, n := range allNodes {
			if scope == "" || strings.Contains(strings.ToLower(n.SourceURI), scopeLower) {
				inScope[n.ID] = true
				byNodeKind[n.Kind]++
			}
		}

		byEdgeKind := make(map[model.EdgeKind]int)
		for _, e := range allEdges {
			if scope == "" || (inScope[e.Src] && inScope[e.Dst]) {
				byEdgeKind[e.Kind]++
			}
		}

		nodeCount := len(inScope)
		edgeCount := 0
		for _, c := range byEdgeKind {
			edgeCount += c
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
			commNodes := make(map[int][]string)
			type ranked struct {
				nr cluster.NodeResult
			}
			var candidates []ranked
			for id, nr := range overlay.Nodes {
				if scope != "" && !inScope[id] {
					continue
				}
				commNodes[nr.CommunityID] = append(commNodes[nr.CommunityID], nr.Label)
				candidates = append(candidates, ranked{nr})
			}

			for commID, labels := range commNodes {
				top := labels
				if len(top) > 5 {
					top = top[:5]
				}
				communities = append(communities, commSummary{
					ID:        commID,
					NodeCount: len(labels),
					TopNodes:  top,
				})
			}
			sort.Slice(communities, func(i, j int) bool {
				return communities[i].NodeCount > communities[j].NodeCount
			})

			sort.Slice(candidates, func(i, j int) bool {
				return candidates[i].nr.Betweenness > candidates[j].nr.Betweenness
			})
			limit := 10
			if len(candidates) < limit {
				limit = len(candidates)
			}
			for _, c := range candidates[:limit] {
				godNodes = append(godNodes, godNodeSummary{
					Label:       c.nr.Label,
					CommunityID: c.nr.CommunityID,
					Betweenness: c.nr.Betweenness,
					Degree:      c.nr.Degree,
				})
			}

			// Trim communities to stay within token budget (~4 chars/token,
			// ~100 chars per community entry, 300 chars fixed overhead).
			charBudget := maxTokens*4 - 300 - len(godNodes)*80
			maxComms := charBudget / 100
			if maxComms < 0 {
				maxComms = 0
			}
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

func jsonResult(v any) (*mcplib.CallToolResult, error) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return mcplib.NewToolResultErrorFromErr("marshal failed", err), nil
	}
	return mcplib.NewToolResultText(string(b)), nil
}
