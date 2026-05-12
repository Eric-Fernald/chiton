// Package mcpsrv builds and returns an MCP server wired to a GraphStore.
package mcpsrv

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	mcplib "github.com/mark3labs/mcp-go/mcp"
	mcpsdk "github.com/mark3labs/mcp-go/server"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// New creates an MCPServer exposing tools against the given graph store.
func New(gs store.GraphStore) *mcpsdk.MCPServer {
	s := mcpsdk.NewMCPServer("chiton", "0.1.0")
	s.AddTool(getNodeTool(), getNodeHandler(gs))
	s.AddTool(neighborhoodTool(), neighborhoodHandler(gs))
	s.AddTool(queryTool(), queryHandler(gs))
	s.AddTool(shortestPathTool(), shortestPathHandler(gs))
	s.AddTool(godNodesTool(), godNodesHandler(gs))
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

func godNodesHandler(gs store.GraphStore) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		limit := req.GetInt("limit", 10)

		var commID *int
		if v := req.GetInt("community_id", -1); v >= 0 {
			commID = &v
		}

		nodes, err := gs.GodNodes(ctx, commID, limit)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("god_nodes failed", err), nil
		}
		if len(nodes) == 0 {
			return mcplib.NewToolResultText("no god nodes found — run 'chiton cluster' first to compute betweenness centrality"), nil
		}

		type godNode struct {
			Label       string   `json:"label"`
			Kind        string   `json:"kind"`
			SourceURI   string   `json:"source_uri"`
			Community   *int     `json:"community_id,omitempty"`
			Betweenness float64  `json:"betweenness"`
			Degree      int      `json:"degree"`
		}
		out := make([]godNode, len(nodes))
		for i, n := range nodes {
			b, _ := n.Attrs["betweenness"].(float64)
			d, _ := n.Attrs["degree"].(int)
			out[i] = godNode{
				Label:       n.Label,
				Kind:        string(n.Kind),
				SourceURI:   n.SourceURI,
				Community:   n.CommunityID,
				Betweenness: b,
				Degree:      d,
			}
		}
		return jsonResult(out)
	}
}

func jsonResult(v any) (*mcplib.CallToolResult, error) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return mcplib.NewToolResultErrorFromErr("marshal failed", err), nil
	}
	return mcplib.NewToolResultText(string(b)), nil
}
