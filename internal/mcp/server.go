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

// New creates an MCPServer exposing four tools against the given graph store.
func New(gs store.GraphStore) *mcpsdk.MCPServer {
	s := mcpsdk.NewMCPServer("chiton", "0.1.0")
	s.AddTool(getNodeTool(), getNodeHandler(gs))
	s.AddTool(neighborhoodTool(), neighborhoodHandler(gs))
	s.AddTool(queryTool(), queryHandler(gs))
	s.AddTool(shortestPathTool(), shortestPathHandler(gs))
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
		nodes, err := gs.SearchNodes(ctx, label, 1)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search failed", err), nil
		}
		if len(nodes) == 0 {
			return mcplib.NewToolResultText(fmt.Sprintf("no node found matching %q", label)), nil
		}
		return jsonResult(toLeanNode(nodes[0]))
	}
}

func neighborhoodHandler(gs store.GraphStore) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		label, err := req.RequireString("label")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		depth := req.GetInt("depth", 2)
		kindStrs := req.GetStringSlice("edge_kinds", nil)

		nodes, err := gs.SearchNodes(ctx, label, 1)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search failed", err), nil
		}
		if len(nodes) == 0 {
			return mcplib.NewToolResultText(fmt.Sprintf("no node found matching %q", label)), nil
		}
		seed := nodes[0]

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
				len(sub.Nodes), len(sub.Edges), depth, seed.Label),
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
		nodes, err := gs.SearchNodes(ctx, query, limit)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("search failed", err), nil
		}
		if len(nodes) == 0 {
			return mcplib.NewToolResultText(fmt.Sprintf("no nodes found matching %q", query)), nil
		}
		lean := make([]leanNode, len(nodes))
		for i, n := range nodes {
			lean[i] = toLeanNode(n)
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

func toLeanEdge(e model.Edge, labels map[string]string) leanEdge {
	return leanEdge{
		Kind:  e.Kind,
		From:  labels[e.Src],
		To:    labels[e.Dst],
		Attrs: e.Attrs,
	}
}

func jsonResult(v any) (*mcplib.CallToolResult, error) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return mcplib.NewToolResultErrorFromErr("marshal failed", err), nil
	}
	return mcplib.NewToolResultText(string(b)), nil
}
