package mcpsrv

import (
	"context"
	"fmt"
	"slices"

	mcplib "github.com/mark3labs/mcp-go/mcp"
	mcpsdk "github.com/mark3labs/mcp-go/server"

	"github.com/Eric-Fernald/chiton/internal/extract/subagent"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/retrieve"
)

// --- semantic_search: hybrid vector+graph retrieval ---

func semanticSearchTool() mcplib.Tool {
	return mcplib.NewTool("semantic_search",
		mcplib.WithDescription("Answer a natural-language question by semantic similarity, not just label substring. Embeds the query, seeds from the nearest nodes by meaning, expands the graph around them, and returns a ranked, token-budgeted subgraph. Use this for 'where do we handle X?' / 'what's the policy for Y?' questions; use query_graph when you know the exact name. Requires a prior 'chiton embed' run."),
		mcplib.WithString("query",
			mcplib.Required(),
			mcplib.Description("Natural-language query"),
		),
		mcplib.WithInteger("k",
			mcplib.DefaultNumber(8),
			mcplib.Description("Number of vector seeds before graph expansion (default 8)"),
			mcplib.Min(1),
			mcplib.Max(50),
		),
		mcplib.WithInteger("hops",
			mcplib.DefaultNumber(1),
			mcplib.Description("Graph expansion depth around each seed (default 1, max 3)"),
			mcplib.Min(0),
			mcplib.Max(3),
		),
		mcplib.WithInteger("max_tokens",
			mcplib.DefaultNumber(4000),
			mcplib.Description("Approximate token budget for the response (default 4000)"),
			mcplib.Min(500),
			mcplib.Max(16000),
		),
		mcplib.WithArray("edge_kinds",
			mcplib.Description("Filter graph expansion to these edge types e.g. [\"CALLS\",\"IMPORTS\"]. Empty = all kinds."),
			mcplib.WithStringItems(),
		),
		mcplib.WithString("source_prefix",
			mcplib.Description("Only return nodes whose source_uri contains this substring (scope to a package/corpus)"),
		),
		mcplib.WithBoolean("include_ambiguous",
			mcplib.Description("Include low-confidence (AMBIGUOUS) LLM-extracted nodes (default false)"),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func semanticSearchHandler(eng *retrieve.Engine, r redactor) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		if eng == nil {
			return mcplib.NewToolResultText("semantic search unavailable — no embeddings indexed. Run 'chiton embed' then restart 'chiton serve'."), nil
		}
		query, err := req.RequireString("query")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		res, err := eng.Query(ctx, query, retrieve.Options{
			SeedK:            clampInt(req.GetInt("k", 8), 1, 50),
			Hops:             clampInt(req.GetInt("hops", 1), 0, 3),
			MaxTokens:        clampInt(req.GetInt("max_tokens", 4000), 500, 16000),
			EdgeKinds:        parseEdgeKinds(req),
			SourcePrefix:     req.GetString("source_prefix", ""),
			IncludeAmbiguous: req.GetBool("include_ambiguous", false),
		})
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("semantic search failed", err), nil
		}

		// Lean output: strip embeddings and internal metadata, and drop nodes whose label names a person.
		type scoredLean struct {
			Score float32  `json:"score"`
			Hop   int      `json:"hop"`
			Node  leanNode `json:"node"`
		}
		labels := make(map[string]string, len(res.Nodes))
		nodes := make([]scoredLean, 0, len(res.Nodes))
		for _, s := range res.Nodes {
			if r.hidden(s.Node) {
				continue
			}
			labels[s.Node.ID] = s.Node.Label
			nodes = append(nodes, scoredLean{Score: s.Score, Hop: s.Hop, Node: r.node(s.Node)})
		}
		edges := slices.DeleteFunc(res.Edges, func(e model.Edge) bool {
			_, src := labels[e.Src]
			_, dst := labels[e.Dst]
			return !src || !dst
		})
		// Recount after redaction, so the summary can't reveal that hidden nodes matched.
		summary := res.Summary
		switch {
		case r.includePII || len(res.Nodes) == 0:
		case len(nodes) == 0:
			summary = "no semantic matches"
		default:
			summary = fmt.Sprintf("%d nodes, %d edges for %q", len(nodes), len(edges), query)
		}
		return jsonResult(struct {
			Nodes   []scoredLean `json:"nodes"`
			Edges   []leanEdge   `json:"edges"`
			Summary string       `json:"summary"`
		}{nodes, toLeanEdges(edges, labels), summary})
	}
}

// --- pending_extractions: emit subagent task specs ---

func pendingExtractionsTool() mcplib.Tool {
	return mcplib.NewTool("pending_extractions",
		mcplib.WithDescription("Return semantic-extraction task specs for doc sections and images that have not been processed yet. Run each task (extract concepts from the text, or describe the image), then call submit_extraction with the results. This is chiton's subagent extraction mode: chiton calls no LLM itself."),
		mcplib.WithString("kind",
			mcplib.Description("Filter to one task kind: doc_concepts | image_description (empty = both)"),
		),
		mcplib.WithInteger("limit",
			mcplib.DefaultNumber(20),
			mcplib.Description("Max task specs to return (default 20)"),
			mcplib.Min(1),
			mcplib.Max(200),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func pendingExtractionsHandler(svc *subagent.Service) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		if svc == nil {
			return mcplib.NewToolResultText("extraction unavailable — the server was started without a graph store."), nil
		}
		specs, err := svc.Pending(ctx, req.GetString("kind", ""), clampInt(req.GetInt("limit", 20), 1, 200))
		return pendingResult(specs, err, "pending_extractions",
			"no pending extractions — every doc section and image is already processed.")
	}
}

// --- submit_extraction: ingest subagent results ---

func submitExtractionTool() mcplib.Tool {
	return mcplib.NewTool("submit_extraction",
		mcplib.WithDescription("Ingest the results of one extraction task (from pending_extractions). Creates concept nodes and semantic edges tagged INFERRED (or AMBIGUOUS below the confidence threshold). Idempotent: re-submitting the same task is a no-op."),
		mcplib.WithString("task_id",
			mcplib.Required(),
			mcplib.Description("The task spec's id: a doc_section or image node ID from pending_extractions. Concepts that already exist are linked, never overwritten."),
		),
		mcplib.WithArray("concepts",
			mcplib.Required(),
			mcplib.Description("Extracted concepts: objects {label, description, confidence}"),
			mcplib.Items(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"label":       map[string]any{"type": "string"},
					"description": map[string]any{"type": "string"},
					"confidence":  map[string]any{"type": "number"},
				},
				"required": []string{"label"},
			}),
		),
		mcplib.WithArray("edges",
			mcplib.Description("Optional typed edges: objects {from, to, kind, confidence}. from empty = the source node; kind defaults to DESCRIBES."),
			mcplib.Items(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"from":       map[string]any{"type": "string"},
					"to":         map[string]any{"type": "string"},
					"kind":       map[string]any{"type": "string"},
					"confidence": map[string]any{"type": "number"},
				},
				"required": []string{"to"},
			}),
		),
	)
}

func submitExtractionHandler(svc *subagent.Service) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		if svc == nil {
			return mcplib.NewToolResultText("extraction unavailable — the server was started without a graph store."), nil
		}
		return bindAndSubmit(ctx, req, "submit_extraction", svc.Submit)
	}
}
