package mcpsrv

import (
	"context"

	mcplib "github.com/mark3labs/mcp-go/mcp"
	mcpsdk "github.com/mark3labs/mcp-go/server"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

func listFindingsTool() mcplib.Tool {
	return mcplib.NewTool("list_findings",
		mcplib.WithDescription("Return rule-engine findings filtered by category, severity, or target scope. Persisted by 'chiton analyze' as graph nodes."),
		mcplib.WithString("category",
			mcplib.Description("One of: architecture, security, bugs, performance, reliability, database, quality, style"),
		),
		mcplib.WithString("min_severity",
			mcplib.Description("Lower bound on severity: info|low|medium|high|critical"),
		),
		mcplib.WithString("scope",
			mcplib.Description("Substring filter against target_uri (e.g. 'internal/auth')"),
		),
		mcplib.WithInteger("limit",
			mcplib.DefaultNumber(50),
			mcplib.Description("Max findings to return (default 50)"),
			mcplib.Min(1),
			mcplib.Max(500),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func listFindingsHandler(gs store.GraphStore) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		views, err := analyze.List(ctx, gs, analyze.ListOptions{
			Category:    req.GetString("category", ""),
			MinSeverity: analyze.Severity(req.GetString("min_severity", "")),
			Scope:       req.GetString("scope", ""),
			Limit:       req.GetInt("limit", 50),
		})
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("list findings failed", err), nil
		}
		return jsonResult(views)
	}
}

func explainFindingTool() mcplib.Tool {
	return mcplib.NewTool("explain_finding",
		mcplib.WithDescription("Return a single finding plus its rule metadata and any related findings on the same target."),
		mcplib.WithString("id",
			mcplib.Required(),
			mcplib.Description("Finding ID (16-hex from list_findings) or partial rule label"),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func explainFindingHandler(gs store.GraphStore) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		id, err := req.RequireString("id")
		if err != nil {
			return mcplib.NewToolResultError(err.Error()), nil
		}
		exp, err := analyze.Explain(ctx, gs, id)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("explain failed", err), nil
		}
		return jsonResult(exp)
	}
}
